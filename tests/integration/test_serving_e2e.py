"""Layer 5 gate: the scoring service end to end.

Against a real ephemeral local `mlflow server` (sqlite backend, same philosophy
as Layer 4's gate) plus a real ephemeral Redis (testcontainers, same as Layer
3b's gate) and the actual `mlflow.pyfunc.scoring_server` subprocess launched
exactly the way `conquer3.serving.supervisor` launches it -- no mocks standing in
for either MLflow or Redis.

This is the file that proves the plan's central claim: `scorer` **is** the
inference endpoint, remote MLflow is storage only, and killing MLflow entirely
leaves `/invocations` serving at full correctness (the "not-a-proxy" gate).
"""

from __future__ import annotations

import signal
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("testcontainers")

import httpx
from testcontainers.community.redis import RedisContainer

from conquer3.config.settings import get_settings

pytestmark = [pytest.mark.integration, pytest.mark.mlflow]

_STARTUP_TIMEOUT_S = 40.0


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_http_ok(url: str, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(0.2)
    return False


@dataclass
class MlflowAndRedis:
    env: dict[str, str]
    _mlflow_proc: subprocess.Popen[bytes]

    def kill_mlflow(self) -> None:
        """Actually terminates the ephemeral mlflow server mid-test -- for the
        not-a-proxy / degraded-boot tests, which must prove behavior against a
        genuinely dead remote, not just a redirected env var no one reads."""
        self._mlflow_proc.terminate()
        self._mlflow_proc.wait(timeout=15)


@pytest.fixture
def mlflow_and_redis(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[MlflowAndRedis]:
    """Ephemeral local `mlflow server` (sqlite backend, local artifact root) +
    ephemeral Redis (testcontainers), wired into conquer3's Settings via env
    vars."""
    port = _free_port()
    backend = tmp_path / "mlflow.db"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    mlflow_uri = f"http://127.0.0.1:{port}"

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--backend-store-uri",
            f"sqlite:///{backend}",
            "--default-artifact-root",
            str(artifacts),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        with RedisContainer("redis:7-alpine") as redis_c:
            assert _wait_http_ok(mlflow_uri + "/health", _STARTUP_TIMEOUT_S), (
                "mlflow server did not become healthy in time"
            )

            env = {
                "MLFLOW_TRACKING_URI": mlflow_uri,
                "C3_MODEL_NAME": "gate_scorer_model",
                "C3_MODEL_CACHE_DIR": str(tmp_path / "modelcache"),
                "C3_MODEL_CHAMPION_CACHE_FILE": str(tmp_path / "champion.json"),
                "C3_WRAPPED_MODEL_DIR": str(tmp_path / "wrapped"),
                "C3_CURRENT_MODEL_SYMLINK": str(tmp_path / "current"),
                "REDIS_HOST": redis_c.get_container_host_ip(),
                "REDIS_PORT": str(redis_c.get_exposed_port(6379)),
                "C3_EVENT_DIR": str(tmp_path / "events"),
                # Long by default -- only the reload test shortens it, so other
                # tests never race a background poll they don't care about.
                "C3_CHAMPION_POLL_S": "3600",
            }
            for key, value in env.items():
                monkeypatch.setenv(key, value)
            get_settings.cache_clear()
            try:
                yield MlflowAndRedis(env=env, _mlflow_proc=proc)
            finally:
                get_settings.cache_clear()
    finally:
        # kill_mlflow() may have already terminated + reaped this process --
        # terminate()/wait() on an already-exited Popen is a documented no-op.
        proc.terminate()
        proc.wait(timeout=15)


def _publish_dummy(*, model_name: str, code_sha: str) -> Any:
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.dummy import DummyClassifier

    from conquer3.contracts.model_registry import publish_model
    from conquer3.core.schema import CATEGORICAL_FEATURES, FEATURE_NAMES, NUMERIC_FEATURES

    rng = np.random.default_rng(0)
    n = 20
    data: dict[str, object] = {name: rng.normal(size=n) for name in NUMERIC_FEATURES}
    for name in CATEGORICAL_FEATURES:
        data[name] = rng.choice(["a", "b"], size=n)
    x_sample = pd.DataFrame(data, columns=list(FEATURE_NAMES))
    y = rng.integers(0, 2, size=n)
    clf = DummyClassifier(strategy="prior").fit(x_sample, y)
    proba = clf.predict_proba(x_sample)
    return publish_model(
        clf,
        x_sample,
        proba,
        sklearn_version=sklearn.__version__,
        code_sha=code_sha,
        decision_threshold=0.5,
        model_name=model_name,
        alias_as_champion=True,
    )


def _txn_record(
    *,
    event_id: str,
    event_ts_us: int,
    account_id: str = "C1",
    dest_id: str = "C900",
    txn_type: str = "TRANSFER",
    amount: float = 100.0,
    oldbalance_org: float = 1000.0,
    newbalance_orig: float = 900.0,
    oldbalance_dest: float = 0.0,
    newbalance_dest: float = 100.0,
    step: int = 1,
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "account_id": account_id,
        "dest_id": dest_id,
        "txn_type": txn_type,
        "amount": amount,
        "oldbalance_org": oldbalance_org,
        "newbalance_orig": newbalance_orig,
        "oldbalance_dest": oldbalance_dest,
        "newbalance_dest": newbalance_dest,
        "event_ts_us": event_ts_us,
        "step": step,
    }


class _RunningServer:
    """A live scoring server this test process can POST /invocations at and
    later tear down. Backs both launch styles below -- the raw get_cmd()
    subprocess (no reload) and the real `conquer3 serve` supervisor (with
    reload) -- since both expose the identical HTTP surface."""

    def __init__(self, proc: subprocess.Popen[bytes], port: int) -> None:
        self.proc = proc
        self.port = port

    def invoke(
        self, records: list[dict[str, Any]], *, params: dict[str, Any] | None = None
    ) -> httpx.Response:
        payload: dict[str, Any] = {"dataframe_records": records}
        if params is not None:
            payload["params"] = params
        return httpx.post(f"http://127.0.0.1:{self.port}/invocations", json=payload, timeout=10)

    def stop(self) -> None:
        with suppress(ProcessLookupError):
            self.proc.send_signal(signal.SIGTERM)
        with suppress(subprocess.TimeoutExpired):
            self.proc.wait(timeout=15)


def _launch_scoring_server_directly(*, nworkers: int = 1) -> _RunningServer:
    """Launches the scoring server the same way supervisor.serve() does
    (mlflow.pyfunc.scoring_server.get_cmd + `bash -c "exec ..."`), but without
    the champion-poll thread or signal forwarding -- for tests that only care
    about /invocations behavior against a fixed, already-active champion."""
    import mlflow.pyfunc.scoring_server as scoring_server

    settings = get_settings()
    port = _free_port()
    command, env = scoring_server.get_cmd(
        model_uri=settings.serving.current_model_symlink,
        port=port,
        host="127.0.0.1",  # type: ignore[arg-type]
        timeout=30,
        nworkers=nworkers,
    )
    proc = subprocess.Popen(["bash", "-c", "exec " + command], env=env)
    assert _wait_http_ok(f"http://127.0.0.1:{port}/ping", _STARTUP_TIMEOUT_S), (
        "scoring server did not become ready in time"
    )
    return _RunningServer(proc, port)


def _launch_supervisor(
    env: dict[str, str], *, extra_env: dict[str, str] | None = None
) -> _RunningServer:
    """Launches the REAL `conquer3 serve` entrypoint as its own process (needed
    because supervisor.serve() installs signal handlers, which only Python's
    main thread may do -- so it must run in its own process, not a thread of
    this test process, mirroring how test_pathway_streaming.py launches
    `conquer3 pathway streaming` for its own single-call-per-process reason)."""
    import os

    port = _free_port()
    # Start from the real OS environment (PATH, HOME, ...) -- env/extra_env are
    # deltas, not a complete environment; a subprocess launched with only the
    # delta dict can't even find `python`/`bash`.
    full_env = {**os.environ, **env, "C3_SCORER_PORT": str(port), "C3_SCORER_HOST": "127.0.0.1"}
    full_env.setdefault("C3_SCORER_WORKERS", "2")  # SIGHUP reload needs >1
    if extra_env:
        full_env.update(extra_env)
    proc = subprocess.Popen([sys.executable, "-m", "conquer3.cli", "serve"], env=full_env)
    assert _wait_http_ok(f"http://127.0.0.1:{port}/ping", _STARTUP_TIMEOUT_S), (
        "conquer3 serve did not become ready in time"
    )
    return _RunningServer(proc, port)


def _model_info(server: _RunningServer) -> dict[str, Any]:
    r = server.invoke([_txn_record(event_id="_info", event_ts_us=0)], params={"op": "model_info"})
    r.raise_for_status()
    result: dict[str, Any] = r.json()["predictions"][0]
    return result


def test_score_endpoint_covers_the_layer5_gate_properties(
    mlflow_and_redis: MlflowAndRedis,
) -> None:
    from conquer3.serving.build import build_and_activate_champion

    ref = _publish_dummy(model_name="gate_scorer_model", code_sha="v1")
    built_ref = build_and_activate_champion(get_settings())
    assert built_ref.version == ref.version

    server = _launch_scoring_server_directly(nworkers=1)
    try:
        # -- back-to-back same-account calls: zero-lag state carries over ------
        r1 = server.invoke([_txn_record(event_id="e1", event_ts_us=1_700_000_000_000_000)])
        assert r1.status_code == 200
        resp1 = r1.json()["predictions"][0]
        assert resp1["had_prev_state"] is False
        assert resp1["seconds_since_last_txn"] is None
        assert resp1["model_version"] == ref.version
        assert resp1["degraded"] is False

        r2 = server.invoke([_txn_record(event_id="e2", event_ts_us=1_700_000_005_000_000)])
        assert r2.status_code == 200
        resp2 = r2.json()["predictions"][0]
        assert resp2["had_prev_state"] is True
        assert resp2["seconds_since_last_txn"] == pytest.approx(5.0)

        # -- the same two transactions as ONE two-row batch, fresh account: ----
        # -- identical result -- proves in-batch per-account sequencing --------
        batch = server.invoke(
            [
                _txn_record(event_id="b1", account_id="C2", event_ts_us=1_700_000_000_000_000),
                _txn_record(event_id="b2", account_id="C2", event_ts_us=1_700_000_005_000_000),
            ]
        )
        assert batch.status_code == 200
        b1, b2 = batch.json()["predictions"]
        assert b1["had_prev_state"] is False
        assert b2["had_prev_state"] is True
        assert b2["seconds_since_last_txn"] == pytest.approx(5.0)
        assert b1["fraud_score"] == pytest.approx(resp1["fraud_score"])
        assert b2["fraud_score"] == pytest.approx(resp2["fraud_score"])

        # -- op=model_info: any syntactically valid row works (schema ----------
        # -- enforcement runs before predict, regardless of op) -----------------
        info = _model_info(server)
        assert info["version"] == ref.version
        assert info["name"] == "gate_scorer_model"
        assert info["degraded"] is False

        # -- dry_run: reads current state (so the score is realistic) but ------
        # -- leaves Redis and the event dir untouched ---------------------------
        event_files = sorted(Path(mlflow_and_redis.env["C3_EVENT_DIR"]).rglob("*.jsonl"))
        lines_before = sum(p.read_text().count("\n") for p in event_files)

        dry = server.invoke(
            [_txn_record(event_id="e3-dry", event_ts_us=1_700_000_010_000_000)],
            params={"dry_run": True},
        )
        assert dry.status_code == 200
        dry_resp = dry.json()["predictions"][0]
        assert dry_resp["had_prev_state"] is True
        assert dry_resp["seconds_since_last_txn"] == pytest.approx(5.0)  # still vs e2

        event_files_after = sorted(Path(mlflow_and_redis.env["C3_EVENT_DIR"]).rglob("*.jsonl"))
        lines_after = sum(p.read_text().count("\n") for p in event_files_after)
        assert lines_after == lines_before, "dry_run must not append an event"

        # A REAL follow-up call must still see e2 (not the dry-run row) as its
        # predecessor -- proves the dry_run left no trace in Redis either.
        r4 = server.invoke([_txn_record(event_id="e4", event_ts_us=1_700_000_015_000_000)])
        resp4 = r4.json()["predictions"][0]
        assert resp4["seconds_since_last_txn"] == pytest.approx(
            10.0
        )  # vs e2 (t+5), not dry e3 (t+10)
    finally:
        server.stop()


def test_concurrent_same_account_requests_never_corrupt_state(
    mlflow_and_redis: MlflowAndRedis,
) -> None:
    """Fires concurrent requests for the same account through the real
    asyncio.to_thread dispatch inside one worker -- the exact scenario the
    monotonic CAS exists for (plan §8.4). Every response must be well-formed;
    the account's final Redis state must be internally consistent (not a torn
    write), even though which specific requests "win" the race is unspecified.
    """
    from concurrent.futures import ThreadPoolExecutor

    import redis as redis_lib

    from conquer3.serving.build import build_and_activate_champion

    _publish_dummy(model_name="gate_scorer_model", code_sha="v1")
    build_and_activate_champion(get_settings())

    server = _launch_scoring_server_directly(nworkers=1)
    try:
        base = 1_700_000_000_000_000

        def _fire(i: int) -> httpx.Response:
            return server.invoke([_txn_record(event_id=f"c{i}", event_ts_us=base + i * 1_000_000)])

        with ThreadPoolExecutor(max_workers=16) as pool:
            responses = list(pool.map(_fire, range(16)))

        assert all(r.status_code == 200 for r in responses)

        r_client = redis_lib.Redis(
            host=mlflow_and_redis.env["REDIS_HOST"], port=int(mlflow_and_redis.env["REDIS_PORT"])
        )
        raw = r_client.get("c3:acct:v1:C1")
        assert raw is not None
        from conquer3.core.serde import state_from_json

        state = state_from_json(raw)
        assert state is not None  # a torn/corrupt write would come back as None
        assert state.txn_count >= 1
    finally:
        server.stop()


def test_not_a_proxy_survives_a_dead_remote_mlflow(mlflow_and_redis: MlflowAndRedis) -> None:
    """The core claim of the architecture: with the scorer already booted and
    healthy, killing remote MLflow entirely must leave /invocations serving at
    full correctness. No client request may reach, or depend on, remote MLflow.
    """
    from conquer3.serving.build import build_and_activate_champion

    ref = _publish_dummy(model_name="gate_scorer_model", code_sha="v1")
    build_and_activate_champion(get_settings())

    server = _launch_scoring_server_directly(nworkers=1)
    try:
        r_before = server.invoke(
            [_txn_record(event_id="before", event_ts_us=1_700_000_000_000_000)]
        )
        assert r_before.status_code == 200

        # Genuinely sever remote MLflow -- not a redirected env var, the actual
        # ephemeral server process this test was talking to.
        mlflow_and_redis.kill_mlflow()

        for i in range(5):
            r = server.invoke(
                [_txn_record(event_id=f"after{i}", event_ts_us=1_700_000_010_000_000 + i)]
            )
            assert r.status_code == 200
            resp = r.json()["predictions"][0]
            assert resp["model_version"] == ref.version
            assert (
                resp["degraded"] is False
            )  # boot succeeded live; killing MLflow after boot changes nothing
    finally:
        server.stop()


def test_degraded_boot_serves_from_cache_when_mlflow_is_dead(
    mlflow_and_redis: MlflowAndRedis,
) -> None:
    """A boot attempt that starts with remote MLflow already unreachable must
    still succeed, from the cached champion + cached artifact -- and every
    response must report degraded=True."""
    from conquer3.serving.build import build_and_activate_champion

    ref = _publish_dummy(model_name="gate_scorer_model", code_sha="v1")
    # First boot: live, populates the champion cache + artifact cache.
    build_and_activate_champion(get_settings())

    # Genuinely sever remote MLflow before the second boot attempt.
    mlflow_and_redis.kill_mlflow()
    get_settings.cache_clear()

    # Second "boot": MLflow is dead, must fall back to the cache.
    degraded_ref = build_and_activate_champion(get_settings())
    assert degraded_ref.degraded is True
    assert degraded_ref.version == ref.version

    server = _launch_scoring_server_directly(nworkers=1)
    try:
        r = server.invoke([_txn_record(event_id="e1", event_ts_us=1_700_000_000_000_000)])
        assert r.status_code == 200
        resp = r.json()["predictions"][0]
        assert resp["degraded"] is True
        assert resp["model_version"] == ref.version
    finally:
        server.stop()


def test_promotion_triggers_reload_within_one_poll_interval_with_zero_5xx(
    mlflow_and_redis: MlflowAndRedis,
) -> None:
    """The real `conquer3 serve` supervisor: boots on v1, a new champion is
    promoted, and within a short poll interval /invocations reports v2 -- with
    zero non-2xx responses observed across the transition (the reload must be a
    clean cutover, never a window of failures)."""
    ref1 = _publish_dummy(model_name="gate_scorer_model", code_sha="v1")

    supervisor = _launch_supervisor(mlflow_and_redis.env, extra_env={"C3_CHAMPION_POLL_S": "2"})
    try:
        info = _model_info(supervisor)
        assert info["version"] == ref1.version

        ref2 = _publish_dummy(model_name="gate_scorer_model", code_sha="v2")
        assert ref2.version != ref1.version

        # Generous margin over C3_CHAMPION_POLL_S=2s: a poll tick also does a
        # full wrapper rebuild (mlflow.pyfunc.save_model, ~2-4s) before SIGHUP,
        # and this test runs slower alongside the rest of the suite's own
        # mlflow-server/uvicorn subprocess churn than it does in isolation.
        deadline = time.monotonic() + 45
        statuses: list[int] = []
        seen_version = ref1.version
        while time.monotonic() < deadline and seen_version != ref2.version:
            r = supervisor.invoke(
                [_txn_record(event_id="poll", event_ts_us=0)], params={"op": "model_info"}
            )
            statuses.append(r.status_code)
            if r.status_code == 200:
                seen_version = r.json()["predictions"][0]["version"]
            time.sleep(0.5)

        assert seen_version == ref2.version, "champion poll never reloaded to the new version"
        assert all(200 <= s < 300 for s in statuses), f"non-2xx during reload: {statuses}"

        r_final = supervisor.invoke([_txn_record(event_id="post", event_ts_us=1)])
        assert r_final.status_code == 200
        assert r_final.json()["predictions"][0]["model_version"] == ref2.version
    finally:
        supervisor.stop()
