"""Layer 5 gate: the scoring service end to end.

Against a real ephemeral local `mlflow server` (sqlite backend, same philosophy
as Layer 4's gate) plus a real ephemeral Redis (testcontainers, same as Layer
3b's gate) and a real `bentoml serve` subprocess launched exactly the way
`conquer3.serving.supervisor` launches it -- no mocks standing in for either
MLflow or Redis.

This is the file that proves the plan's central claim: `scorer` **is** the
inference endpoint, remote MLflow is storage only, and killing MLflow entirely
leaves `/predict` serving at full correctness (the "not-a-proxy" gate). Under
BentoML that claim is stronger than it was: a worker process never imports a
tracking URI at all -- the supervisor resolves the champion and pins the version
in a pointer file, and workers load only from the local artifact cache.
"""

from __future__ import annotations

import json
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

_STARTUP_TIMEOUT_S = 60.0

# How long a champion promotion may refuse connections. The restart is a real
# cutover window (SIGTERM drain + BentoML boot + model load); this is the ceiling
# the gate holds it to, and the number the README's Layer 5 row quotes.
_MAX_CUTOVER_S = 25.0


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
                "C3_ACTIVE_CHAMPION_FILE": str(tmp_path / "active.json"),
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
    """A live scoring server this test process can call and later tear down.
    Backs both launch styles below -- the raw `bentoml serve` subprocess (no
    reload) and the real `conquer3 serve` supervisor (with reload) -- since both
    expose the identical HTTP surface."""

    def __init__(self, proc: subprocess.Popen[bytes], port: int) -> None:
        self.proc = proc
        self.port = port

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def predict(
        self, records: list[dict[str, Any]], *, dry_run: bool | None = None
    ) -> httpx.Response:
        payload: dict[str, Any] = {"transactions": records}
        if dry_run is not None:
            payload["dry_run"] = dry_run
        return httpx.post(self.url("/predict"), json=payload, timeout=15)

    def model_info(self) -> httpx.Response:
        return httpx.post(self.url("/model_info"), json={}, timeout=15)

    def invocations(
        self, records: list[dict[str, Any]], *, params: dict[str, Any] | None = None
    ) -> httpx.Response:
        payload: dict[str, Any] = {"dataframe_records": records}
        if params is not None:
            payload["params"] = params
        return httpx.post(self.url("/invocations"), json=payload, timeout=15)

    def stop(self) -> None:
        with suppress(ProcessLookupError):
            self.proc.send_signal(signal.SIGTERM)
        with suppress(subprocess.TimeoutExpired):
            self.proc.wait(timeout=20)


def _launch_bentoml_directly(*, workers: int = 1) -> _RunningServer:
    """Launches `bentoml serve` the same way supervisor._spawn() does, but without
    the champion-poll thread or signal forwarding -- for tests that only care about
    request behavior against a fixed, already-activated champion."""
    port = _free_port()
    import os

    env = {**os.environ, "C3_SCORER_WORKERS": str(workers)}
    proc = subprocess.Popen(
        [
            "bentoml",
            "serve",
            "conquer3.serving.service:FraudScorerService",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
    )
    assert _wait_http_ok(f"http://127.0.0.1:{port}/readyz", _STARTUP_TIMEOUT_S), (
        "bentoml server did not become ready in time"
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
    # delta dict can't even find `python`/`bentoml`.
    full_env = {**os.environ, **env, "C3_SCORER_PORT": str(port), "C3_SCORER_HOST": "127.0.0.1"}
    if extra_env:
        full_env.update(extra_env)
    proc = subprocess.Popen([sys.executable, "-m", "conquer3.cli", "serve"], env=full_env)
    assert _wait_http_ok(f"http://127.0.0.1:{port}/readyz", _STARTUP_TIMEOUT_S), (
        "conquer3 serve did not become ready in time"
    )
    return _RunningServer(proc, port)


def test_score_endpoint_covers_the_layer5_gate_properties(
    mlflow_and_redis: MlflowAndRedis,
) -> None:
    from conquer3.serving.champion import activate_champion

    ref = _publish_dummy(model_name="gate_scorer_model", code_sha="v1")
    active_ref = activate_champion(get_settings())
    assert active_ref.version == ref.version

    server = _launch_bentoml_directly()
    try:
        # -- back-to-back same-account calls: zero-lag state carries over ------
        r1 = server.predict([_txn_record(event_id="e1", event_ts_us=1_700_000_000_000_000)])
        assert r1.status_code == 200
        resp1 = r1.json()[0]
        assert resp1["had_prev_state"] is False
        assert resp1["seconds_since_last_txn"] is None
        assert resp1["model_version"] == ref.version
        assert resp1["degraded"] is False

        r2 = server.predict([_txn_record(event_id="e2", event_ts_us=1_700_000_005_000_000)])
        assert r2.status_code == 200
        resp2 = r2.json()[0]
        assert resp2["had_prev_state"] is True
        assert resp2["seconds_since_last_txn"] == pytest.approx(5.0)

        # -- the same two transactions as ONE two-row batch, fresh account: ----
        # -- identical result -- proves in-batch per-account sequencing --------
        batch = server.predict(
            [
                _txn_record(event_id="b1", account_id="C2", event_ts_us=1_700_000_000_000_000),
                _txn_record(event_id="b2", account_id="C2", event_ts_us=1_700_000_005_000_000),
            ]
        )
        assert batch.status_code == 200
        b1, b2 = batch.json()
        assert b1["had_prev_state"] is False
        assert b2["had_prev_state"] is True
        assert b2["seconds_since_last_txn"] == pytest.approx(5.0)
        assert b1["fraud_score"] == pytest.approx(resp1["fraud_score"])
        assert b2["fraud_score"] == pytest.approx(resp2["fraud_score"])

        # -- /model_info is a real route now: no body, no placeholder row ------
        info = server.model_info()
        assert info.status_code == 200
        assert info.json()["version"] == ref.version
        assert info.json()["name"] == "gate_scorer_model"
        assert info.json()["degraded"] is False

        # -- dry_run: reads current state (so the score is realistic) but ------
        # -- leaves Redis and the event dir untouched ---------------------------
        event_files = sorted(Path(mlflow_and_redis.env["C3_EVENT_DIR"]).rglob("*.jsonl"))
        lines_before = sum(p.read_text().count("\n") for p in event_files)

        dry = server.predict(
            [_txn_record(event_id="e3-dry", event_ts_us=1_700_000_010_000_000)], dry_run=True
        )
        assert dry.status_code == 200
        dry_resp = dry.json()[0]
        assert dry_resp["had_prev_state"] is True
        assert dry_resp["seconds_since_last_txn"] == pytest.approx(5.0)  # still vs e2

        event_files_after = sorted(Path(mlflow_and_redis.env["C3_EVENT_DIR"]).rglob("*.jsonl"))
        lines_after = sum(p.read_text().count("\n") for p in event_files_after)
        assert lines_after == lines_before, "dry_run must not append an event"

        # A REAL follow-up call must still see e2 (not the dry-run row) as its
        # predecessor -- proves the dry_run left no trace in Redis either.
        r4 = server.predict([_txn_record(event_id="e4", event_ts_us=1_700_000_015_000_000)])
        resp4 = r4.json()[0]
        assert resp4["seconds_since_last_txn"] == pytest.approx(
            10.0
        )  # vs e2 (t+5), not dry e3 (t+10)
    finally:
        server.stop()


def test_openapi_spec_describes_the_whole_request_and_response_contract(
    mlflow_and_redis: MlflowAndRedis,
) -> None:
    """The point of the migration: a served OpenAPI document that actually
    describes the payloads, not an untyped raw-body endpoint. Asserted against the
    *running* server, not the in-process class, so a serving-time regression
    (wrong mount path, spec not published) fails here."""
    from conquer3.serving.api_models import TXN_FIELD_NAMES
    from conquer3.serving.champion import activate_champion

    _publish_dummy(model_name="gate_scorer_model", code_sha="v1")
    activate_champion(get_settings())

    server = _launch_bentoml_directly()
    try:
        spec = httpx.get(server.url("/docs.json"), timeout=15).json()
        assert spec["openapi"].startswith("3.")
        for route in ("/predict", "/model_info", "/invocations"):
            assert route in spec["paths"], route

        body = spec["paths"]["/predict"]["post"]["requestBody"]
        schema = body["content"]["application/json"]["schema"]
        assert set(schema["properties"]) == {"transactions", "dry_run"}

        txn = spec["components"]["schemas"]["TransactionIn"]
        assert list(txn["properties"]) == list(TXN_FIELD_NAMES)
        assert set(txn["required"]) == set(TXN_FIELD_NAMES)
        # Every field documented -- an empty description ships a blank API doc row.
        assert all(txn["properties"][name].get("description") for name in TXN_FIELD_NAMES)

        result = spec["components"]["schemas"]["ScoreResult"]
        assert set(result["properties"]) == {
            "event_id",
            "fraud_score",
            "decision",
            "had_prev_state",
            "seconds_since_last_txn",
            "model_version",
            "feature_schema_version",
            "degraded",
        }

        # The compatibility route is published as deprecated, in the document a
        # client actually reads -- not only in a source comment.
        assert "DEPRECATED" in spec["paths"]["/invocations"]["post"]["description"]

        # A malformed request is rejected by the schema, before any scoring runs.
        bad = httpx.post(
            server.url("/predict"), json={"transactions": [{"event_id": "only"}]}, timeout=15
        )
        assert bad.status_code == 400
    finally:
        server.stop()


def test_invocations_alias_matches_predict_exactly(mlflow_and_redis: MlflowAndRedis) -> None:
    """The deprecated MLflow envelope is an adapter over /predict, not a second
    implementation. Same inputs must produce the same numbers, and `op=model_info`
    must agree with /model_info."""
    from conquer3.serving.champion import activate_champion

    ref = _publish_dummy(model_name="gate_scorer_model", code_sha="v1")
    activate_champion(get_settings())

    server = _launch_bentoml_directly()
    try:
        # dry_run on both sides so neither call mutates state the other would see.
        records = [
            _txn_record(event_id="x1", account_id="CX", event_ts_us=1_700_000_000_000_000),
            _txn_record(event_id="x2", account_id="CX", event_ts_us=1_700_000_005_000_000),
        ]
        typed = server.predict(records, dry_run=True)
        legacy = server.invocations(records, params={"dry_run": True})
        assert typed.status_code == 200
        assert legacy.status_code == 200
        assert legacy.json()["predictions"] == typed.json()

        legacy_info = server.invocations(
            [_txn_record(event_id="_info", event_ts_us=0)], params={"op": "model_info"}
        )
        assert legacy_info.status_code == 200
        assert legacy_info.json()["predictions"][0]["version"] == ref.version
        assert legacy_info.json()["predictions"] == [server.model_info().json()]

        # An unknown op is still a client error, not a silently-scored request.
        bogus = server.invocations(
            [_txn_record(event_id="_bad", event_ts_us=0)], params={"op": "bogus"}
        )
        assert bogus.status_code >= 400
    finally:
        server.stop()


def test_concurrent_same_account_requests_never_corrupt_state(
    mlflow_and_redis: MlflowAndRedis,
) -> None:
    """Fires concurrent requests for the same account through the real server --
    the exact scenario the monotonic CAS exists for (plan §8.4). Every response
    must be well-formed; the account's final Redis state must be internally
    consistent (not a torn write), even though which specific requests "win" the
    race is unspecified.
    """
    from concurrent.futures import ThreadPoolExecutor

    import redis as redis_lib

    from conquer3.serving.champion import activate_champion

    _publish_dummy(model_name="gate_scorer_model", code_sha="v1")
    activate_champion(get_settings())

    server = _launch_bentoml_directly()
    try:
        base = 1_700_000_000_000_000

        def _fire(i: int) -> httpx.Response:
            return server.predict([_txn_record(event_id=f"c{i}", event_ts_us=base + i * 1_000_000)])

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
    healthy, killing remote MLflow entirely must leave /predict serving at full
    correctness. No client request may reach, or depend on, remote MLflow.
    """
    from conquer3.serving.champion import activate_champion

    ref = _publish_dummy(model_name="gate_scorer_model", code_sha="v1")
    activate_champion(get_settings())

    server = _launch_bentoml_directly()
    try:
        r_before = server.predict(
            [_txn_record(event_id="before", event_ts_us=1_700_000_000_000_000)]
        )
        assert r_before.status_code == 200

        # Genuinely sever remote MLflow -- not a redirected env var, the actual
        # ephemeral server process this test was talking to.
        mlflow_and_redis.kill_mlflow()

        for i in range(5):
            r = server.predict(
                [_txn_record(event_id=f"after{i}", event_ts_us=1_700_000_010_000_000 + i)]
            )
            assert r.status_code == 200
            resp = r.json()[0]
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
    from conquer3.serving.champion import activate_champion

    ref = _publish_dummy(model_name="gate_scorer_model", code_sha="v1")
    # First boot: live, populates the champion cache + artifact cache.
    activate_champion(get_settings())

    # Genuinely sever remote MLflow before the second boot attempt.
    mlflow_and_redis.kill_mlflow()
    get_settings.cache_clear()

    # Second "boot": MLflow is dead, must fall back to the cache.
    degraded_ref = activate_champion(get_settings())
    assert degraded_ref.degraded is True
    assert degraded_ref.version == ref.version

    server = _launch_bentoml_directly()
    try:
        r = server.predict([_txn_record(event_id="e1", event_ts_us=1_700_000_000_000_000)])
        assert r.status_code == 200
        resp = r.json()[0]
        assert resp["degraded"] is True
        assert resp["model_version"] == ref.version
    finally:
        server.stop()


def test_promotion_reloads_within_one_poll_interval_with_no_error_responses(
    mlflow_and_redis: MlflowAndRedis,
) -> None:
    """The real `conquer3 serve` supervisor: boots on v1, a new champion is
    promoted, and within a short poll interval /model_info reports v2.

    Reload is a child restart, so unlike the previous MLflow/SIGHUP implementation
    there IS a cutover window in which connections are refused. Two things are
    asserted about it, and they are the gate:

    * **No request ever receives an HTTP error status.** A 4xx/5xx would mean a
      broken model was served; a refused connection means the server was down.
      Those are different failures and only the second one is tolerated here.
    * **The outage is bounded** by `_MAX_CUTOVER_S`, measured from the first
      refusal to the first success after it.
    """
    ref1 = _publish_dummy(model_name="gate_scorer_model", code_sha="v1")

    supervisor = _launch_supervisor(mlflow_and_redis.env, extra_env={"C3_CHAMPION_POLL_S": "2"})
    try:
        assert supervisor.model_info().json()["version"] == ref1.version

        ref2 = _publish_dummy(model_name="gate_scorer_model", code_sha="v2")
        assert ref2.version != ref1.version

        deadline = time.monotonic() + 120
        statuses: list[int] = []
        outage_start: float | None = None
        max_outage = 0.0
        seen_version = ref1.version

        while time.monotonic() < deadline and seen_version != ref2.version:
            try:
                r = supervisor.model_info()
            except httpx.TransportError:
                # Connection refused: the restart's cutover window. Time it.
                if outage_start is None:
                    outage_start = time.monotonic()
            else:
                statuses.append(r.status_code)
                if outage_start is not None:
                    max_outage = max(max_outage, time.monotonic() - outage_start)
                    outage_start = None
                if r.status_code == 200:
                    seen_version = r.json()["version"]
            time.sleep(0.25)

        assert seen_version == ref2.version, "champion poll never reloaded to the new version"
        assert all(200 <= s < 300 for s in statuses), f"error responses during reload: {statuses}"
        assert max_outage <= _MAX_CUTOVER_S, (
            f"cutover window {max_outage:.1f}s exceeded the {_MAX_CUTOVER_S}s budget"
        )

        r_final = supervisor.predict([_txn_record(event_id="post", event_ts_us=1)])
        assert r_final.status_code == 200
        assert r_final.json()[0]["model_version"] == ref2.version
    finally:
        supervisor.stop()


def test_supervisor_pins_the_version_workers_load(mlflow_and_redis: MlflowAndRedis) -> None:
    """The supervisor owns registry contact; workers read a pointer file and the
    local artifact cache. That split is what keeps a restarting worker from
    independently resolving a different champion than the one just recorded."""
    from conquer3.serving.champion import activate_champion, read_active_ref

    ref = _publish_dummy(model_name="gate_scorer_model", code_sha="v1")
    activate_champion(get_settings())

    active_path = Path(mlflow_and_redis.env["C3_ACTIVE_CHAMPION_FILE"])
    assert active_path.is_file()
    assert json.loads(active_path.read_text())["version"] == ref.version
    assert read_active_ref(get_settings()).version == ref.version
