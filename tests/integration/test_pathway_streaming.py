"""Layer 3b gate: streaming state repair against ephemeral Postgres + Redis
containers -- pickup latency and kill/restart deduplication.

Launches `conquer3 pathway streaming` as a background subprocess (required
because Pathway's engine only supports calling ``pw.run()`` once per process),
points it at a tmp staging dir + tmp persist dir, and drives it by dropping JSONL
files into the staging dir while polling Redis.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import redis as redis_lib

pytest.importorskip("testcontainers")

from testcontainers.community.postgres import PostgresContainer
from testcontainers.community.redis import RedisContainer

from conquer3.config.settings import get_settings
from conquer3.core.serde import redis_state_key
from conquer3.db.bootstrap import apply_ddl
from conquer3.db.engine import pg_connection

pytestmark = [pytest.mark.integration, pytest.mark.pathway]

_PICKUP_TIMEOUT_S = 5.0
_STARTUP_GRACE_S = 3.0


@pytest.fixture
def warehouse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[tuple[dict[str, str], redis_lib.Redis]]:
    """Ephemeral Postgres + Redis, with the medallion DDL applied. Yields
    (pg_env, redis_client) -- pg_env for the streaming subprocess's environment,
    a ready-made client for asserting against Redis directly.
    """
    with (
        PostgresContainer("postgres:17-alpine") as pg,
        RedisContainer("redis:7-alpine") as redis_c,
    ):
        monkeypatch.setenv("POSTGRES_HOST", pg.get_container_host_ip())
        monkeypatch.setenv("POSTGRES_PORT", str(pg.get_exposed_port(5432)))
        monkeypatch.setenv("POSTGRES_DB", pg.dbname)
        monkeypatch.setenv("POSTGRES_USER", pg.username)
        monkeypatch.setenv("POSTGRES_PASSWORD", pg.password)
        get_settings.cache_clear()
        try:
            with pg_connection() as conn:
                apply_ddl(conn)

            pg_env = {
                "POSTGRES_HOST": pg.get_container_host_ip(),
                "POSTGRES_PORT": str(pg.get_exposed_port(5432)),
                "POSTGRES_DB": pg.dbname,
                "POSTGRES_USER": pg.username,
                "POSTGRES_PASSWORD": pg.password,
                "REDIS_HOST": redis_c.get_container_host_ip(),
                "REDIS_PORT": str(redis_c.get_exposed_port(6379)),
            }
            yield pg_env, redis_c.get_client()
        finally:
            get_settings.cache_clear()


def _write_jsonl(staging_dir: Path, name: str, rows: list[dict[str, object]]) -> None:
    ctx_dir = staging_dir / "ctx"
    ctx_dir.mkdir(parents=True, exist_ok=True)
    path = ctx_dir / name
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row))
            fh.write("\n")


def _txn_row(
    *, event_id: str, account_id: str, amount: float, step: int, event_ts_us: int
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "account_id": account_id,
        "dest_id": "M1",
        "txn_type": "PAYMENT",
        "amount": amount,
        "oldbalance_org": 1000.0,
        "newbalance_orig": 1000.0 - amount,
        "oldbalance_dest": 0.0,
        "newbalance_dest": 0.0,
        "step": step,
        "event_ts_us": event_ts_us,
    }


def _wait_for_txn_count(
    redis_client: redis_lib.Redis, account_id: str, expected_count: int, *, timeout_s: float
) -> float:
    """Polls Redis until the account's txn_count matches, returns elapsed seconds."""
    key = redis_state_key(account_id)
    start = time.monotonic()
    deadline = start + timeout_s
    while time.monotonic() < deadline:
        raw = redis_client.get(key)
        if raw is not None:
            state = json.loads(raw)
            if state["txn_count"] == expected_count:
                return time.monotonic() - start
        time.sleep(0.02)
    raise AssertionError(
        f"{account_id} did not reach txn_count={expected_count} within {timeout_s}s "
        f"(last seen: {redis_client.get(key)!r})"
    )


def _get_state(redis_client: redis_lib.Redis, account_id: str) -> dict[str, object]:
    raw = redis_client.get(redis_state_key(account_id))
    assert raw is not None, f"no state for {account_id!r}"
    result: dict[str, object] = json.loads(raw)
    return result


def _launch_streaming(
    *, pg_env: dict[str, str], staging_dir: Path, persist_dir: Path
) -> subprocess.Popen[bytes]:
    env = {
        **os.environ,
        **pg_env,
        "C3_EVENT_STAGING_DIR": str(staging_dir),
        "C3_PATHWAY_PERSIST_DIR": str(persist_dir),
        "PATHWAY_LICENSE_KEY": "",
        "C3_PATHWAY_PG_SINK": "auto",
    }
    return subprocess.Popen(
        [sys.executable, "-m", "conquer3.cli", "pathway", "streaming"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def test_streaming_picks_up_a_new_file_quickly(
    warehouse: tuple[dict[str, str], redis_lib.Redis], tmp_path: Path
) -> None:
    pg_env, redis_client = warehouse
    staging_dir = tmp_path / "staging"
    persist_dir = tmp_path / "persist"
    (staging_dir / "ctx").mkdir(parents=True)

    proc = _launch_streaming(pg_env=pg_env, staging_dir=staging_dir, persist_dir=persist_dir)
    try:
        time.sleep(_STARTUP_GRACE_S)

        _write_jsonl(
            staging_dir,
            "part-00000.jsonl",
            [
                _txn_row(
                    event_id="e1", account_id="PICKUP1", amount=42.0, step=1, event_ts_us=1_000_000
                )
            ],
        )
        elapsed = _wait_for_txn_count(redis_client, "PICKUP1", 1, timeout_s=_PICKUP_TIMEOUT_S)
        assert elapsed < 2.0, f"pickup took {elapsed:.2f}s, expected under 2s"
    finally:
        proc.terminate()
        proc.wait(timeout=15)


def test_kill_and_restart_resumes_without_duplicating(
    warehouse: tuple[dict[str, str], redis_lib.Redis], tmp_path: Path
) -> None:
    pg_env, redis_client = warehouse
    staging_dir = tmp_path / "staging"
    persist_dir = tmp_path / "persist"
    (staging_dir / "ctx").mkdir(parents=True)

    proc = _launch_streaming(pg_env=pg_env, staging_dir=staging_dir, persist_dir=persist_dir)
    try:
        time.sleep(_STARTUP_GRACE_S)
        _write_jsonl(
            staging_dir,
            "part-00000.jsonl",
            [
                _txn_row(
                    event_id="e1",
                    account_id="KILLTEST1",
                    amount=10.0,
                    step=1,
                    event_ts_us=1_000_000,
                )
            ],
        )
        _wait_for_txn_count(redis_client, "KILLTEST1", 1, timeout_s=_PICKUP_TIMEOUT_S)
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=15)

    # Restart against the same persist dir + staging dir: must not double-count
    # the row the first process already processed and persisted.
    proc2 = _launch_streaming(pg_env=pg_env, staging_dir=staging_dir, persist_dir=persist_dir)
    try:
        time.sleep(_STARTUP_GRACE_S)
        state = _get_state(redis_client, "KILLTEST1")
        assert state["txn_count"] == 1, "restart alone must not reprocess the persisted row"

        _write_jsonl(
            staging_dir,
            "part-00001.jsonl",
            [
                _txn_row(
                    event_id="e2", account_id="KILLTEST1", amount=5.0, step=2, event_ts_us=2_000_000
                )
            ],
        )
        _wait_for_txn_count(redis_client, "KILLTEST1", 2, timeout_s=_PICKUP_TIMEOUT_S)
        state = _get_state(redis_client, "KILLTEST1")
        assert state["amount_sum"] == pytest.approx(15.0)
    finally:
        proc2.terminate()
        proc2.wait(timeout=15)
