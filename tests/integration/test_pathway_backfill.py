"""Layer 3b gate: static backfill against ephemeral Postgres + Redis containers.

Seeds a small synthetic dataset directly into ``silver.txn`` via SQL (bypassing
bronze/silver transforms entirely -- fast and independently runnable, matching
``scripts/smoke/layer3_feature_core.sh``'s philosophy rather than requiring the
full 6.3M-row Layer 2 dataset already loaded), exports it to JSONL staging, then
invokes ``conquer3 pathway backfill`` in a **subprocess** -- required because
Pathway's engine only supports calling ``pw.run()`` once per process.

Needs Docker (testcontainers spins up postgres:17-alpine and a redis container).
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("testcontainers")

from testcontainers.community.postgres import PostgresContainer
from testcontainers.community.redis import RedisContainer

from conquer3.config.settings import get_settings
from conquer3.db.bootstrap import apply_ddl
from conquer3.db.engine import pg_connection

pytestmark = [pytest.mark.integration, pytest.mark.pathway]

# account_id, dest_id, txn_type, amount, oldbalance_org, newbalance_orig,
# oldbalance_dest, newbalance_dest, step, event_ts_us
_SEED_ROWS: list[tuple[str, str, str, float, float, float, float, float, int, int]] = [
    ("C100", "M1", "PAYMENT", 100.0, 1000.0, 900.0, 0.0, 0.0, 1, 1_000_000),
    ("C100", "C500", "CASH_OUT", 300.0, 900.0, 600.0, 0.0, 300.0, 2, 2_000_000),
    ("C100", "M2", "PAYMENT", 50.0, 600.0, 550.0, 0.0, 0.0, 3, 3_000_000),
    ("C101", "M2", "PAYMENT", 50.0, 500.0, 450.0, 0.0, 0.0, 1, 1_500_000),
    ("C102", "C500", "CASH_OUT", 900.0, 900.0, 0.0, 0.0, 900.0, 1, 1_200_000),
]


@pytest.fixture
def warehouse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[tuple[dict[str, str], dict[str, str]]]:
    """Ephemeral Postgres + Redis, with the medallion DDL applied and silver.txn
    seeded. Yields (pg_env, redis_env) -- the env vars a `conquer3` subprocess
    needs to reach the same containers, since the containers' host-mapped ports
    are only known once they've started.
    """
    # Pinned to the same tags docker-compose.yaml uses, so the image is already
    # pulled on any machine that's run the local stack -- and so testcontainers
    # never falls back to `redis:latest`, which forces a registry pull.
    with (
        PostgresContainer("postgres:17-alpine") as pg,
        RedisContainer("redis:7-alpine") as redis_c,
    ):
        monkeypatch.setenv("POSTGRES_HOST", pg.get_container_host_ip())
        monkeypatch.setenv("POSTGRES_PORT", str(pg.get_exposed_port(5432)))
        monkeypatch.setenv("POSTGRES_DB", pg.dbname)
        monkeypatch.setenv("POSTGRES_USER", pg.username)
        monkeypatch.setenv("POSTGRES_PASSWORD", pg.password)
        # The testcontainers postgres:17-alpine image has no SSL configured.
        monkeypatch.setenv("POSTGRES_SSLMODE", "disable")
        monkeypatch.setenv("POSTGRES_CHANNEL_BINDING", "disable")
        monkeypatch.setenv("REDIS_HOST", redis_c.get_container_host_ip())
        monkeypatch.setenv("REDIS_PORT", str(redis_c.get_exposed_port(6379)))
        monkeypatch.setenv("C3_DUCKDB_PATH", str(tmp_path / "analytics.duckdb"))
        monkeypatch.setenv("C3_DUCKDB_TEMP_DIR", str(tmp_path / "duckdb_tmp"))
        get_settings.cache_clear()
        try:
            with pg_connection() as conn:
                apply_ddl(conn)
                _seed_silver(conn)

            pg_env = {
                "POSTGRES_HOST": pg.get_container_host_ip(),
                "POSTGRES_PORT": str(pg.get_exposed_port(5432)),
                "POSTGRES_DB": pg.dbname,
                "POSTGRES_USER": pg.username,
                "POSTGRES_PASSWORD": pg.password,
            }
            redis_env = {
                "REDIS_HOST": redis_c.get_container_host_ip(),
                "REDIS_PORT": str(redis_c.get_exposed_port(6379)),
            }
            yield pg_env, redis_env
        finally:
            get_settings.cache_clear()


def _seed_silver(conn: Any) -> None:
    with conn.cursor() as cur:
        for i, row in enumerate(_SEED_ROWS, start=1):
            (
                account_id,
                dest_id,
                txn_type,
                amount,
                oldbalance_org,
                newbalance_orig,
                oldbalance_dest,
                newbalance_dest,
                step,
                event_ts_us,
            ) = row
            cur.execute(
                """
                INSERT INTO silver.txn (
                    event_id, account_id, dest_id, txn_type, amount,
                    oldbalance_org, newbalance_orig, oldbalance_dest, newbalance_dest,
                    step, event_ts_us, is_fraud, is_flagged_fraud, bronze_row_num
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, false, false, %s)
                """,
                (
                    f"seed-{i:04d}",
                    account_id,
                    dest_id,
                    txn_type,
                    amount,
                    oldbalance_org,
                    newbalance_orig,
                    oldbalance_dest,
                    newbalance_dest,
                    step,
                    event_ts_us,
                    i,
                ),
            )


def _run_backfill(
    *,
    pg_env: dict[str, str],
    redis_env: dict[str, str],
    staging_dir: Path,
    extra_env: dict[str, str] | None = None,
) -> None:
    import os

    env = {**os.environ, **pg_env, **redis_env, "C3_EVENT_STAGING_DIR": str(staging_dir)}
    if extra_env:
        env.update(extra_env)
    subprocess.run(
        [sys.executable, "-m", "conquer3.cli", "pathway", "backfill"],
        env=env,
        check=True,
        timeout=60,
        capture_output=True,
        text=True,
    )


def _dump_account_state(pg_env: dict[str, str]) -> dict[str, dict[str, Any]]:
    import psycopg2

    dsn = (
        f"host={pg_env['POSTGRES_HOST']} port={pg_env['POSTGRES_PORT']} "
        f"dbname={pg_env['POSTGRES_DB']} user={pg_env['POSTGRES_USER']} "
        f"password={pg_env['POSTGRES_PASSWORD']}"
    )
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT account_id, state_version, state_json, updated_at_us "
                "FROM gold.account_state"
            )
            return {
                account_id: {
                    "state_version": state_version,
                    "state": json.loads(state_json),
                    "updated_at_us": updated_at_us,
                }
                for account_id, state_version, state_json, updated_at_us in cur.fetchall()
            }
    finally:
        conn.close()


def _truncate_account_state(pg_env: dict[str, str]) -> None:
    import psycopg2

    dsn = (
        f"host={pg_env['POSTGRES_HOST']} port={pg_env['POSTGRES_PORT']} "
        f"dbname={pg_env['POSTGRES_DB']} user={pg_env['POSTGRES_USER']} "
        f"password={pg_env['POSTGRES_PASSWORD']}"
    )
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE gold.account_state")
    finally:
        conn.close()


def test_static_backfill_row_count_parity(
    warehouse: tuple[dict[str, str], dict[str, str]], tmp_path: Path
) -> None:
    pg_env, redis_env = warehouse
    from conquer3.pipelines.transforms.export_staging import export_staging

    staging_dir = tmp_path / "staging"
    export_staging(staging_dir=str(staging_dir))

    _run_backfill(pg_env=pg_env, redis_env=redis_env, staging_dir=staging_dir)

    dump = _dump_account_state(pg_env)
    expected_accounts = {row[0] for row in _SEED_ROWS}
    assert set(dump) == expected_accounts

    # C100 has 3 transactions; C101 and C102 have 1 each.
    assert dump["C100"]["state"]["txn_count"] == 3
    assert dump["C101"]["state"]["txn_count"] == 1
    assert dump["C102"]["state"]["txn_count"] == 1


def test_licensed_and_unset_backfills_agree(
    warehouse: tuple[dict[str, str], dict[str, str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both sinks must produce byte-identical account state -- the fallback exists
    precisely because a Pathway license may not be configured, and it must never
    silently diverge from the licensed path (architecture plan, section 6).

    No real PATHWAY_LICENSE_KEY is needed to exercise the "licensed" code path:
    confirmed by reading the installed pathway package's source that
    pw.io.postgres.write's `_check_entitlements` call is gated behind a
    "postgres-wal-reader" check that lives only in `read()` (the CDC/replication
    path), never in `write()`. This test still routes through C3_PATHWAY_PG_SINK
    to force each path explicitly, so it's exercising real implementation parity
    between the Rust snapshot-writer and the psycopg fallback either way.
    """
    pg_env, redis_env = warehouse
    from conquer3.pipelines.transforms.export_staging import export_staging

    staging_dir = tmp_path / "staging"
    export_staging(staging_dir=str(staging_dir))

    _run_backfill(
        pg_env=pg_env,
        redis_env=redis_env,
        staging_dir=staging_dir,
        extra_env={"PATHWAY_LICENSE_KEY": "", "C3_PATHWAY_PG_SINK": "psycopg"},
    )
    unset_dump = _dump_account_state(pg_env)
    _truncate_account_state(pg_env)

    _run_backfill(
        pg_env=pg_env,
        redis_env=redis_env,
        staging_dir=staging_dir,
        extra_env={"PATHWAY_LICENSE_KEY": "", "C3_PATHWAY_PG_SINK": "licensed"},
    )
    licensed_dump = _dump_account_state(pg_env)

    assert set(unset_dump) == set(licensed_dump)
    for account_id in unset_dump:
        assert unset_dump[account_id] == licensed_dump[account_id]
