"""Postgres sink for the account-state table: the licensed pw.io.postgres.write
connector when a Pathway license key is configured, else a hand-rolled CAS-guarded
upsert. Both paths must run in CI or the fallback rots silently the day the license
policy changes (architecture plan, section 6).
"""

from __future__ import annotations

import threading
from typing import Any

import pathway as pw
import psycopg

from conquer3.config.settings import PathwaySettings, PgSettings

__all__ = ["PsycopgUpsertObserver", "write_account_state"]

_UPSERT_SQL = """
    INSERT INTO gold.account_state (account_id, state_version, state_json, updated_at_us)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (account_id) DO UPDATE
    SET state_version = excluded.state_version,
        state_json = excluded.state_json,
        updated_at_us = excluded.updated_at_us
    WHERE excluded.updated_at_us > gold.account_state.updated_at_us
"""


class PsycopgUpsertObserver(pw.io.python.ConnectorObserver):
    """CAS-guarded upsert -- the fallback's equivalent of the Redis Lua script.

    Never deletes on retraction. Guarded by a lock: on_change's docstring only
    promises ordering *within* a processing-time batch is unspecified, not that
    calls are single-threaded, and a bare psycopg.Connection is not safe for
    concurrent use from multiple threads.
    """

    def __init__(self, *, pg_settings: PgSettings) -> None:
        self._conn = psycopg.connect(pg_settings.libpq_dsn, autocommit=True)
        self._lock = threading.Lock()

    def on_change(self, key: Any, row: dict[str, Any], time: int, is_addition: bool) -> None:
        if not is_addition:
            return
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                _UPSERT_SQL,
                (row["account_id"], row["state_version"], row["state_json"], row["updated_at_us"]),
            )

    def on_end(self) -> None:
        self._conn.close()


def write_account_state(
    result: pw.Table[Any], *, pathway_settings: PathwaySettings, pg_settings: PgSettings
) -> None:
    if _should_use_licensed(pathway_settings):
        pw.io.postgres.write(
            result,
            postgres_settings={
                "host": pg_settings.host,
                "port": str(pg_settings.port),
                "dbname": pg_settings.db,
                "user": pg_settings.user,
                "password": pg_settings.password,
            },
            table_name="account_state",
            schema_name="gold",
            init_mode="default",  # Airflow/db-migrate owns the DDL, never Pathway
            output_table_type="snapshot",
            primary_key=[result.account_id],
        )
    else:
        pw.io.python.write(result, PsycopgUpsertObserver(pg_settings=pg_settings))


def _should_use_licensed(pathway_settings: PathwaySettings) -> bool:
    if pathway_settings.pg_sink == "licensed":
        return True
    if pathway_settings.pg_sink == "psycopg":
        return False
    return bool(pathway_settings.license_key)  # "auto"
