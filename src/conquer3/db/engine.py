"""Warehouse connections.

Two connections, two jobs:

* :func:`pg_connection` -- a plain psycopg2 connection for control statements (DDL,
  TRUNCATE, ``ops.pipeline_runs`` bookkeeping).
* :func:`get_ibis_connection` -- an Ibis/DuckDB backend with Postgres ``ATTACH``ed as
  the ``pg`` catalog, for the bulk SQL that actually moves data between medallion
  layers. This is what makes "DuckDB+Ibis transforms" real rather than just psycopg2
  with extra steps -- see ``PgSettings.libpq_dsn``'s docstring.
"""

from __future__ import annotations

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import psycopg2

from conquer3.config.settings import DuckSettings, PgSettings, get_settings

if TYPE_CHECKING:
    import ibis

__all__ = ["get_ibis_connection", "pg_connection"]


@contextmanager
def pg_connection(pg: PgSettings | None = None) -> Iterator[psycopg2.extensions.connection]:
    """A short-lived, autocommit psycopg2 connection."""
    settings = pg if pg is not None else get_settings().pg
    conn = psycopg2.connect(settings.libpq_dsn)
    conn.autocommit = True
    try:
        yield conn
    finally:
        conn.close()


def get_ibis_connection(
    *, duck: DuckSettings | None = None, pg: PgSettings | None = None
) -> ibis.BaseBackend:
    """An Ibis DuckDB backend with Postgres attached as the ``pg`` catalog.

    Call ``.disconnect()`` (or use as a context manager) when done -- the DuckDB file
    is held open until then.

    ``ibis`` is imported here, not at module level, so that ``pg_connection`` above
    stays usable without dragging ibis/duckdb into a process that only needs plain
    Postgres (e.g. ``conquer3.ui``, which must not gain the pipeline extra's weight).

    DuckDB allows only one process to hold a given on-disk file open at a time and
    rejects a second opener immediately rather than waiting for the first to finish
    (see https://duckdb.org/docs/stable/connect/concurrency) -- this bit
    ``dag_bronze_ingest_paysim``'s ``ingest_bronze`` when it was manually triggered
    while ``dag_medallion_batch``'s hourly ``bronze_to_silver`` (a ~20-minute run
    over 6.3M rows) already held the file, two independent Airflow DAGs with no
    coordination between them. The blocking ``flock`` below, released only when the
    caller disconnects, makes every caller of this function -- any DAG, the CLI,
    tests -- queue for the DuckDB file instead of crashing when another caller
    already holds it.
    """
    import ibis

    duck_settings = duck if duck is not None else get_settings().duck
    pg_settings = pg if pg is not None else get_settings().pg

    db_path = Path(duck_settings.path)
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        Path(duck_settings.temp_dir).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print("Trying to make duckdb's dir at", duck_settings.path)
        print("Trying to locate duckdb's tempdir at", duck_settings.temp_dir)
        raise RuntimeError(
            f"cannot create DuckDB storage/temp directory ({exc}). Set "
            "C3_DUCKDB_PATH/C3_DUCKDB_TEMP_DIR to writable paths when running "
            "outside the container that mounts them."
        ) from exc

    lock_file = (db_path.parent / f"{db_path.name}.lock").open("w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"Waiting for DuckDB file lock on {db_path} (held by another process)...")
        fcntl.flock(lock_file, fcntl.LOCK_EX)

    try:
        con = ibis.duckdb.connect(
            str(db_path),
            extensions=["postgres"],
            memory_limit=duck_settings.memory_limit,
            threads=duck_settings.threads,
            temp_directory=duck_settings.temp_dir,
        )
        con.raw_sql(f"ATTACH IF NOT EXISTS '{pg_settings.libpq_dsn}' AS pg (TYPE postgres)")
    except BaseException:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()
        raise

    original_disconnect = con.disconnect

    def _disconnect_and_unlock() -> None:
        try:
            original_disconnect()
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()

    con.disconnect = _disconnect_and_unlock
    return con
