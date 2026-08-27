"""Warehouse connections.

Two connections, two jobs:

* :func:`pg_connection` -- a plain psycopg connection for control statements (DDL,
  TRUNCATE, ``ops.pipeline_runs`` bookkeeping).
* :func:`get_ibis_connection` -- an Ibis/DuckDB backend with Postgres ``ATTACH``ed as
  the ``pg`` catalog, for the bulk SQL that actually moves data between medallion
  layers. This is what makes "DuckDB+Ibis transforms" real rather than just psycopg
  with extra steps -- see ``PgSettings.libpq_dsn``'s docstring.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import ibis
import psycopg

from conquer3.config.settings import DuckSettings, PgSettings, get_settings

__all__ = ["get_ibis_connection", "pg_connection"]


@contextmanager
def pg_connection(pg: PgSettings | None = None) -> Iterator[psycopg.Connection]:
    """A short-lived, autocommit psycopg connection."""
    settings = pg if pg is not None else get_settings().pg
    conn = psycopg.connect(settings.libpq_dsn, autocommit=True)
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
    """
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

    con = ibis.duckdb.connect(
        str(db_path),
        extensions=["postgres"],
        memory_limit=duck_settings.memory_limit,
        threads=duck_settings.threads,
        temp_directory=duck_settings.temp_dir,
    )
    con.raw_sql(f"ATTACH IF NOT EXISTS '{pg_settings.libpq_dsn}' AS pg (TYPE postgres)")
    return con
