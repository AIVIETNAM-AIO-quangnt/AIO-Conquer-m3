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

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, TYPE_CHECKING

import psycopg2

from conquer3.config.settings import DuckSettings, PgSettings, get_settings

if TYPE_CHECKING:
    import ibis

__all__ = ["get_ibis_connection", "pg_connection"]

# `fcntl.flock` doesn't exist on Windows -- the DuckDB file lock below (see
# ``get_ibis_connection``'s docstring) needs a cross-platform exclusive lock on a
# sidecar ``.lock`` file, so branch on the one primitive each platform actually has:
# ``fcntl.flock`` on POSIX, ``msvcrt.locking`` on Windows. ``msvcrt.locking`` has no
# blocking mode of its own (``LK_LOCK`` internally retries for ~10s then raises), so
# both branches are built as a non-blocking probe plus our own retry loop, giving the
# same indefinite-blocking semantics on both platforms.
if sys.platform == "win32":
    import msvcrt

    def _try_lock(f: IO[str]) -> bool:
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _unlock(f: IO[str]) -> None:
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
else:
    import fcntl

    def _try_lock(f: IO[str]) -> bool:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _unlock(f: IO[str]) -> None:
        fcntl.flock(f, fcntl.LOCK_UN)


def _lock_blocking(f: IO[str]) -> None:
    """Block until ``f`` is exclusively locked (see the platform ``_try_lock``s)."""
    while not _try_lock(f):
        time.sleep(0.2)


@contextmanager
def pg_connection(pg: PgSettings | None = None) -> Iterator[psycopg2.extensions.connection]:
    """A short-lived, autocommit psycopg2 connection.

    "Short-lived" is the intent, not a guarantee every caller honors --
    ``bronze_to_silver``/``silver_to_gold`` keep one of these open (idle) across
    ``ops.track_run`` while a *separate* Ibis/DuckDB connection does the actual
    bulk transform, which can run for tens of minutes over the full table. Without
    TCP keepalives, a connection idle that long gets silently dropped by whatever
    sits between here and Postgres (cloud LB, NAT, pooler) with no FIN/RST -- the
    next statement on it (``track_run``'s finishing ``UPDATE``) then fails as
    ``OperationalError: SSL SYSCALL error: EOF detected`` instead of a clean,
    retryable error. Keepalive probes are real packets, so they reset that
    idle-timer even though no query is sent.
    """
    settings = pg if pg is not None else get_settings().pg
    conn = psycopg2.connect(
        settings.libpq_dsn,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
    )
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

    lock_path = db_path.parent / f"{db_path.name}.lock"
    # Ensure the byte msvcrt.locking needs to lock exists *before* opening for the
    # lock itself, and open with "r+" (not "w") from then on -- "w" truncates on
    # every open, including a second caller's, which on Windows was observed to drop
    # the first caller's still-held lock along with the truncated byte. fcntl.flock
    # doesn't care about file content either way, so this is safe on POSIX too.
    if not lock_path.exists():
        lock_path.write_text("0")
    lock_file = lock_path.open("r+")
    if not _try_lock(lock_file):
        print(f"Waiting for DuckDB file lock on {db_path} (held by another process)...")
        _lock_blocking(lock_file)

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
        _unlock(lock_file)
        lock_file.close()
        raise

    original_disconnect = con.disconnect

    def _disconnect_and_unlock() -> None:
        try:
            original_disconnect()
        finally:
            _unlock(lock_file)
            lock_file.close()

    con.disconnect = _disconnect_and_unlock
    return con
