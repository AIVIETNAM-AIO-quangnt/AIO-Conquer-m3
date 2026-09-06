"""Regression test for the cross-DAG DuckDB race ``get_ibis_connection`` now guards
against: ``dag_bronze_ingest_paysim``'s ``ingest_bronze`` crashed with DuckDB's
"Conflicting lock is held" ``IOException`` when manually triggered while
``dag_medallion_batch``'s hourly ``bronze_to_silver`` (a ~20-minute run) already had
the file open -- two independent Airflow DAGs racing for the same on-disk file with
no coordination between them.

Exercises the exact synchronization primitives ``get_ibis_connection`` (see
``conquer3/db/engine.py``) wraps every connection in: an exclusive lock on a
``<db path>.lock`` sidecar file, held until the caller disconnects. Uses the same
``_try_lock``/``_unlock``/``_lock_blocking`` helpers engine.py uses itself (``fcntl.flock``
on POSIX, ``msvcrt.locking`` on Windows -- see engine.py's platform branch), so this
stays a from-scratch repro of the primitive without needing a real DuckDB file,
Postgres, or Docker, on either platform.
"""

from __future__ import annotations

import threading
from pathlib import Path

from conquer3.db.engine import _lock_blocking, _try_lock, _unlock


def test_second_acquirer_blocks_until_first_releases(tmp_path: Path) -> None:
    """Documents the fix: a second caller queues for the lock instead of racing
    DuckDB's own immediate-reject file lock."""
    db_path = tmp_path / "analytics.duckdb"
    lock_path = db_path.parent / f"{db_path.name}.lock"  # same convention as engine.py

    # "r+", not "w" -- "w" truncates on every open, which on Windows was observed to
    # drop another handle's still-held lock along with the truncated byte (see
    # engine.py's own comment on this). The byte must already exist before opening.
    lock_path.write_text("0")
    first = lock_path.open("r+")
    assert _try_lock(first)

    acquired = threading.Event()

    def _second_acquirer() -> None:
        second = lock_path.open("r+")
        try:
            _lock_blocking(second)  # blocks until `first` unlocks
            acquired.set()
        finally:
            _unlock(second)
            second.close()

    waiter = threading.Thread(target=_second_acquirer)
    waiter.start()

    # The second acquirer must still be blocked -- `first` hasn't released yet.
    assert not acquired.wait(timeout=0.5)

    _unlock(first)
    first.close()

    waiter.join(timeout=5)
    assert acquired.is_set()
    assert not waiter.is_alive()


def test_uncontended_lock_acquires_without_blocking(tmp_path: Path) -> None:
    """The common case: no other caller holds the file, so acquiring is immediate
    (mirrors get_ibis_connection's non-blocking-first-attempt fast path)."""
    db_path = tmp_path / "analytics.duckdb"
    lock_path = db_path.parent / f"{db_path.name}.lock"

    lock_path.write_text("0")
    lock_file = lock_path.open("r+")
    try:
        assert _try_lock(lock_file)
    finally:
        _unlock(lock_file)
        lock_file.close()
