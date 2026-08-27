"""Bookkeeping for pipeline runs, in ``ops.pipeline_runs``.

Used by ``pipelines/ingest`` and ``pipelines/transforms`` so every load/transform
leaves an audit trail Layer 6's DQ/skew-audit DAGs can read later.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import psycopg

__all__ = ["RunHandle", "track_run"]


@dataclass
class RunHandle:
    run_id: int
    rows_in: int | None = None
    rows_out: int | None = None
    detail: str | None = None


@contextmanager
def track_run(conn: psycopg.Connection, layer: str) -> Iterator[RunHandle]:
    """Records one ``ops.pipeline_runs`` row for the lifetime of the ``with`` block.

    Set ``rows_in``/``rows_out``/``detail`` on the yielded handle before the block
    exits. Marked ``success`` on a clean exit, ``failed`` (with the exception message
    in ``detail``) if the block raises -- the exception still propagates.
    """
    with conn.cursor() as cur:
        cur.execute("INSERT INTO ops.pipeline_runs (layer) VALUES (%s) RETURNING run_id", (layer,))
        row = cur.fetchone()
        assert row is not None
        run_id = row[0]

    handle = RunHandle(run_id=run_id)
    try:
        yield handle
    except Exception as exc:
        _finish(conn, run_id, status="failed", handle=handle, detail=str(exc))
        raise
    else:
        _finish(conn, run_id, status="success", handle=handle, detail=handle.detail)


def _finish(
    conn: psycopg.Connection, run_id: int, *, status: str, handle: RunHandle, detail: str | None
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE ops.pipeline_runs "
            "SET finished_at = now(), status = %s, rows_in = %s, rows_out = %s, detail = %s "
            "WHERE run_id = %s",
            (status, handle.rows_in, handle.rows_out, detail, run_id),
        )
