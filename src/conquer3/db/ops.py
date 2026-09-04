"""Bookkeeping for pipeline runs, in ``ops.pipeline_runs``.

Used by ``pipelines/ingest`` and ``pipelines/transforms`` so every load/transform
leaves an audit trail Layer 6's DQ/skew-audit DAGs can read later.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import psycopg2

from conquer3.contracts.model_registry import ModelRef

__all__ = [
    "RunHandle",
    "delete_prediction_labels",
    "fetch_prediction_labels",
    "record_model_deployment",
    "track_run",
    "upsert_prediction_labels",
]


@dataclass
class RunHandle:
    run_id: int
    rows_in: int | None = None
    rows_out: int | None = None
    detail: str | None = None


@contextmanager
def track_run(conn: psycopg2.extensions.connection, layer: str) -> Iterator[RunHandle]:
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


@dataclass(frozen=True, slots=True)
class PredictionLabel:
    event_id: str
    is_fraud: bool
    source: str  # 'ui' | 'csv'


def upsert_prediction_labels(
    conn: psycopg2.extensions.connection, rows: Sequence[PredictionLabel]
) -> None:
    """Writes ground-truth labels for the UI's Inspection tab -- ``ops.prediction_labels``
    (``db/ddl/45_ops_labels.sql``). ``event_id`` is the primary key, so re-labeling an
    already-labeled prediction updates it in place rather than erroring.
    """
    if not rows:
        return
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO ops.prediction_labels (event_id, is_fraud, source) "
            "VALUES (%s, %s, %s) "
            "ON CONFLICT (event_id) DO UPDATE SET "
            "is_fraud = EXCLUDED.is_fraud, source = EXCLUDED.source, labeled_at = now()",
            [(r.event_id, r.is_fraud, r.source) for r in rows],
        )


def fetch_prediction_labels(
    conn: psycopg2.extensions.connection, event_ids: Sequence[str]
) -> dict[str, bool]:
    """Ground truth for the given ``event_id``s. An id absent from the returned
    mapping is ``unlabeled`` -- that third state is row absence, not a column value
    (``is_fraud`` is ``NOT NULL``), so callers must not default a missing key to
    ``False``.
    """
    if not event_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT event_id, is_fraud FROM ops.prediction_labels WHERE event_id = ANY(%s)",
            (list(event_ids),),
        )
        return dict(cur.fetchall())


def delete_prediction_labels(
    conn: psycopg2.extensions.connection, event_ids: Sequence[str]
) -> None:
    """Moves the given predictions back to ``unlabeled`` by removing their row --
    the Inspection tab's three-state label editor calls this when a row is set back
    to ``unlabeled`` rather than ``fraud``/``legit``."""
    if not event_ids:
        return
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM ops.prediction_labels WHERE event_id = ANY(%s)", (list(event_ids),)
        )


def record_model_deployment(conn: psycopg2.extensions.connection, ref: ModelRef) -> None:
    """Audit-trail row for a resolved model version. Not called by anything in
    Layer 4 -- Layer 5's serving boot sequence calls resolve_champion() then this,
    in that order, since contracts.model_registry can never import conquer3.db
    (layering forbids it)."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO ops.model_deployments (model_name, version, run_id, alias, degraded) "
            "VALUES (%s, %s, %s, %s, %s)",
            (ref.name, ref.version, ref.run_id, ref.alias, ref.degraded),
        )


def _finish(
    conn: psycopg2.extensions.connection,
    run_id: int,
    *,
    status: str,
    handle: RunHandle,
    detail: str | None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE ops.pipeline_runs "
            "SET finished_at = now(), status = %s, rows_in = %s, rows_out = %s, detail = %s "
            "WHERE run_id = %s",
            (status, handle.rows_in, handle.rows_out, detail, run_id),
        )
