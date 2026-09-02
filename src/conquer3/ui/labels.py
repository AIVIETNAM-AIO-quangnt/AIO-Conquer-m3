"""``ops.prediction_labels`` read/write for the Inspection tab's label editor.

Goes through ``db.engine.pg_connection`` -- the single Postgres entry point --
never a second ``psycopg.connect`` call.
"""

from __future__ import annotations

from collections.abc import Mapping

from conquer3.db.engine import pg_connection
from conquer3.db.ops import (
    PredictionLabel,
    delete_prediction_labels,
    fetch_prediction_labels,
    upsert_prediction_labels,
)

__all__ = ["apply_label_edits", "get_labels"]


def get_labels(event_ids: list[str]) -> dict[str, bool]:
    """Ground truth for ``event_ids``. A missing key means ``unlabeled`` -- that
    third state is row absence, never a boolean default."""
    with pg_connection() as conn:
        return fetch_prediction_labels(conn, event_ids)


def apply_label_edits(edits: Mapping[str, str | None], *, source: str) -> None:
    """Applies a batch of label-editor edits in one connection.

    ``edits`` maps ``event_id`` to ``"fraud"`` / ``"legit"`` / ``None`` (moving
    the row back to ``unlabeled``). Splitting into an upsert batch and a delete
    batch, but running both against the same connection, keeps a mixed edit
    (some rows labeled, others cleared) from landing half-applied.
    """
    to_upsert = [
        PredictionLabel(event_id=event_id, is_fraud=(value == "fraud"), source=source)
        for event_id, value in edits.items()
        if value is not None
    ]
    to_delete = [event_id for event_id, value in edits.items() if value is None]
    with pg_connection() as conn:
        upsert_prediction_labels(conn, to_upsert)
        delete_prediction_labels(conn, to_delete)
