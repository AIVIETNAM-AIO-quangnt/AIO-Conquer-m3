"""Reads the scorer's scored-event JSONL history for the Inspection and Benchmark tabs.

Reuses ``contracts.events`` wholesale -- the hour-partitioned path layout and the
``ScoredEvent`` record -- rather than re-deriving either; that module is
stdlib-only specifically so both the scorer and this reader can agree on the
layout without sharing a dependency. ``events_dir`` is mounted read-only into the
``ui`` container: this module only ever reads.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from conquer3.contracts.events import SCORED_SUBDIR, ScoredEvent

__all__ = ["events_to_frame", "filter_events", "load_events_frame", "load_recent_events"]


def _iter_jsonl_files(events_dir: Path, *, max_files: int) -> Iterator[Path]:
    """Hour-partition files, newest hour-directory first, capped at ``max_files``.

    ``dt=YYYY-MM-DD/hr=HH`` sorts lexicographically the same as chronologically,
    so a plain reverse sort on the directory name is newest-first with no
    date parsing.
    """
    scored_root = events_dir / SCORED_SUBDIR
    if not scored_root.is_dir():
        return
    hour_dirs = sorted(
        (
            hour_dir
            for day_dir in scored_root.iterdir()
            if day_dir.is_dir()
            for hour_dir in day_dir.iterdir()
            if hour_dir.is_dir()
        ),
        reverse=True,
    )
    count = 0
    for hour_dir in hour_dirs:
        for path in sorted(hour_dir.glob("part-*.jsonl")):
            yield path
            count += 1
            if count >= max_files:
                return


def load_recent_events(
    events_dir: str | Path, *, max_files: int = 200, max_rows: int = 50_000
) -> list[ScoredEvent]:
    """Every ``ScoredEvent`` under ``events_dir``, newest-scored-first.

    Bounded by ``max_files`` (hour-partition JSONL files scanned) and
    ``max_rows`` (records kept after sorting) -- a scorer running for days
    accumulates many hourly files, and without a bound the Inspection tab would
    grow unloadable rather than merely stale.
    """
    events: list[ScoredEvent] = []
    for path in _iter_jsonl_files(Path(events_dir), max_files=max_files):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(ScoredEvent.from_json_line(line))
    events.sort(key=lambda e: e.scored_at_us, reverse=True)
    return events[:max_rows]


def filter_events(
    events: Sequence[ScoredEvent],
    *,
    min_confidence: float = 0.0,
    max_confidence: float = 1.0,
) -> list[ScoredEvent]:
    """Keeps events whose ``fraud_score`` falls in ``[min_confidence, max_confidence]``."""
    return [e for e in events if min_confidence <= e.fraud_score <= max_confidence]


def events_to_frame(events: Sequence[ScoredEvent]) -> pd.DataFrame:
    """Flattens ``ScoredEvent`` rows into a DataFrame for the Inspection tab.

    ``transaction``/``features`` stay as nested dict columns -- the 2D plot's
    axis picker reads out of them directly rather than exploding every possible
    column up front.
    """
    columns = (
        "event_id",
        "account_id",
        "scored_at_us",
        "event_ts_us",
        "fraud_score",
        "decision",
        "threshold",
        "had_prev_state",
        "model_name",
        "model_version",
        "feature_schema_version",
        "transaction",
        "features",
    )
    if not events:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = [
        {
            "event_id": e.event_id,
            "account_id": e.account_id,
            "scored_at_us": e.scored_at_us,
            "event_ts_us": e.event_ts_us,
            "fraud_score": e.fraud_score,
            "decision": e.decision,
            "threshold": e.threshold,
            "had_prev_state": e.had_prev_state,
            "model_name": e.model_name,
            "model_version": e.model_version,
            "feature_schema_version": e.feature_schema_version,
            "transaction": e.transaction,
            "features": e.features,
        }
        for e in events
    ]
    return pd.DataFrame(rows, columns=list(columns))


@st.cache_data(ttl=15, show_spinner=False)
def load_events_frame(events_dir: str, max_files: int, max_rows: int) -> pd.DataFrame:
    """Cached ``events_to_frame(load_recent_events(...))`` shared by every tab that
    browses scored-event history, so two tabs open in the same session scan the
    JSONL files once, not once per tab."""
    events = load_recent_events(events_dir, max_files=max_files, max_rows=max_rows)
    return events_to_frame(events)
