"""The scored-event record and the JSONL path layout.

stdlib only: BentoML writes these files and the pipeline reads them, and the two
sides must agree without sharing a dependency.

Path layout::

    ${C3_EVENT_DIR}/scored/dt=YYYY-MM-DD/hr=HH/part-{host}-{pid}-{worker}.jsonl

**One file per worker process.** Multiple BentoML workers appending to a single file
interleave partial writes once a line exceeds PIPE_BUF, producing truncated JSON that
only shows up under load. Per-worker files remove the failure mode entirely.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

SCORED_SUBDIR = "scored"
SUCCESS_MARKER = "_SUCCESS"
EVENT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ScoredEvent:
    """One transaction, its features, and the score assigned to it.

    Carries the **full feature vector**, not just the score. That is what makes the
    daily skew audit possible: the pipeline recomputes features from the raw fields
    and diffs them against what serving actually used.
    """

    event_id: str
    account_id: str
    event_ts_us: int
    scored_at_us: int
    fraud_score: float
    decision: str
    threshold: float
    had_prev_state: bool
    model_name: str
    model_version: str
    feature_schema_version: int
    transaction: dict[str, Any]
    features: dict[str, Any]
    event_schema_version: int = EVENT_SCHEMA_VERSION
    trace_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json_line(self) -> str:
        """A single line, newline-terminated, written with one atomic os.write()."""
        from dataclasses import asdict

        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True) + "\n"

    @classmethod
    def from_json_line(cls, line: str | bytes) -> ScoredEvent:
        payload = json.loads(line)
        known = {f: payload[f] for f in cls.__slots__ if f in payload}
        return cls(**known)


def hour_partition(ts_us: int) -> tuple[str, str]:
    """``(dt=YYYY-MM-DD, hr=HH)`` for an event timestamp."""
    moment = datetime.fromtimestamp(ts_us / 1_000_000, tz=UTC)
    return f"dt={moment:%Y-%m-%d}", f"hr={moment:%H}"


def event_file_relpath(ts_us: int, *, hostname: str, pid: int, worker_id: int) -> str:
    """Relative path of the JSONL file a given worker appends to for this hour."""
    day, hour = hour_partition(ts_us)
    return f"{SCORED_SUBDIR}/{day}/{hour}/part-{hostname}-{pid}-{worker_id}.jsonl"


def success_marker_relpath(ts_us: int) -> str:
    """Marker written when an hour's directory is closed.

    Lets the ingest DAG distinguish "this hour is complete" from "still being
    written", which is the difference between a clean load and a truncated one.
    """
    day, hour = hour_partition(ts_us)
    return f"{SCORED_SUBDIR}/{day}/{hour}/{SUCCESS_MARKER}"
