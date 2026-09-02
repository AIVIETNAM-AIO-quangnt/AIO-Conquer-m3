"""ui.history: real ScoredEvent JSONL files in, ordered/filtered results out --
exercises the actual scan + parse + sort path the Inspection tab uses, not a
mock of it.
"""

from __future__ import annotations

from pathlib import Path

from conquer3.contracts.events import ScoredEvent, event_file_relpath
from conquer3.ui.history import events_to_frame, filter_events, load_recent_events


def _write_event(events_dir: Path, *, event_id: str, scored_at_us: int, fraud_score: float) -> None:
    event = ScoredEvent(
        event_id=event_id,
        account_id="C1",
        event_ts_us=scored_at_us,
        scored_at_us=scored_at_us,
        fraud_score=fraud_score,
        decision="FRAUD" if fraud_score >= 0.5 else "LEGIT",
        threshold=0.5,
        had_prev_state=False,
        model_name="paysim_fraud_clf",
        model_version="1",
        feature_schema_version=1,
        transaction={"amount": 100.0},
        features={"log1p_amount": 4.6},
    )
    relpath = event_file_relpath(scored_at_us, hostname="host", pid=1, worker_id=0)
    path = events_dir / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(event.to_json_line())


def test_load_recent_events_orders_newest_scored_first(tmp_path: Path) -> None:
    _write_event(tmp_path, event_id="e1", scored_at_us=1_700_000_000_000_000, fraud_score=0.1)
    _write_event(tmp_path, event_id="e2", scored_at_us=1_700_003_600_000_000, fraud_score=0.9)
    _write_event(tmp_path, event_id="e3", scored_at_us=1_700_001_800_000_000, fraud_score=0.5)

    events = load_recent_events(tmp_path)

    assert [e.event_id for e in events] == ["e2", "e3", "e1"]


def test_load_recent_events_respects_max_rows(tmp_path: Path) -> None:
    for i in range(5):
        _write_event(
            tmp_path, event_id=f"e{i}", scored_at_us=1_700_000_000_000_000 + i, fraud_score=0.5
        )

    events = load_recent_events(tmp_path, max_rows=2)

    assert len(events) == 2
    assert events[0].event_id == "e4"


def test_load_recent_events_on_empty_directory_returns_empty_list(tmp_path: Path) -> None:
    assert load_recent_events(tmp_path) == []


def test_filter_events_keeps_only_confidence_range(tmp_path: Path) -> None:
    _write_event(tmp_path, event_id="low", scored_at_us=1_700_000_000_000_000, fraud_score=0.1)
    _write_event(tmp_path, event_id="mid", scored_at_us=1_700_000_001_000_000, fraud_score=0.5)
    _write_event(tmp_path, event_id="high", scored_at_us=1_700_000_002_000_000, fraud_score=0.9)
    events = load_recent_events(tmp_path)

    filtered = filter_events(events, min_confidence=0.3, max_confidence=0.7)

    assert [e.event_id for e in filtered] == ["mid"]


def test_events_to_frame_preserves_row_count_and_feature_dict(tmp_path: Path) -> None:
    _write_event(tmp_path, event_id="e1", scored_at_us=1_700_000_000_000_000, fraud_score=0.8)
    events = load_recent_events(tmp_path)

    frame = events_to_frame(events)

    assert len(frame) == 1
    assert frame.iloc[0]["features"]["log1p_amount"] == 4.6
