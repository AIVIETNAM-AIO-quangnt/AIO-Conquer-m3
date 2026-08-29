"""JsonlEventSink: correct path layout, hour rotation + _SUCCESS marker, and
survives concurrent-thread appends without a corrupted line -- the property that
lets multiple request threads inside one uvicorn worker safely share one sink
(plan §8.4/§8.6)."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from conquer3.config.settings import EventSettings
from conquer3.contracts.events import SUCCESS_MARKER, ScoredEvent
from conquer3.serving.event_sink import JsonlEventSink

_PID_TOKEN = f"-{os.getpid()}-"


def _event(*, event_id: str, scored_at_us: int, account_id: str = "C1") -> ScoredEvent:
    return ScoredEvent(
        event_id=event_id,
        account_id=account_id,
        event_ts_us=scored_at_us,
        scored_at_us=scored_at_us,
        fraud_score=0.1,
        decision="LEGIT",
        threshold=0.5,
        had_prev_state=False,
        model_name="m",
        model_version="1",
        feature_schema_version=1,
        transaction={},
        features={},
    )


def _read_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_append_writes_one_line_to_the_hour_partitioned_path(tmp_path: Path) -> None:
    sink = JsonlEventSink(event_settings=EventSettings(dir=str(tmp_path)))
    # 2024-01-01T00:00:00Z in microseconds.
    sink.append(_event(event_id="e1", scored_at_us=1_704_067_200_000_000))
    sink.close()

    files = list((tmp_path / "scored" / "dt=2024-01-01" / "hr=00").glob("*.jsonl"))
    assert len(files) == 1
    assert _PID_TOKEN in files[0].name
    lines = _read_lines(files[0])
    assert len(lines) == 1
    assert lines[0]["event_id"] == "e1"


def test_multiple_appends_in_the_same_hour_share_one_file(tmp_path: Path) -> None:
    sink = JsonlEventSink(event_settings=EventSettings(dir=str(tmp_path)))
    base = 1_704_067_200_000_000
    for i in range(5):
        sink.append(_event(event_id=f"e{i}", scored_at_us=base + i * 1_000_000))
    sink.close()

    files = list((tmp_path / "scored" / "dt=2024-01-01" / "hr=00").glob("*.jsonl"))
    assert len(files) == 1
    assert len(_read_lines(files[0])) == 5


def test_crossing_an_hour_boundary_rotates_and_drops_success_marker(tmp_path: Path) -> None:
    sink = JsonlEventSink(event_settings=EventSettings(dir=str(tmp_path)))
    hour0 = 1_704_067_200_000_000  # hr=00
    hour1 = hour0 + 3_600_000_000  # hr=01
    sink.append(_event(event_id="e0", scored_at_us=hour0))
    sink.append(_event(event_id="e1", scored_at_us=hour1))
    sink.close()

    dir0 = tmp_path / "scored" / "dt=2024-01-01" / "hr=00"
    dir1 = tmp_path / "scored" / "dt=2024-01-01" / "hr=01"
    assert (dir0 / SUCCESS_MARKER).is_file()
    assert not (dir1 / SUCCESS_MARKER).is_file()  # still open -- close() never marks it
    assert len(_read_lines(next(dir0.glob("*.jsonl")))) == 1
    assert len(_read_lines(next(dir1.glob("*.jsonl")))) == 1


def test_concurrent_thread_appends_never_interleave_or_corrupt_a_line(tmp_path: Path) -> None:
    """Multiple threads sharing one sink instance -- the exact scenario
    /invocations creates once concurrent requests run predict() on parallel
    threads inside a single worker process (plan §8.4)."""
    sink = JsonlEventSink(event_settings=EventSettings(dir=str(tmp_path)))
    base = 1_704_067_200_000_000
    n_threads = 8
    n_per_thread = 50

    def _write(thread_idx: int) -> None:
        for i in range(n_per_thread):
            sink.append(_event(event_id=f"t{thread_idx}-{i}", scored_at_us=base))

    threads = [threading.Thread(target=_write, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    sink.close()

    files = list((tmp_path / "scored" / "dt=2024-01-01" / "hr=00").glob("*.jsonl"))
    assert len(files) == 1
    lines = _read_lines(files[0])  # raises on any truncated/malformed JSON line
    assert len(lines) == n_threads * n_per_thread
    assert {line["event_id"] for line in lines} == {
        f"t{t}-{i}" for t in range(n_threads) for i in range(n_per_thread)
    }


def test_close_before_any_append_is_a_no_op(tmp_path: Path) -> None:
    sink = JsonlEventSink(event_settings=EventSettings(dir=str(tmp_path)))
    sink.close()  # must not raise
    sink.close()  # idempotent
