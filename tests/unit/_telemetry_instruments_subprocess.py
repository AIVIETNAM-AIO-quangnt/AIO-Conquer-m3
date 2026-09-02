"""Standalone script, NOT a pytest test module (no `test_` prefix, so pytest never
collects it) -- run via subprocess.run() from test_telemetry_instruments.py so its
global MeterProvider/InMemoryMetricReader setup can't leak into the rest of the
unit test suite. Same concern as test_otel.py's module docstring.

Exercises FraudScorer.score() (same fakes-for-collaborators pattern as
tests/unit/test_scorer.py) against a real JsonlEventSink, then forces a real
append() I/O failure, and prints every recorded metric as JSON so the calling test
can assert on it.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from typing import Any

import pandas as pd
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

_reader = InMemoryMetricReader()
metrics.set_meter_provider(MeterProvider(metric_readers=[_reader]))

from conquer3.config.settings import EventSettings  # noqa: E402
from conquer3.contracts.events import ScoredEvent  # noqa: E402
from conquer3.contracts.model_registry import ModelRef  # noqa: E402
from conquer3.core.types import TransactionEvent  # noqa: E402
from conquer3.serving.event_sink import JsonlEventSink  # noqa: E402
from conquer3.serving.scorer import FraudScorer  # noqa: E402


class _FakePipe:
    def predict_proba(self, rows: pd.DataFrame) -> Any:
        return pd.DataFrame(
            {
                0: [0.1 for _ in rows["amount"]],
                1: [0.9 for _ in rows["amount"]],
            }
        ).to_numpy()


class _FakeStateStore:
    def __init__(self) -> None:
        self.data: dict[str, object] = {}

    def get(self, account_id: str) -> object | None:
        return self.data.get(account_id)

    def commit(self, state: object) -> bool:
        self.data[state.account_id] = state  # type: ignore[attr-defined]
        return True


def main() -> None:
    tmp_dir = sys.argv[1]

    sink = JsonlEventSink(event_settings=EventSettings(dir=tmp_dir))
    scorer = FraudScorer(
        pipe=_FakePipe(),
        ref=ModelRef(name="m", version="1", run_id="r", alias="champion", tags={}),
        threshold=0.5,
        state=_FakeStateStore(),  # type: ignore[arg-type]
        sink=sink,
    )

    scorer.score(
        [
            TransactionEvent(
                event_id="e1",
                account_id="C1",
                dest_id="C900",
                txn_type="TRANSFER",
                amount=999.0,
                oldbalance_org=1000.0,
                newbalance_orig=1.0,
                oldbalance_dest=0.0,
                newbalance_dest=999.0,
                event_ts_us=1_700_000_000_000_000,
                step=1,
            )
        ]
    )

    # Force a genuine append() I/O failure: close the fd out from under the sink
    # (same hour, so no rotation happens first), so its next os.write() raises
    # EBADF -- proves c3_event_append_failures_total fires on a real failure, not
    # a synthetic counter bump.
    os.close(sink._fd)  # type: ignore[arg-type]
    with contextlib.suppress(OSError):
        sink.append(
            ScoredEvent(
                event_id="e2",
                account_id="C1",
                event_ts_us=1_700_000_000_000_000,
                scored_at_us=1_700_000_000_000_000,
                fraud_score=0.1,
                decision="LEGIT",
                threshold=0.5,
                had_prev_state=True,
                model_name="m",
                model_version="1",
                feature_schema_version=1,
                transaction={},
                features={},
            )
        )

    metrics_by_name: dict[str, list[dict[str, object]]] = {}
    for rm in _reader.get_metrics_data().resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                points = []
                for dp in m.data.data_points:
                    value = getattr(dp, "value", None)
                    point = (
                        {"value": value}
                        if value is not None
                        else {"sum": dp.sum, "count": dp.count}
                    )
                    point["attributes"] = dict(dp.attributes)
                    points.append(point)
                metrics_by_name.setdefault(m.name, []).extend(points)

    print(json.dumps(metrics_by_name))


if __name__ == "__main__":
    main()
