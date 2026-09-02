"""Helper for layer7_observability.sh: drives one real FraudScorerModel.predict()
call (fakes for Redis/mlflow, matching tests/unit/test_pyfunc_model.py's pattern --
no Docker dependency beyond the collector itself) so the redis_get/predict/
redis_set/file_append/score_batch spans and the scorer's custom metrics actually
get exported to whatever OTEL_EXPORTER_OTLP_ENDPOINT points at, plus one log record
so all three signals have something to check, then force-flushes before exiting so
the batch processors don't just buffer and drop on exit.

Used by layer7_observability.sh's live-remote-stack reachability check --
whatever .env's OTEL_EXPORTER_OTLP_ENDPOINT currently points at, no local
otel-collector container involved. Searches for the fixed service name
conquer3-scorer-layer7-gate.
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from conquer3.contracts.model_registry import ModelRef
from conquer3.serving.pyfunc_model import FraudScorerModel
from conquer3.telemetry.otel import init_telemetry


class _FakePipe:
    def predict_proba(self, rows: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({0: [0.1] * len(rows), 1: [0.9] * len(rows)}).to_numpy()


class _FakeStateStore:
    def __init__(self) -> None:
        self.data: dict[str, object] = {}

    def get(self, account_id: str) -> object | None:
        return self.data.get(account_id)

    def commit(self, state: object) -> bool:
        self.data[state.account_id] = state  # type: ignore[attr-defined]
        return True


class _FakeEventSink:
    def append(self, event: object) -> None:
        pass


def main() -> None:
    init_telemetry("conquer3-scorer-layer7-gate")

    model = FraudScorerModel()
    model._pipe = _FakePipe()  # type: ignore[attr-defined]
    model._ref = ModelRef(name="m", version="1", run_id="r", alias="champion", tags={})  # type: ignore[attr-defined]
    model._threshold = 0.5  # type: ignore[attr-defined]
    model._state = _FakeStateStore()  # type: ignore[attr-defined]
    model._sink = _FakeEventSink()  # type: ignore[attr-defined]
    model._init_instruments()  # type: ignore[attr-defined]

    row = {
        "event_id": "e1",
        "account_id": "C1",
        "dest_id": "C900",
        "txn_type": "TRANSFER",
        "amount": 999.0,
        "oldbalance_org": 1000.0,
        "newbalance_orig": 1.0,
        "oldbalance_dest": 0.0,
        "newbalance_dest": 999.0,
        "event_ts_us": 1_700_000_000_000_000,
        "step": 1,
    }
    model.predict(None, pd.DataFrame([row]))
    logging.getLogger("conquer3.layer7_gate").warning("LAYER7_GATE_LOG_PROBE score_batch done")

    from opentelemetry import _logs as logs_api
    from opentelemetry import metrics, trace

    trace.get_tracer_provider().force_flush()
    metrics.get_meter_provider().force_flush()
    logs_api.get_logger_provider().force_flush()
    time.sleep(2)  # PeriodicExportingMetricReader's own export cycle lags force_flush slightly


if __name__ == "__main__":
    main()
