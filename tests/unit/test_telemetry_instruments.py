"""Plan §10's "Custom instruments" list actually gets recorded, for the
instruments the scorer owns: c3_decision_total, c3_fraud_score,
c3_score_latency_ms, c3_feature_null_total (FraudScorerModel), and
c3_event_append_failures_total (JsonlEventSink). c3_state_{hit,miss,cas_rejected}
and c3_model_resolution_degraded already have their own coverage (state_store.py's
docstring, tests/integration/test_model_registry_e2e.py).

Runs the exercise in a subprocess (_telemetry_instruments_subprocess.py) because it
needs to install its own MeterProvider -- see that script's docstring, and
test_otel.py's, for why that can't happen in-process without wedging every other
unit test that touches metrics in the same pytest session.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).parent / "_telemetry_instruments_subprocess.py"


def test_scorer_and_event_sink_instruments_are_recorded(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    metrics_by_name = json.loads(result.stdout.strip().splitlines()[-1])

    decision_points = metrics_by_name["c3_decision_total"]
    assert any(
        p["attributes"].get("decision") == "FRAUD" and p["value"] >= 1 for p in decision_points
    )

    fraud_score_points = metrics_by_name["c3_fraud_score"]
    assert fraud_score_points[0]["count"] == 1
    assert fraud_score_points[0]["sum"] == 0.9

    latency_points = metrics_by_name["c3_score_latency_ms"]
    assert latency_points[0]["count"] == 1
    assert latency_points[0]["sum"] >= 0

    # First transaction for this account -- cold start, so several window
    # features are null (core.schema.COLD_START_NULL_FEATURES).
    null_points = metrics_by_name["c3_feature_null_total"]
    assert any(p["attributes"].get("feature") == "seconds_since_last_txn" for p in null_points)

    failure_points = metrics_by_name["c3_event_append_failures_total"]
    assert sum(p["value"] for p in failure_points) >= 1
