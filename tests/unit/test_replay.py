"""producer.replay: raw-CSV-row -> TransactionEvent-field mapping, and the
replay() HTTP loop, against a fake /predict (httpx.MockTransport -- no real
server, no extra test dependency).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from conquer3.core.timeref import derive_event_ts_us
from conquer3.producer.replay import load_raw_paysim, replay, run_replay, to_transactions_frame

_RAW_ROW = {
    "step": 1,
    "type": "TRANSFER",
    "amount": 181.0,
    "nameOrig": "C1",
    "oldbalanceOrg": 181.0,
    "newbalanceOrig": 0.0,
    "nameDest": "M900",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 181.0,
    "isFraud": 1,
    "isFlaggedFraud": 0,
}


def _raw_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_load_raw_paysim_rejects_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("step,type,amount\n1,TRANSFER,10.0\n")
    with pytest.raises(ValueError, match="missing raw PaySim1 columns"):
        load_raw_paysim(csv_path)


def test_load_raw_paysim_respects_limit(tmp_path: Path) -> None:
    csv_path = tmp_path / "ok.csv"
    frame = _raw_frame([_RAW_ROW, {**_RAW_ROW, "step": 2}])
    frame.to_csv(csv_path, index=False)
    assert len(load_raw_paysim(csv_path)) == 2
    assert len(load_raw_paysim(csv_path, limit=1)) == 1


def test_to_transactions_frame_maps_fields_and_derives_event_id() -> None:
    out = to_transactions_frame(_raw_frame([_RAW_ROW]))
    row = out.iloc[0]
    assert row["event_id"] == "ps-0000000001"
    assert row["account_id"] == "C1"
    assert row["dest_id"] == "M900"
    assert row["txn_type"] == "TRANSFER"
    assert row["amount"] == 181.0
    assert row["oldbalance_org"] == 181.0
    assert row["newbalance_orig"] == 0.0
    assert row["oldbalance_dest"] == 0.0
    assert row["newbalance_dest"] == 181.0
    assert bool(row["is_fraud"]) is True
    assert bool(row["is_flagged_fraud"]) is False


def test_to_transactions_frame_event_ts_us_matches_timeref_for_a_single_row_step() -> None:
    # Alone in its step, this row is intra_step_seq=1 of step_cardinality=1.
    out = to_transactions_frame(_raw_frame([_RAW_ROW]))
    assert int(out.iloc[0]["event_ts_us"]) == derive_event_ts_us(1, 1, 1)


def test_to_transactions_frame_orders_same_step_rows_by_file_position() -> None:
    rows = [{**_RAW_ROW, "nameOrig": f"C{i}"} for i in range(3)]  # all step=1
    out = to_transactions_frame(_raw_frame(rows))
    expected = [derive_event_ts_us(1, i + 1, 3) for i in range(3)]
    assert list(out["event_ts_us"].astype(int)) == expected
    # Strictly increasing -- file order within a step is a real time ordering.
    assert expected == sorted(expected)


def test_to_transactions_frame_event_ids_are_1_based_and_zero_padded() -> None:
    rows = [_RAW_ROW, {**_RAW_ROW, "nameOrig": "C2"}]
    out = to_transactions_frame(_raw_frame(rows))
    assert list(out["event_id"]) == ["ps-0000000001", "ps-0000000002"]


def _mock_predict_transport(
    *, captured: list[dict[str, Any]], score: float = 0.5
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/predict"
        body = json.loads(request.content)
        captured.append(body)
        results = [
            {
                "event_id": txn["event_id"],
                "fraud_score": score,
                "decision": "FRAUD" if score >= 0.5 else "LEGIT",
                "had_prev_state": False,
                "seconds_since_last_txn": None,
                "model_version": "1",
                "feature_schema_version": 1,
                "degraded": False,
            }
            for txn in body["transactions"]
        ]
        return httpx.Response(200, json=results)

    return httpx.MockTransport(handler)


def test_replay_sends_json_safe_native_types_not_numpy_scalars(tmp_path: Path) -> None:
    """The regression this guards: pandas int64/float64/bool_ scalars aren't
    JSON-serializable by the stdlib json module httpx uses -- to_json's
    round trip must strip them before the request body is built."""
    csv_path = tmp_path / "in.csv"
    _raw_frame([_RAW_ROW]).to_csv(csv_path, index=False)

    captured: list[dict[str, Any]] = []
    client = httpx.Client(transport=_mock_predict_transport(captured=captured))

    list(replay(csv_path, endpoint="http://scorer.test", client=client))

    assert len(captured) == 1
    (txn,) = captured[0]["transactions"]
    assert isinstance(txn["amount"], float)
    assert isinstance(txn["step"], int)
    assert isinstance(txn["event_ts_us"], int)
    assert isinstance(txn["account_id"], str)


def test_replay_propagates_dry_run(tmp_path: Path) -> None:
    csv_path = tmp_path / "in.csv"
    _raw_frame([_RAW_ROW]).to_csv(csv_path, index=False)
    captured: list[dict[str, Any]] = []
    client = httpx.Client(transport=_mock_predict_transport(captured=captured))

    list(replay(csv_path, endpoint="http://scorer.test", dry_run=True, client=client))

    assert captured[0]["dry_run"] is True


def test_replay_merges_labels_with_predictions(tmp_path: Path) -> None:
    csv_path = tmp_path / "in.csv"
    _raw_frame([_RAW_ROW]).to_csv(csv_path, index=False)
    client = httpx.Client(transport=_mock_predict_transport(captured=[], score=0.9))

    (batch,) = list(replay(csv_path, endpoint="http://scorer.test", client=client))

    row = batch.iloc[0]
    assert row["event_id"] == "ps-0000000001"
    assert bool(row["is_fraud"]) is True
    assert row["fraud_score"] == 0.9
    assert row["decision"] == "FRAUD"


def test_replay_batches_respect_batch_size(tmp_path: Path) -> None:
    csv_path = tmp_path / "in.csv"
    rows = [{**_RAW_ROW, "nameOrig": f"C{i}"} for i in range(5)]
    _raw_frame(rows).to_csv(csv_path, index=False)
    captured: list[dict[str, Any]] = []
    client = httpx.Client(transport=_mock_predict_transport(captured=captured))

    batches = list(replay(csv_path, endpoint="http://scorer.test", batch_size=2, client=client))

    assert [len(b) for b in batches] == [2, 2, 1]
    assert [len(c["transactions"]) for c in captured] == [2, 2, 1]


def test_replay_raises_with_context_on_a_non_200(tmp_path: Path) -> None:
    csv_path = tmp_path / "in.csv"
    _raw_frame([_RAW_ROW]).to_csv(csv_path, index=False)

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(422, text="bad request")

    client = httpx.Client(transport=httpx.MockTransport(handler))

    with pytest.raises(RuntimeError, match="ps-0000000001"):
        list(replay(csv_path, endpoint="http://scorer.test", client=client))


def test_run_replay_writes_expected_csv_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_replay() builds its own httpx.Client internally, so this proves the
    CSV-writing contract (header, row count, column values) by faking the
    replay() generator it drives rather than a real HTTP round trip -- the
    request/response shape itself is already covered by the replay() tests
    above."""
    import conquer3.producer.replay as replay_mod

    csv_path = tmp_path / "in.csv"
    rows = [{**_RAW_ROW, "nameOrig": f"C{i}"} for i in range(3)]
    _raw_frame(rows).to_csv(csv_path, index=False)
    out_path = tmp_path / "out" / "results.csv"

    def fake_replay(*args: Any, **kwargs: Any) -> Any:
        frame = to_transactions_frame(load_raw_paysim(csv_path))
        yield pd.DataFrame(
            {
                "event_id": frame["event_id"],
                "is_fraud": frame["is_fraud"],
                "is_flagged_fraud": frame["is_flagged_fraud"],
                "fraud_score": 0.1,
                "decision": "LEGIT",
                "had_prev_state": False,
                "seconds_since_last_txn": None,
                "model_version": "1",
                "feature_schema_version": 1,
                "degraded": False,
            }
        )

    monkeypatch.setattr(replay_mod, "replay", fake_replay)

    total = run_replay(csv_path, out_path, endpoint="http://scorer.test", progress_every=0)

    assert total == 3
    assert out_path.is_file()
    with out_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        written = list(reader)
    assert len(written) == 3
    assert written[0]["event_id"] == "ps-0000000001"
    assert written[0]["decision"] == "LEGIT"
