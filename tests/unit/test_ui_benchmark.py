"""ui.benchmark: real ScoredEvent JSONL fixtures in, natural-key grouping and
cross-model comparison out -- no mocking, no Streamlit runtime, same style as
test_ui_history.py.
"""

from __future__ import annotations

from pathlib import Path

from conquer3.contracts.events import ScoredEvent, event_file_relpath
from conquer3.ui.benchmark import (
    _aggregate_by_model,
    _apply_search_filters,
    _compare_rows,
    _distinct_transactions,
    _has_any_filter,
    _with_natural_key_fields,
)
from conquer3.ui.history import load_events_frame


def _write_event(
    events_dir: Path,
    *,
    event_id: str,
    scored_at_us: int,
    account_id: str = "C1",
    dest_id: str = "M1",
    txn_type: str = "PAYMENT",
    amount: float = 100.0,
    model_name: str = "paysim_fraud_clf",
    model_version: str = "1",
    fraud_score: float = 0.5,
    threshold: float = 0.5,
) -> None:
    event = ScoredEvent(
        event_id=event_id,
        account_id=account_id,
        event_ts_us=scored_at_us,
        scored_at_us=scored_at_us,
        fraud_score=fraud_score,
        decision="FRAUD" if fraud_score >= threshold else "LEGIT",
        threshold=threshold,
        had_prev_state=False,
        model_name=model_name,
        model_version=model_version,
        feature_schema_version=1,
        transaction={
            "account_id": account_id,
            "dest_id": dest_id,
            "txn_type": txn_type,
            "amount": amount,
        },
        features={},
    )
    relpath = event_file_relpath(scored_at_us, hostname="host", pid=1, worker_id=0)
    path = events_dir / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(event.to_json_line())


def _load(events_dir: Path):
    return _with_natural_key_fields(load_events_frame(str(events_dir), 200, 50_000))


def test_with_natural_key_fields_derives_dest_txn_amount(tmp_path: Path) -> None:
    _write_event(
        tmp_path,
        event_id="e1",
        scored_at_us=1_700_000_000_000_000,
        account_id="C1",
        dest_id="M9",
        txn_type="CASH_OUT",
        amount=42.5,
    )

    df = _load(tmp_path)

    row = df.iloc[0]
    assert row["account_id"] == "C1"
    assert row["dest_id"] == "M9"
    assert row["txn_type"] == "CASH_OUT"
    assert row["amount"] == 42.5


def test_aggregate_by_model_counts_and_sorts_desc(tmp_path: Path) -> None:
    for i in range(3):
        _write_event(
            tmp_path, event_id=f"a{i}", scored_at_us=1_700_000_000_000_000 + i, model_version="2"
        )
    _write_event(tmp_path, event_id="b0", scored_at_us=1_700_000_000_000_100, model_version="1")

    counts = _aggregate_by_model(_load(tmp_path))

    assert counts.to_dict("records") == [
        {"model": "paysim_fraud_clf@2", "predictions": 3},
        {"model": "paysim_fraud_clf@1", "predictions": 1},
    ]


def test_distinct_transactions_groups_by_natural_key(tmp_path: Path) -> None:
    _write_event(
        tmp_path,
        event_id="e1",
        scored_at_us=1_700_000_000_000_000,
        account_id="C1",
        dest_id="M1",
        txn_type="PAYMENT",
        amount=100.0,
        model_version="1",
    )
    _write_event(
        tmp_path,
        event_id="e2",
        scored_at_us=1_700_000_000_000_100,
        account_id="C1",
        dest_id="M1",
        txn_type="PAYMENT",
        amount=100.0,
        model_version="2",
    )
    _write_event(
        tmp_path,
        event_id="e3",
        scored_at_us=1_700_000_000_000_200,
        account_id="C1",
        dest_id="M1",
        txn_type="PAYMENT",
        amount=250.0,
        model_version="1",
    )

    distinct = _distinct_transactions(_load(tmp_path))

    assert len(distinct) == 2
    same_key_row = distinct[distinct["amount"] == 100.0].iloc[0]
    assert same_key_row["predictions"] == 2
    other_row = distinct[distinct["amount"] == 250.0].iloc[0]
    assert other_row["predictions"] == 1


def test_compare_rows_matches_natural_key_across_different_event_ids_and_models(
    tmp_path: Path,
) -> None:
    _write_event(
        tmp_path,
        event_id="e1",
        scored_at_us=1_700_000_000_000_000,
        account_id="C1",
        dest_id="M1",
        txn_type="PAYMENT",
        amount=100.0,
        model_name="paysim_fraud_clf",
        model_version="1",
        fraud_score=0.2,
    )
    _write_event(
        tmp_path,
        event_id="e2",
        scored_at_us=1_700_000_000_100_000,
        account_id="C1",
        dest_id="M1",
        txn_type="PAYMENT",
        amount=100.0,
        model_name="paysim_fraud_clf",
        model_version="2",
        fraud_score=0.8,
    )
    # Same account/dest/type, different amount -- must NOT be pulled into the comparison.
    _write_event(
        tmp_path,
        event_id="e3",
        scored_at_us=1_700_000_000_200_000,
        account_id="C1",
        dest_id="M1",
        txn_type="PAYMENT",
        amount=999.0,
        model_name="paysim_fraud_clf",
        model_version="1",
    )

    df = _load(tmp_path)
    natural_key = {"account_id": "C1", "dest_id": "M1", "txn_type": "PAYMENT", "amount": 100.0}

    compared = _compare_rows(df, natural_key)

    assert list(compared["model@version"]) == ["paysim_fraud_clf@1", "paysim_fraud_clf@2"]
    assert list(compared["fraud_score"]) == [0.2, 0.8]
    assert compared["timestamp"].is_monotonic_increasing


def test_apply_search_filters_combine_with_and(tmp_path: Path) -> None:
    _write_event(
        tmp_path,
        event_id="e1",
        scored_at_us=1_700_000_000_000_000,
        account_id="C1",
        txn_type="PAYMENT",
    )
    _write_event(
        tmp_path,
        event_id="e2",
        scored_at_us=1_700_000_000_000_100,
        account_id="C2",
        txn_type="PAYMENT",
    )

    df = _load(tmp_path)
    matched = _apply_search_filters(df, name_orig="C1", name_dest="", txn_type="PAYMENT")

    assert list(matched["event_id"]) == ["e1"]


def test_has_any_filter() -> None:
    assert _has_any_filter(name_orig="", name_dest="", txn_type="All") is False
    assert _has_any_filter(name_orig="C1", name_dest="", txn_type="All") is True
    assert _has_any_filter(name_orig="", name_dest="M1", txn_type="All") is True
    assert _has_any_filter(name_orig="", name_dest="", txn_type="PAYMENT") is True
