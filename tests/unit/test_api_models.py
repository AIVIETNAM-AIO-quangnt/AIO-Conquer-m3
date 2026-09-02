"""The REST contract is generated from TransactionEvent, not hand-maintained.

These tests are what makes that claim enforceable: they fail the moment the
pydantic request model, the OpenAPI schema, and the core dataclass disagree.
"""

from __future__ import annotations

import dataclasses

import pytest

from conquer3.core.types import TransactionEvent
from conquer3.serving.api_models import (
    TXN_FIELD_NAMES,
    LegacyParams,
    ScoreResult,
    TransactionIn,
    to_transaction_events,
)

_VALID_ROW = {
    "event_id": "e1",
    "account_id": "C1",
    "dest_id": "M900",
    "txn_type": "TRANSFER",
    "amount": 181.0,
    "oldbalance_org": 181.0,
    "newbalance_orig": 0.0,
    "oldbalance_dest": 0.0,
    "newbalance_dest": 181.0,
    "event_ts_us": 1_700_000_000_000_000,
    "step": 1,
}


def test_txn_field_names_match_the_dataclass_exactly() -> None:
    assert tuple(f.name for f in dataclasses.fields(TransactionEvent)) == TXN_FIELD_NAMES


def test_transaction_in_has_exactly_the_dataclass_fields_all_required() -> None:
    schema = TransactionIn.model_json_schema()
    assert list(schema["properties"]) == list(TXN_FIELD_NAMES)
    assert set(schema["required"]) == set(TXN_FIELD_NAMES)


def test_every_field_carries_a_description_for_the_openapi_document() -> None:
    """An undocumented field would ship a blank row in the published API docs."""
    properties = TransactionIn.model_json_schema()["properties"]
    undocumented = [n for n in TXN_FIELD_NAMES if not properties[n].get("description")]
    assert not undocumented


def test_field_json_types_follow_the_dataclass_annotations() -> None:
    expected = {"str": "string", "float": "number", "int": "integer"}
    properties = TransactionIn.model_json_schema()["properties"]
    for f in dataclasses.fields(TransactionEvent):
        assert isinstance(f.type, str)
        assert properties[f.name]["type"] == expected[f.type], f.name


def test_round_trips_onto_the_core_dataclass() -> None:
    (txn,) = to_transaction_events([TransactionIn(**_VALID_ROW)])
    assert isinstance(txn, TransactionEvent)
    assert dataclasses.asdict(txn) == _VALID_ROW


def test_missing_field_is_rejected_at_the_door() -> None:
    from pydantic import ValidationError

    incomplete = {k: v for k, v in _VALID_ROW.items() if k != "amount"}
    with pytest.raises(ValidationError, match="amount"):
        TransactionIn(**incomplete)


def test_score_result_marks_only_the_cold_start_field_optional() -> None:
    """`seconds_since_last_txn` is undefined on an account's first transaction;
    every other result field is always present."""
    schema = ScoreResult.model_json_schema()
    assert set(schema["properties"]) - set(schema["required"]) == {"seconds_since_last_txn"}


def test_legacy_params_defaults_match_the_previous_mlflow_signature() -> None:
    params = LegacyParams()
    assert params.op == "score"
    assert params.dry_run is False
