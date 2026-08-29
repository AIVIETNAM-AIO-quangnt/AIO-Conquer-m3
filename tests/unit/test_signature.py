"""serving/signature.py is generated from TransactionEvent, not hand-duplicated --
these tests prove the generation actually stays in sync, the same property
tests/unit/test_pathway_schemas.py proves for Pathway's schema."""

from __future__ import annotations

import dataclasses

from mlflow.types import DataType

from conquer3.core.types import TransactionEvent
from conquer3.serving.signature import TXN_FIELD_NAMES, build_signature


def test_txn_field_names_match_transaction_event_dataclass_fields() -> None:
    assert tuple(f.name for f in dataclasses.fields(TransactionEvent)) == TXN_FIELD_NAMES


def test_input_schema_has_one_required_column_per_transaction_event_field() -> None:
    signature = build_signature()
    cols = signature.inputs.inputs
    assert [c.name for c in cols] == list(TXN_FIELD_NAMES)
    assert all(c.required for c in cols)


def test_input_schema_types_match_transaction_event_annotations() -> None:
    expected = {"str": DataType.string, "float": DataType.double, "int": DataType.long}
    signature = build_signature()
    by_name = {c.name: c for c in signature.inputs.inputs}
    for f in dataclasses.fields(TransactionEvent):
        assert isinstance(f.type, str)
        assert by_name[f.name].type == expected[f.type]


def test_output_schema_has_the_score_response_columns() -> None:
    signature = build_signature()
    names = [c.name for c in signature.outputs.inputs]
    assert names == [
        "event_id",
        "fraud_score",
        "decision",
        "had_prev_state",
        "seconds_since_last_txn",
        "model_version",
        "feature_schema_version",
        "degraded",
    ]


def test_param_schema_has_dry_run_and_op_with_documented_defaults() -> None:
    signature = build_signature()
    params = {p.name: p for p in signature.params.params}
    assert params["dry_run"].default is False
    assert params["dry_run"].dtype == DataType.boolean
    assert params["op"].default == "score"
    assert params["op"].dtype == DataType.string
