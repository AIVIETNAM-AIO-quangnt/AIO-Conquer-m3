"""TransactionEventSchema must never drift from conquer3.core.types.TransactionEvent
-- it's built from the dataclass's own fields (see pipelines/pathway/schemas.py), so
this is a regression guard for the day a new field's annotation isn't in
_PY_TYPE_BY_ANNOTATION's lookup table.
"""

from __future__ import annotations

import dataclasses

from conquer3.core.types import TransactionEvent
from conquer3.pipelines.pathway.schemas import _PY_TYPE_BY_ANNOTATION, TransactionEventSchema


def test_schema_columns_match_transaction_event_fields() -> None:
    expected_names = [f.name for f in dataclasses.fields(TransactionEvent)]
    assert TransactionEventSchema.column_names() == expected_names


def test_schema_types_match_transaction_event_annotations() -> None:
    expected = {
        f.name: _PY_TYPE_BY_ANNOTATION[f.type] for f in dataclasses.fields(TransactionEvent)
    }
    assert TransactionEventSchema.typehints() == expected
