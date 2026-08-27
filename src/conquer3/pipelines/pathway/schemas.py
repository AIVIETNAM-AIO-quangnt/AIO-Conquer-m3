"""Pathway pw.Schema for staged transaction rows, built from
conquer3.core.types.TransactionEvent's own dataclass fields via
dataclasses.fields() -- so this schema cannot drift from TransactionEvent's shape
the way a hand-written parallel schema could. Carries no feature logic (section 0).
"""

from __future__ import annotations

import dataclasses
from typing import cast

import pathway as pw

from conquer3.core.types import TransactionEvent

__all__ = ["TransactionEventSchema"]

# TransactionEvent uses `from __future__ import annotations`, so
# dataclasses.fields()[i].type is the *string* annotation, not the real type object.
_PY_TYPE_BY_ANNOTATION: dict[str, type] = {"str": str, "float": float, "int": int}


def _build_transaction_event_schema() -> type[pw.Schema]:
    kwargs: dict[str, type] = {}
    for f in dataclasses.fields(TransactionEvent):
        annotation = f.type
        assert isinstance(annotation, str), (
            "core/types.py must keep `from __future__ import annotations` for this "
            "string-annotation assumption to hold"
        )
        try:
            kwargs[f.name] = _PY_TYPE_BY_ANNOTATION[annotation]
        except KeyError as exc:
            raise TypeError(
                f"TransactionEvent.{f.name} has annotation {annotation!r}, which "
                "pipelines.pathway.schemas doesn't know how to map onto a Pathway "
                "column type -- add it to _PY_TYPE_BY_ANNOTATION."
            ) from exc
    return cast("type[pw.Schema]", pw.schema_from_types(_name="TransactionEventSchema", **kwargs))


TransactionEventSchema: type[pw.Schema] = _build_transaction_event_schema()
