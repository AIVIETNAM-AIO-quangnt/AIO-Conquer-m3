"""The typed request/response models behind ``/predict``, ``/model_info``, and
the deprecated ``/invocations`` alias.

:class:`TransactionIn` is **generated** from
:class:`conquer3.core.types.TransactionEvent`'s own dataclass fields via
``dataclasses.fields()`` -- the same "generated, not hand-duplicated" rule
``pipelines/pathway/schemas.py`` follows for Pathway's schema and
``db/ddl_gen.py`` follows for the warehouse DDL. Adding a field to
``TransactionEvent`` therefore extends the REST contract and its OpenAPI schema
automatically; forgetting to describe that new field fails loudly at import time
(see ``_FIELD_DOCS``) rather than shipping an undocumented field.

The caller sends a *raw transaction*, not features -- pushing feature
computation onto the caller would destroy the whole premise of Layer 0 (one pure
module computes every feature).

These are pydantic models rather than an MLflow ``ModelSignature`` because
BentoML derives the served OpenAPI 3 document from them: the field types,
defaults, and ``description``s written here are exactly what a client reads at
``/docs.json``. That is the difference the migration buys -- MLflow's scoring
server took ``/invocations`` as a raw request body and could describe none of it.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, Field, create_model

from conquer3.core.types import TransactionEvent

__all__ = [
    "TXN_FIELD_NAMES",
    "LegacyParams",
    "LegacyResponse",
    "ModelInfoResponse",
    "ScoreResult",
    "TransactionIn",
    "to_transaction_events",
]

# TransactionEvent uses `from __future__ import annotations`, so
# dataclasses.fields()[i].type is the *string* annotation, not the real type
# object -- same assumption pipelines/pathway/schemas.py makes for the same reason.
_PY_TYPE_BY_ANNOTATION: dict[str, type] = {"str": str, "float": float, "int": int}

TXN_FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in dataclasses.fields(TransactionEvent))

# Keyed by field name so the OpenAPI schema documents every field. Deliberately
# exhaustive: _build_transaction_in() raises if this drifts from the dataclass.
_FIELD_DOCS: dict[str, str] = {
    "event_id": "Caller-assigned unique id for this transaction. Echoed back on the result.",
    "account_id": "Originating account (PaySim `nameOrig`). State is keyed by this.",
    "dest_id": "Destination account (PaySim `nameDest`). An `M` prefix marks a merchant.",
    "txn_type": "One of TRANSFER, PAYMENT, CASH_OUT, DEBIT, CASH_IN.",
    "amount": "Transaction amount.",
    "oldbalance_org": "Originating account balance before the transaction.",
    "newbalance_orig": "Originating account balance after the transaction.",
    "oldbalance_dest": "Destination account balance before the transaction.",
    "newbalance_dest": "Destination account balance after the transaction.",
    "event_ts_us": (
        "Event time, Unix microseconds. Always supplied by the caller, never read "
        "from the clock server-side, so batch and online paths stay bit-identical."
    ),
    "step": "PaySim simulation step (hours since the start of the simulation).",
}


def _build_transaction_in() -> type[BaseModel]:
    missing = set(TXN_FIELD_NAMES) - set(_FIELD_DOCS)
    extra = set(_FIELD_DOCS) - set(TXN_FIELD_NAMES)
    if missing or extra:
        raise TypeError(
            "serving.api_models._FIELD_DOCS is out of sync with TransactionEvent: "
            f"undocumented={sorted(missing)}, unknown={sorted(extra)}."
        )

    definitions: dict[str, Any] = {}
    for f in dataclasses.fields(TransactionEvent):
        annotation = f.type
        assert isinstance(annotation, str), (
            "core/types.py must keep `from __future__ import annotations` for this "
            "string-annotation assumption to hold"
        )
        try:
            py_type = _PY_TYPE_BY_ANNOTATION[annotation]
        except KeyError as exc:
            raise TypeError(
                f"TransactionEvent.{f.name} has annotation {annotation!r}, which "
                "serving.api_models doesn't know how to map onto a JSON field type "
                "-- add it to _PY_TYPE_BY_ANNOTATION."
            ) from exc
        definitions[f.name] = (
            Annotated[py_type, Field(description=_FIELD_DOCS[f.name])],
            ...,
        )

    return create_model(
        "TransactionIn",
        __doc__="One raw transaction to score. All fields are required.",
        **definitions,
    )


if TYPE_CHECKING:
    # `create_model` produces a class at runtime, but a *variable* to a type
    # checker, and BentoML needs TransactionIn in an annotation position (that is
    # how it derives the OpenAPI request schema). Declaring the empty base here
    # gives mypy a usable type without hand-restating the fields -- restating them
    # is precisely the duplication this module exists to prevent. Nothing reads a
    # field off it statically; `to_transaction_events` goes through getattr with
    # the generated names.
    class TransactionIn(BaseModel):
        pass

else:
    TransactionIn = _build_transaction_in()


def to_transaction_events(rows: list[TransactionIn]) -> list[TransactionEvent]:
    """Adapt validated request rows onto the core dataclass.

    pydantic has already coerced and validated every field by the time this runs,
    so this is a pure shape change with no second validation pass.
    """
    return [
        TransactionEvent(**{name: getattr(row, name) for name in TXN_FIELD_NAMES}) for row in rows
    ]


class ScoreResult(BaseModel):
    """The score assigned to one transaction."""

    event_id: str = Field(description="Echoes the request's `event_id`.")
    fraud_score: float = Field(description="Predicted fraud probability, 0.0-1.0.")
    decision: str = Field(
        description="`FRAUD` when `fraud_score >= C3_DECISION_THRESHOLD`, else `LEGIT`."
    )
    had_prev_state: bool = Field(
        description="False on an account's first-ever transaction (cold start)."
    )
    # Undefined on an account's first-ever transaction -- cold-start policy, the
    # same rule core.features applies to every recency feature.
    seconds_since_last_txn: float | None = Field(
        default=None,
        description="Seconds since this account's previous transaction; null on cold start.",
    )
    model_version: str = Field(description="Registry version of the champion that scored this.")
    feature_schema_version: int = Field(description="Feature schema the score was computed under.")
    degraded: bool = Field(
        description=(
            "True when the champion was resolved from the local cache because the "
            "registry was unreachable at boot. Scores are still correct; the model "
            "may not be the newest champion."
        )
    )


class ModelInfoResponse(BaseModel):
    """The champion currently being served -- a :class:`ModelRef`, as JSON."""

    name: str = Field(description="Registered model name.")
    version: str = Field(description="Registry version currently loaded by this worker.")
    run_id: str = Field(description="MLflow run that produced this version.")
    alias: str = Field(description="Registry alias that resolved to it (normally `champion`).")
    tags: dict[str, str] = Field(default_factory=dict, description="Model version tags.")
    degraded: bool = Field(
        default=False, description="True when resolved from the local cache; see `/predict`."
    )


class LegacyParams(BaseModel):
    """The `params` object of MLflow's scoring-server envelope."""

    dry_run: bool = False
    op: str = "score"


class LegacyResponse(BaseModel):
    """MLflow's scoring-server response envelope.

    `predictions` holds :class:`ScoreResult` rows for `op="score"` and a single
    :class:`ModelInfoResponse` row for `op="model_info"` -- heterogeneous by
    design, which is precisely why the typed `/predict` and `/model_info` routes
    replaced it.
    """

    predictions: list[dict[str, Any]]
