"""The ModelSignature for ``/invocations``.

**Input** is generated from :class:`conquer3.core.types.TransactionEvent`'s own
dataclass fields via ``dataclasses.fields()`` -- the same "generated, not
hand-duplicated" rule ``pipelines/pathway/schemas.py`` follows for Pathway's schema
and ``db/ddl_gen.py`` follows for the warehouse DDL. The caller sends a *raw
transaction*, not features -- pushing feature computation onto the caller would
destroy the whole premise of Layer 0 (one pure module computes every feature).

**Output** describes the ``op="score"`` response shape. ``op="model_info"``
deliberately returns a different, smaller frame (the resolved ``ModelRef``) --
confirmed by reading the installed scoring server's source (``mlflow==3.15.1``)
that ``predictions_to_json`` serializes whatever ``predict()`` returns with no
output-schema enforcement at request time, so this never conflicts with the
declared signature; ``outputs=`` here is documentation, not an invocations-time
gate the way ``inputs=`` is (input *is* enforced, at the door, before predict()
runs at all).

**Params** are the only channel MLflow gives a stateless caller for non-row
arguments -- ``dry_run`` and ``op`` live here because there is nowhere else to
put them (no custom routes exist, and none can be added; see the architecture
plan's Layer 5 section).
"""

from __future__ import annotations

import dataclasses

from mlflow.models import ModelSignature
from mlflow.types import ColSpec, DataType, ParamSchema, ParamSpec, Schema, TensorSpec

from conquer3.core.types import TransactionEvent

__all__ = ["TXN_FIELD_NAMES", "build_signature"]

# TransactionEvent uses `from __future__ import annotations`, so
# dataclasses.fields()[i].type is the *string* annotation, not the real type
# object -- same assumption pipelines/pathway/schemas.py makes for the same reason.
_MLFLOW_TYPE_BY_ANNOTATION: dict[str, DataType] = {
    "str": DataType.string,
    "float": DataType.double,
    "int": DataType.long,
}

TXN_FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in dataclasses.fields(TransactionEvent))


def _input_schema() -> Schema:
    cols: list[ColSpec | TensorSpec] = []
    for f in dataclasses.fields(TransactionEvent):
        annotation = f.type
        assert isinstance(annotation, str), (
            "core/types.py must keep `from __future__ import annotations` for this "
            "string-annotation assumption to hold"
        )
        try:
            dtype = _MLFLOW_TYPE_BY_ANNOTATION[annotation]
        except KeyError as exc:
            raise TypeError(
                f"TransactionEvent.{f.name} has annotation {annotation!r}, which "
                "serving.signature doesn't know how to map onto an MLflow column "
                "type -- add it to _MLFLOW_TYPE_BY_ANNOTATION."
            ) from exc
        cols.append(ColSpec(dtype, f.name))
    return Schema(cols)


def _output_schema() -> Schema:
    return Schema(
        [
            ColSpec(DataType.string, "event_id"),
            ColSpec(DataType.double, "fraud_score"),
            ColSpec(DataType.string, "decision"),
            ColSpec(DataType.boolean, "had_prev_state"),
            # Undefined on an account's first-ever transaction -- cold-start policy,
            # same rule core.features applies to every recency feature.
            ColSpec(DataType.double, "seconds_since_last_txn", required=False),
            ColSpec(DataType.string, "model_version"),
            ColSpec(DataType.long, "feature_schema_version"),
            ColSpec(DataType.boolean, "degraded"),
        ]
    )


def _param_schema() -> ParamSchema:
    return ParamSchema(
        [
            # Score without writing state or appending an event -- the skew audit's
            # replay path, so re-scoring for audit purposes can never corrupt live
            # Redis (plan §8.2).
            ParamSpec("dry_run", DataType.boolean, False),
            # "score" -> normal scoring. "model_info" -> the resolved ModelRef as a
            # one-row frame, replacing the /model_info route BentoML had (there are
            # no custom routes to add one to here).
            ParamSpec("op", DataType.string, "score"),
        ]
    )


def build_signature() -> ModelSignature:
    return ModelSignature(inputs=_input_schema(), outputs=_output_schema(), params=_param_schema())
