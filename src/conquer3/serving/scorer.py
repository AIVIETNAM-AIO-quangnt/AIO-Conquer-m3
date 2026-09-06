"""Everything Layer 5 actually *does*, with no web framework in sight.

This class knows nothing about BentoML, HTTP, or MLflow's serving stack: it takes
already-validated :class:`TransactionEvent`s and returns :class:`ScoreResult`s.
``serving/service.py`` owns the transport, ``serving/champion.py`` owns model
resolution, and this owns the fold. Keeping the boundary there is what lets
``scripts/smoke/_layer7_emit_spans.py`` exercise the real instrumentation with
fake collaborators and no infrastructure at all.

It loads nothing itself -- the champion pipeline and its :class:`ModelRef` are
constructor arguments. ``pipe`` must be the raw sklearn estimator (loaded with
``mlflow.sklearn.load_model``, never ``mlflow.pyfunc.load_model``) so it has a
real ``predict_proba``; a pyfunc wrapper's ``.predict()`` would return class
labels instead of a score.

**Thread-safety.** BentoML dispatches concurrent requests onto threads within one
worker process, on top of parallel *processes* across ``workers=N``. This class
holds no per-request mutable instance state: every request builds its own
``TransactionEvent``/``AccountState``/``FeatureVector`` locally. Same-account
races are handled downstream by ``RedisStateStore``'s monotonic CAS, not by
locking here.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import pandas as pd

from conquer3.contracts.events import ScoredEvent
from conquer3.contracts.model_registry import ModelRef
from conquer3.core.features import compute
from conquer3.core.schema import EXTERNAL_MODEL_FEATURES, FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from conquer3.core.types import FeatureVector, TransactionEvent
from conquer3.serving.api_models import ScoreResult
from conquer3.serving.event_sink import JsonlEventSink
from conquer3.serving.state_store import RedisStateStore
from conquer3.telemetry.otel import get_meter, get_tracer

__all__ = ["GOLD_SCHEMA_COLUMNS", "Champion", "FraudScorer"]

# Every column name gold.txn_features can hold: FEATURE_NAMES (what core.features
# actually computes live) plus EXTERNAL_MODEL_FEATURES (reserved for other
# registered model families' own inputs -- see core/schema.py's docstring on that
# constant). A model's declared input schema overlapping *this* set, not just
# FEATURE_NAMES, is the bar _predict_proba uses to decide "pluggable at all" --
# broader than "computed live" on purpose, since most EXTERNAL_MODEL_FEATURES
# columns are NULL at request time until a dedicated backfill job exists for that
# model family (see that constant's docstring), yet the model is still a real
# gold-schema citizen, not a foreign one.
#
# Public (not module-private) because serving/service.py's pre-load step reuses
# it verbatim to *exclude* a hopeless model from the swap pool up front, rather
# than let it 500 on every /predict after being switched to -- see that
# module's pre-load validation.
GOLD_SCHEMA_COLUMNS: frozenset[str] = frozenset(FEATURE_NAMES) | {
    name for name, _ in EXTERNAL_MODEL_FEATURES
}


def _extract_score(result: Any) -> float:
    """Normalizes a bare pyfunc model's ``.predict()`` return (used only when
    the loaded ``Champion.pipe`` has no ``predict_proba`` -- see
    ``_predict_proba``) down to a single float.

    A hand-authored ``python_function``-only model's ``predict()`` is its own
    contract, not this scorer's -- it may return a 1-D array of one score per
    row, a 2-D ``[P(legit), P(fraud)]`` array like a native ``predict_proba``,
    or a single-column frame/series. Only the shape is normalized here; the
    value itself is trusted as-is (see the ``probability_type: raw|calibrated``
    tags real registrations like ``paysim-fraud-xgb-baseline`` carry -- this
    scorer has no way to re-derive a calibrated probability from a raw margin
    itself, nor any reason to assume every custom model returns one).
    """
    import numpy as np

    arr = np.asarray(result)
    if arr.ndim == 2:
        return float(arr[0, -1])
    return float(arr.reshape(-1)[0])


@dataclasses.dataclass(frozen=True, slots=True)
class Champion:
    pipe: Any
    ref: ModelRef
    # The model's own declared input columns (from its MLflow signature), or
    # None if it has no named signature -- see contracts.model_registry's
    # model_input_columns. Drives the gold-schema-overlap check and column
    # padding _predict_proba does, so a model registered outside publish_model
    # with a different (but at least partially gold-schema-overlapping) feature
    # set can still be served.
    input_columns: tuple[str, ...] | None = None
    # The subset of input_columns MLflow declares as string-typed -- see
    # contracts.model_registry.model_string_input_columns. Cast to pandas
    # `category` dtype before scoring; empty for a model with no string
    # columns (most of them) or no signature at all.
    categorical_columns: frozenset[str] = dataclasses.field(default_factory=frozenset)


class FraudScorer:
    def __init__(
        self,
        *,
        pipe: Any,
        ref: ModelRef,
        threshold: float,
        state: RedisStateStore,
        sink: JsonlEventSink,
        input_columns: tuple[str, ...] | None = None,
        categorical_columns: frozenset[str] = frozenset(),
    ) -> None:
        self._champion = Champion(
            pipe=pipe, ref=ref, input_columns=input_columns, categorical_columns=categorical_columns
        )
        self._threshold = threshold
        self._state = state
        self._sink = sink
        self._init_instruments()

    @property
    def ref(self) -> ModelRef:
        """The champion this scorer is serving."""
        return self._champion.ref

    def reload(
        self,
        *,
        pipe: Any,
        ref: ModelRef,
        input_columns: tuple[str, ...] | None = None,
        categorical_columns: frozenset[str] = frozenset(),
    ) -> None:
        """Atomically swaps in a newly resolved champion, in place -- no restart.

        A single attribute assignment is atomic under CPython's GIL, so a request
        already in flight sees either the whole old ``Champion`` or the whole new
        one, never a torn pipe/ref pairing -- as long as callers snapshot
        ``self._champion`` once into a local rather than re-reading it mid-request
        (``score`` does exactly that).
        """
        self._champion = Champion(
            pipe=pipe, ref=ref, input_columns=input_columns, categorical_columns=categorical_columns
        )

    def _init_instruments(self) -> None:
        """Tracer + the scorer's share of plan §10's "Custom instruments" list.

        Split out from ``__init__`` so tests and ``scripts/smoke/_layer7_emit_spans.py``
        can call it directly on a hand-assembled instance without a real model,
        Redis, or event dir -- get_tracer/get_meter are safe no-ops whether or not
        init_telemetry has run.
        """
        self._tracer = get_tracer(__name__)
        meter = get_meter(__name__)
        self._score_latency_histogram = meter.create_histogram(
            "c3_score_latency_ms",
            unit="ms",
            description="Wall-clock time to compute features and score one transaction",
        )
        self._fraud_score_histogram = meter.create_histogram(
            "c3_fraud_score", description="Predicted fraud probability, per scored transaction"
        )
        self._decision_counter = meter.create_counter(
            "c3_decision_total", description="Scored transactions by decision"
        )
        self._feature_null_counter = meter.create_counter(
            "c3_feature_null_total", description="Null feature values, by feature name"
        )

    def score(self, txns: list[TransactionEvent], *, dry_run: bool = False) -> list[ScoreResult]:
        # Snapshot once, up front: a concurrent reload() must never let one batch
        # request see a mix of the old pipe and the new ref (or vice versa) across
        # its own rows -- see reload()'s docstring.
        champion = self._champion

        # Group by account_id, fold in event_ts_us order -- back-to-back
        # transactions for the same account in one payload must see each other's
        # state (plan §8.4). Across accounts this parallelizes safely; within one
        # account it does not, so the fold below is strictly sequential per group.
        by_account: dict[str, list[int]] = {}
        for i, txn in enumerate(txns):
            by_account.setdefault(txn.account_id, []).append(i)

        responses: list[ScoreResult | None] = [None] * len(txns)
        with self._tracer.start_as_current_span(
            "score_batch", attributes={"dry_run": dry_run, "row_count": len(txns)}
        ):
            for account_id, idxs in by_account.items():
                idxs.sort(key=lambda i: (txns[i].event_ts_us, txns[i].event_id))
                # Reading current state is not a side effect a dry_run needs to
                # avoid -- only the write side (commit + event append) is skipped
                # below, so a dry_run reproduces the score a live request would
                # actually get.
                with self._tracer.start_as_current_span(
                    "redis_get", attributes={"account_id": account_id}
                ):
                    prev = self._state.get(account_id)
                for i in idxs:
                    txn = txns[i]
                    with self._tracer.start_as_current_span(
                        "predict",
                        attributes={"account_id": account_id, "event_id": txn.event_id},
                    ):
                        start = time.perf_counter()
                        features, new_state = compute(txn, prev)
                        proba = self._predict_proba(features, champion)
                        self._score_latency_histogram.record((time.perf_counter() - start) * 1000)

                    new_state = dataclasses.replace(new_state, last_fraud_score=proba)
                    decision = "FRAUD" if proba >= self._threshold else "LEGIT"
                    had_prev_state = prev is not None
                    self._fraud_score_histogram.record(proba)
                    self._decision_counter.add(1, {"decision": decision})
                    for feature_name, value in features.values.items():
                        if value is None:
                            self._feature_null_counter.add(1, {"feature": feature_name})

                    if not dry_run:
                        with self._tracer.start_as_current_span(
                            "redis_set", attributes={"account_id": account_id}
                        ):
                            self._state.commit(new_state)
                        with self._tracer.start_as_current_span(
                            "file_append", attributes={"account_id": account_id}
                        ):
                            self._sink.append(
                                self._to_scored_event(
                                    txn, features, proba, decision, had_prev_state, champion.ref
                                )
                            )

                    seconds_since = features.values["seconds_since_last_txn"]
                    responses[i] = ScoreResult(
                        event_id=txn.event_id,
                        fraud_score=proba,
                        decision=decision,
                        had_prev_state=had_prev_state,
                        seconds_since_last_txn=(
                            None if seconds_since is None else float(seconds_since)
                        ),
                        model_version=champion.ref.version,
                        feature_schema_version=FEATURE_SCHEMA_VERSION,
                        degraded=champion.ref.degraded,
                    )
                    prev = new_state

        assert all(r is not None for r in responses)
        return [r for r in responses if r is not None]

    def _predict_proba(self, features: FeatureVector, champion: Champion) -> float:
        """Builds the model's input row from the computed features.

        When the champion carries a declared input schema that doesn't exactly
        match ``FEATURE_NAMES`` (a version registered outside ``publish_model``,
        e.g. a different training pipeline), the row sent is shaped to the
        model's *own* declared columns, in its own declared order -- not a
        reduced subset -- so a model expecting a fixed-width input never sees a
        truncated row. Each declared column is filled from the live-computed
        feature when ``core.features`` computes it (i.e. it's in
        ``FEATURE_NAMES``); every other declared column -- typically one of
        gold.txn_features's ``EXTERNAL_MODEL_FEATURES`` reservations, which
        ``core.features`` never computes live -- gets ``NaN``, the same "not
        available yet" value that column holds in gold.txn_features itself. A
        model with no signature (or one matching FEATURE_NAMES exactly) is
        unaffected: it still gets the full row it always did.

        Every column MLflow declared as string-typed (``champion.categorical_columns``)
        is cast to pandas ``category`` dtype before scoring -- a native
        LightGBM/XGBoost booster trained on a category column strictly
        validates the *count* of categorical-dtype columns handed to it at
        predict time, and rejects a plain-float stand-in outright (confirmed
        empirically against ``paysim-fraud-lightgbm`` v4's real ``type``
        column). A NaN-filled categorical column (unfed, reserved-only) still
        casts fine and lands on the booster's own "missing" category code.

        Only refused outright when the model declares *no* column at all that
        gold.txn_features could ever hold (``GOLD_SCHEMA_COLUMNS``) -- that's a
        model foreign to this domain entirely, not merely one this deployment
        can't fully feed yet. In practice ``serving/service.py``'s pre-load
        step already excludes such a model (and one with no signature at all)
        from the swap pool before this method ever runs; this check stays as
        defense in depth for any caller that builds a ``Champion`` a different
        way.

        Cold-start ``None``s (``core.schema.COLD_START_NULL_FEATURES`` -- e.g.
        an account's first-ever transaction) are normalized to ``float("nan")``
        here, unconditionally, before either branch below builds a row.
        ``pd.DataFrame([{"x": None}])`` infers column dtype ``object``, not
        ``float64``, and every native estimator flavor this scorer loads
        (``load_native_estimator``'s sklearn/lightgbm/xgboost) rejects an
        object-dtype numeric column outright -- confirmed empirically:
        LightGBM raises ``ValueError: pandas dtypes must be int, float or
        bool`` for exactly this input, on a model matching FEATURE_NAMES
        exactly, no partial-overlap padding involved. This is a real,
        stdlib-`None`-vs-numpy-`NaN` type boundary, not a model-compatibility
        concern, so it applies before the compatibility branch below, to
        every model regardless of its declared schema.

        Dispatches on capability, not on MLflow flavor: a real native
        estimator (whatever ``load_native_estimator`` loaded) always has
        ``predict_proba``; ``serving/service.py``'s pyfunc fallback for a
        model with no sklearn/lightgbm/xgboost flavor at all (e.g. a
        hand-authored ``python_function``-only model) hands over a bare
        ``mlflow.pyfunc.PyFuncModel``, which never does -- so its own
        ``.predict()`` is called instead and the result shape is normalized to
        a single float by ``_extract_score``.
        """
        inputs = {
            name: (float("nan") if value is None else value)
            for name, value in features.model_inputs().items()
        }
        if champion.input_columns is not None:
            if not set(champion.input_columns) & GOLD_SCHEMA_COLUMNS:
                raise ValueError(
                    f"model {champion.ref.name!r} v{champion.ref.version} declares input "
                    f"columns {champion.input_columns!r}, none of which gold.txn_features "
                    "has room for -- not a gold-schema model, nothing to score with"
                )
            inputs = {name: inputs.get(name, float("nan")) for name in champion.input_columns}
            row = pd.DataFrame([inputs], columns=champion.input_columns)
        else:
            row = pd.DataFrame([inputs])

        for col in champion.categorical_columns:
            row[col] = row[col].astype("category")

        if hasattr(champion.pipe, "predict_proba"):
            return float(champion.pipe.predict_proba(row)[0, 1])
        return _extract_score(champion.pipe.predict(row))

    def _to_scored_event(
        self,
        txn: TransactionEvent,
        features: FeatureVector,
        proba: float,
        decision: str,
        had_prev_state: bool,
        ref: ModelRef,
    ) -> ScoredEvent:
        return ScoredEvent(
            event_id=txn.event_id,
            account_id=txn.account_id,
            event_ts_us=txn.event_ts_us,
            scored_at_us=time.time_ns() // 1000,
            fraud_score=proba,
            decision=decision,
            threshold=self._threshold,
            had_prev_state=had_prev_state,
            model_name=ref.name,
            model_version=ref.version,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            transaction=dataclasses.asdict(txn),
            features=features.as_dict(),
        )
