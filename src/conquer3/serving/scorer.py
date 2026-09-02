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
from conquer3.core.schema import FEATURE_SCHEMA_VERSION
from conquer3.core.types import FeatureVector, TransactionEvent
from conquer3.serving.api_models import ScoreResult
from conquer3.serving.event_sink import JsonlEventSink
from conquer3.serving.state_store import RedisStateStore
from conquer3.telemetry.otel import get_meter, get_tracer

__all__ = ["FraudScorer"]


class FraudScorer:
    def __init__(
        self,
        *,
        pipe: Any,
        ref: ModelRef,
        threshold: float,
        state: RedisStateStore,
        sink: JsonlEventSink,
    ) -> None:
        self._pipe = pipe
        self._ref = ref
        self._threshold = threshold
        self._state = state
        self._sink = sink
        self._init_instruments()

    @property
    def ref(self) -> ModelRef:
        """The champion this scorer is serving."""
        return self._ref

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
                        proba = self._predict_proba(features)
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
                                    txn, features, proba, decision, had_prev_state
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
                        model_version=self._ref.version,
                        feature_schema_version=FEATURE_SCHEMA_VERSION,
                        degraded=self._ref.degraded,
                    )
                    prev = new_state

        assert all(r is not None for r in responses)
        return [r for r in responses if r is not None]

    def _predict_proba(self, features: FeatureVector) -> float:
        row = pd.DataFrame([features.model_inputs()])
        return float(self._pipe.predict_proba(row)[0, 1])

    def _to_scored_event(
        self,
        txn: TransactionEvent,
        features: FeatureVector,
        proba: float,
        decision: str,
        had_prev_state: bool,
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
            model_name=self._ref.name,
            model_version=self._ref.version,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            transaction=dataclasses.asdict(txn),
            features=features.as_dict(),
        )
