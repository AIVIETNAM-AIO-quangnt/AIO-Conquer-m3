"""The custom pyfunc model served by MLflow's own scoring server.

The stock scoring server is stateless; everything Layer 5 actually *does* lives in
this one class. ``load_context`` runs once per uvicorn worker **process** (a fresh
Python interpreter for each of ``--workers N``), never per request. It loads the
champion via ``mlflow.sklearn.load_model`` -- the flavor-specific loader, not
``mlflow.pyfunc.load_model`` -- so ``self._pipe`` is the raw sklearn estimator with
a real ``predict_proba``, not a pyfunc wrapper whose ``.predict()`` would return
class labels instead of a score.

**Thread-safety.** ``/invocations`` is ``async def`` but dispatches through
``await asyncio.to_thread(...)``, so concurrent requests run ``predict`` on
parallel threads inside this one process, on top of parallel *processes* across
``--workers N`` (plan §8.4). This class holds no per-request mutable instance
state: every request builds its own ``TransactionEvent``/``AccountState``/
``FeatureVector`` locally. Same-account races are handled downstream by
``RedisStateStore``'s monotonic CAS, not by locking here.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, ClassVar

import mlflow.sklearn
import pandas as pd
from mlflow.pyfunc.model import PythonModel, PythonModelContext

from conquer3.config.settings import get_settings
from conquer3.contracts.events import ScoredEvent
from conquer3.contracts.model_registry import ModelRef
from conquer3.core.features import compute
from conquer3.core.schema import FEATURE_SCHEMA_VERSION
from conquer3.core.types import FeatureVector, TransactionEvent
from conquer3.serving.event_sink import JsonlEventSink
from conquer3.serving.signature import TXN_FIELD_NAMES
from conquer3.serving.state_store import RedisStateStore

__all__ = ["FraudScorerModel"]

# TransactionEvent uses `from __future__ import annotations` (string annotations),
# same assumption serving/signature.py and pipelines/pathway/schemas.py make.
_TXN_FIELD_CASTS: dict[str, type] = {
    f.name: {"str": str, "float": float, "int": int}[f.type]  # type: ignore[index]
    for f in dataclasses.fields(TransactionEvent)
}


class FraudScorerModel(PythonModel):
    _KNOWN_OPS: ClassVar[frozenset[str]] = frozenset({"score", "model_info"})

    def load_context(self, context: PythonModelContext) -> None:
        self._pipe = mlflow.sklearn.load_model(context.artifacts["champion"])
        self._ref = ModelRef(
            **json.loads(Path(context.artifacts["ref"]).read_text(encoding="utf-8"))
        )
        settings = get_settings()
        self._threshold = settings.serving.decision_threshold
        self._state = RedisStateStore(redis_settings=settings.redis, state_settings=settings.state)
        self._sink = JsonlEventSink(event_settings=settings.event)

    def predict(
        self,
        context: PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        params = params or {}
        op = params.get("op", "score")
        if op not in self._KNOWN_OPS:
            raise ValueError(f"unknown op {op!r}; expected one of {sorted(self._KNOWN_OPS)}")
        if op == "model_info":
            return pd.DataFrame([dataclasses.asdict(self._ref)])
        return self._score(model_input, dry_run=bool(params.get("dry_run", False)))

    def _score(self, model_input: pd.DataFrame, *, dry_run: bool) -> pd.DataFrame:
        txns = [
            TransactionEvent(
                **{name: _TXN_FIELD_CASTS[name](row[name]) for name in TXN_FIELD_NAMES}
            )
            for row in model_input.to_dict(orient="records")
        ]

        # Group by account_id, fold in event_ts_us order -- back-to-back
        # transactions for the same account in one payload must see each other's
        # state (plan §8.4). Across accounts this parallelizes safely; within one
        # account it does not, so the fold below is strictly sequential per group.
        by_account: dict[str, list[int]] = {}
        for i, txn in enumerate(txns):
            by_account.setdefault(txn.account_id, []).append(i)

        responses: list[dict[str, Any] | None] = [None] * len(txns)
        for account_id, idxs in by_account.items():
            idxs.sort(key=lambda i: (txns[i].event_ts_us, txns[i].event_id))
            # Reading current state is not a side effect a dry_run needs to avoid --
            # only the write side (commit + event append) is skipped below, so a
            # dry_run reproduces the score a live request would actually get.
            prev = self._state.get(account_id)
            for i in idxs:
                txn = txns[i]
                features, new_state = compute(txn, prev)
                proba = self._predict_proba(features)
                new_state = dataclasses.replace(new_state, last_fraud_score=proba)
                decision = "FRAUD" if proba >= self._threshold else "LEGIT"
                had_prev_state = prev is not None

                if not dry_run:
                    self._state.commit(new_state)
                    self._sink.append(
                        self._to_scored_event(txn, features, proba, decision, had_prev_state)
                    )

                responses[i] = {
                    "event_id": txn.event_id,
                    "fraud_score": proba,
                    "decision": decision,
                    "had_prev_state": had_prev_state,
                    "seconds_since_last_txn": features.values["seconds_since_last_txn"],
                    "model_version": self._ref.version,
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "degraded": self._ref.degraded,
                }
                prev = new_state

        assert all(r is not None for r in responses)
        return pd.DataFrame(responses)

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
