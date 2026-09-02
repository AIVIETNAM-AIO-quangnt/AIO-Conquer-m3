"""The BentoML service: Layer 5's HTTP surface.

Three routes, all typed, all in the OpenAPI 3 document BentoML generates at
``/docs.json`` (Swagger UI at ``/``):

* ``POST /predict``      -- score transactions. The endpoint.
* ``POST /model_info``   -- the champion this worker is serving.
* ``POST /invocations``  -- **deprecated**. MLflow's scoring-server envelope,
  kept so callers written against the previous implementation keep working. It is
  a pure adapter: it unwraps the legacy body, calls the same private helpers
  ``predict``/``model_info`` call, and rewraps the result, so it holds no scoring
  logic that could drift from the typed routes.

  It does **not** call ``self.predict(...)``/``self.model_info()`` directly.
  ``@bentoml.api`` methods are dispatched through BentoML's request pipeline, not
  invoked as plain Python methods; calling one from inside another route handler
  makes the pipeline wait for a worker slot the outer request is already holding,
  which deadlocks until the request timeout (confirmed empirically -- every such
  call hung for exactly ``traffic.timeout`` before failing). The private
  ``_predict``/``_model_info`` helpers below are what both routes actually share.

Plus BentoML's own ``/livez``, ``/healthz``, ``/readyz`` and Prometheus
``/metrics``. ``/readyz`` returns 500 until the workers have finished
``__init__`` -- which is exactly "the champion is loaded" -- so it is what the
container healthcheck probes.

``__init__`` runs once per worker **process**, never per request.
"""

from __future__ import annotations

import dataclasses

import bentoml

from conquer3.config.settings import get_settings
from conquer3.serving.api_models import (
    LegacyParams,
    LegacyResponse,
    ModelInfoResponse,
    ScoreResult,
    TransactionIn,
    to_transaction_events,
)
from conquer3.serving.champion import load_active_champion
from conquer3.serving.event_sink import JsonlEventSink
from conquer3.serving.scorer import FraudScorer
from conquer3.serving.state_store import RedisStateStore
from conquer3.telemetry.otel import init_telemetry

__all__ = ["FraudScorerService"]

# Read at import time, in the worker process, so `bentoml serve` picks up the
# same C3_* environment the supervisor was configured with. settings.py stays the
# only place environment variables are read.
_settings = get_settings()

_KNOWN_OPS = frozenset({"score", "model_info"})


@bentoml.service(
    name="conquer3-scorer",
    workers=_settings.serving.scorer_workers,
    traffic={"timeout": _settings.serving.scorer_timeout_s},
)
class FraudScorerService:
    """Real-time credit-fraud scoring for PaySim transactions.

    Send raw transactions, not features: feature computation belongs to the
    server (`conquer3.core.features`) so that live serving, batch backfill, and
    Colab training stay bit-for-bit identical. Per-account state is folded in
    `event_ts_us` order, so several transactions for one account in a single
    request see each other's state.
    """

    def __init__(self) -> None:
        # BentoML workers are separate OS processes from the supervisor that
        # already called init_telemetry() in cli.py's _cmd_serve -- this is the
        # one that actually runs in the process serving requests. Without it,
        # every span and metric below disappears silently.
        init_telemetry("conquer3-scorer")
        pipe, ref = load_active_champion(_settings)
        self._scorer = FraudScorer(
            pipe=pipe,
            ref=ref,
            threshold=_settings.serving.decision_threshold,
            state=RedisStateStore(redis_settings=_settings.redis, state_settings=_settings.state),
            sink=JsonlEventSink(
                event_settings=_settings.event,
                worker_id=bentoml.server_context.worker_index or 0,
            ),
        )

    def _predict(self, transactions: list[TransactionIn], *, dry_run: bool) -> list[ScoreResult]:
        return self._scorer.score(to_transaction_events(transactions), dry_run=dry_run)

    def _model_info(self) -> ModelInfoResponse:
        return ModelInfoResponse(**dataclasses.asdict(self._scorer.ref))

    @bentoml.api(route="/predict")
    def predict(
        self, transactions: list[TransactionIn], dry_run: bool = False
    ) -> list[ScoreResult]:
        """Score one or more transactions.

        Set `dry_run` to score without writing: Redis state and the scored-event
        log are left untouched, but current state is still *read*, so the score
        matches what a live request would have returned. That is the skew audit's
        replay path -- re-scoring for audit purposes can never corrupt live state.
        """
        return self._predict(transactions, dry_run=dry_run)

    @bentoml.api(route="/model_info")
    def model_info(self) -> ModelInfoResponse:
        """The champion model version this worker currently has loaded.

        The supervisor polls this to confirm a promotion actually took effect
        before recording it in the deployment audit trail.
        """
        return self._model_info()

    @bentoml.api(route="/invocations")
    def invocations(
        self, dataframe_records: list[TransactionIn], params: LegacyParams | None = None
    ) -> LegacyResponse:
        """DEPRECATED -- use `/predict` and `/model_info` instead.

        MLflow scoring-server compatibility envelope, kept so callers written
        against the previous implementation keep working unchanged. `params.op`
        selects between scoring and metadata; both are dedicated routes now.
        """
        params = params or LegacyParams()
        if params.op not in _KNOWN_OPS:
            raise ValueError(f"unknown op {params.op!r}; expected one of {sorted(_KNOWN_OPS)}")
        if params.op == "model_info":
            return LegacyResponse(predictions=[self._model_info().model_dump()])
        results = self._predict(dataframe_records, dry_run=params.dry_run)
        return LegacyResponse(predictions=[r.model_dump() for r in results])
