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
import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import bentoml
import bentoml.exceptions
import pandas as pd

from conquer3.config.settings import Settings, get_settings
from conquer3.contracts.model_registry import (
    ModelRef,
    ModelRegistryError,
    cached_model_dir,
    list_model_versions,
    list_registered_models,
    load_native_estimator,
    model_input_columns,
    model_string_input_columns,
    resolve_champion,
    resolve_version,
    should_reload,
)
from conquer3.serving.api_models import (
    LegacyParams,
    LegacyResponse,
    LoadedModelInfo,
    ModelInfoResponse,
    ScoreResult,
    TransactionIn,
    to_transaction_events,
)
from conquer3.serving.event_sink import JsonlEventSink
from conquer3.serving.scorer import GOLD_SCHEMA_COLUMNS, Champion, FraudScorer
from conquer3.serving.state_store import RedisStateStore
from conquer3.telemetry.otel import init_telemetry

if TYPE_CHECKING:
    from mlflow.pyfunc import PyFuncModel

__all__ = ["FraudScorerService"]

_logger = logging.getLogger(__name__)

# Read at import time, in the worker process, so `bentoml serve` picks up the
# same C3_* environment the supervisor was configured with. settings.py stays the
# only place environment variables are read.
_settings = get_settings()

_KNOWN_OPS = frozenset({"score", "model_info"})


def _load_estimator(local_dir: Path, pyfunc_model: PyFuncModel) -> Any:
    """Loads the raw native estimator when one exists, else falls back to the
    already-loaded generic pyfunc wrapper.

    Not every real registration has a sklearn/lightgbm/xgboost flavor --
    ``load_native_estimator``'s dispatch raises ``ModelRegistryError`` for a
    model logged with only a ``python_function`` flavor (a hand-authored
    ``mlflow.pyfunc.PythonModel``, e.g. a real registered
    ``paysim-fraud-xgb-baseline``/``-enhanced``/``-optimal``, each carrying a
    ``probability_type: raw|calibrated`` tag documenting their own
    ``.predict()`` output). Falling back to ``pyfunc_model`` -- already paid
    for by the caller's own resolve -- rather than re-raising is what makes
    such a model loadable and swappable at all; ``FraudScorer._predict_proba``
    dispatches on ``hasattr(pipe, "predict_proba")``, so a bare
    ``PyFuncModel`` (which never has one) automatically takes its own
    ``.predict()`` path there.
    """
    try:
        return load_native_estimator(local_dir)
    except ModelRegistryError:
        return pyfunc_model


def _require_feedable(input_columns: tuple[str, ...] | None, ref: ModelRef) -> None:
    """Raises ``ModelRegistryError`` for a model this scorer can never build a
    well-defined input row for -- one with no declared MLflow signature at
    all, or one whose declared columns share zero overlap with
    ``GOLD_SCHEMA_COLUMNS``.

    Both are the same failure in substance: a model registered with no usable
    contract for what its input row should look like. Left unchecked, either
    one still resolves and loads fine, only to raise on *every single*
    ``/predict`` call once switched to (confirmed against a real registration,
    ``paysim-fraud-lightgbm`` v1: no signature, sent the full live feature
    set, guaranteed mismatch against what it actually trained on). Called
    from both ``_resolve_sklearn_champion`` and ``_preload_all_versions`` so
    such a model is excluded up front -- "fail at load, not at inference" --
    instead of silently trapping a future ``/switch_model`` caller.
    """
    if input_columns is None:
        raise ModelRegistryError(
            f"{ref.name!r} v{ref.version} has no declared MLflow signature -- this scorer "
            "cannot know what row it actually expects, refusing to serve it"
        )
    if not set(input_columns) & GOLD_SCHEMA_COLUMNS:
        raise ModelRegistryError(
            f"{ref.name!r} v{ref.version} declares input columns {input_columns!r}, none of "
            "which gold.txn_features has room for -- not a gold-schema model, refusing to serve it"
        )


def _smoke_score(
    pipe: Any,
    pyfunc_model: PyFuncModel,
    local_dir: Path,
    categorical_columns: frozenset[str],
    ref: ModelRef,
) -> None:
    """Scores the version's own MLflow-logged input example once, at pre-load
    time, so a version that loads fine but whose actual ``predict``/
    ``predict_proba`` call breaks -- a training/serving skew ``_require_feedable``
    can't see, since it only checks the declared schema, never calls the
    estimator -- is excluded now, not discovered on the first real ``/predict``
    after a swap (the exact shape of the original ``paysim-fraud-lightgbm`` v4
    categorical-dtype bug, now fixed for that one case but not guarded against
    generically).

    Uses the version's own real logged example, not a synthetic all-NaN/zero
    row: several registered models are plain sklearn
    ``DecisionTreeClassifier``/``RandomForestClassifier``, which raise on NaN
    input, so a synthetic row would incorrectly reject models already known to
    work. Skipped, not failed, when the version never logged an example (not
    every registration captures one -- its absence doesn't mean the model
    itself is broken) or logged one in a shape other than a DataFrame.
    """
    try:
        example = pyfunc_model.metadata.load_input_example(str(local_dir))
    except Exception as exc:
        raise ModelRegistryError(
            f"{ref.name!r} v{ref.version} declares an input example that could not be read: {exc}"
        ) from exc
    if not isinstance(example, pd.DataFrame):
        return
    row = example.copy()
    for col in categorical_columns & set(row.columns):
        row[col] = row[col].astype("category")
    try:
        pipe.predict_proba(row) if hasattr(pipe, "predict_proba") else pipe.predict(row)
    except Exception as exc:
        raise ModelRegistryError(
            f"{ref.name!r} v{ref.version} failed to score its own logged input example "
            f"({type(exc).__name__}: {exc}) -- refusing to serve it"
        ) from exc


def _resolve_sklearn_champion(
    settings: Settings,
) -> tuple[Any, ModelRef, tuple[str, ...] | None, frozenset[str]]:
    """Resolves the model to serve and returns the *raw native estimator* (or
    a pyfunc fallback -- see ``_load_estimator``) ``FraudScorer`` needs, not
    the generic pyfunc wrapper either resolve path below returns on its own --
    plus that pyfunc wrapper's declared input columns (``model_input_columns``)
    and which of those are string-typed (``model_string_input_columns``), read
    from it before it's discarded, so the scorer can feed a
    partially-overlapping model just the intersection, correctly typed.

    If ``settings.model.version`` is set, pins to that exact registered
    version via ``resolve_version`` -- for a model that isn't (or isn't yet)
    aliased -- instead of resolving the "champion" alias. This has no
    degraded-cache fallback (see ``resolve_version``'s docstring): a transient
    MLflow outage while pinned is handled the same way it always was, by the
    caller's own retry-next-tick logic (``_champion_reload_loop``), not by a
    stale-cache substitute.

    Otherwise (the default, empty pin) live-resolves the champion alias
    (bounded timeouts, degraded-cache fallback -- see ``resolve_champion``'s
    docstring). Either way loads via ``mlflow.pyfunc.load_model`` -- a
    deliberately generic, tested contract (Layer 4's gate; also used by the
    skew audit and the CLI) that must not change. But as a side effect it
    downloads the full artifact into ``cached_model_dir(...)``, so re-loading
    that same local directory via :func:`_load_estimator` (no network --
    exactly what ``champion.load_active_champion`` already does for the
    supervisor-cached case) gives the real ``predict_proba`` without touching
    that contract, whatever flavor the artifact actually is.

    Raises via ``_require_feedable`` for a model with no signature, or zero
    gold-schema overlap, and via ``_smoke_score`` for a model whose own logged
    input example it cannot actually score -- the pinned/aliased default must
    be genuinely feedable and scoreable, same bar every other pooled version
    is held to.
    """
    if settings.model.version:
        pyfunc_model, ref = resolve_version(
            settings.model.name, settings.model.version, settings=settings
        )
    else:
        pyfunc_model, ref = resolve_champion(settings.model.name, settings=settings)
    local_dir = cached_model_dir(settings.model, ref.name, ref.version)
    pipe = _load_estimator(local_dir, pyfunc_model)
    input_columns = model_input_columns(pyfunc_model)
    _require_feedable(input_columns, ref)
    categorical_columns = model_string_input_columns(pyfunc_model)
    _smoke_score(pipe, pyfunc_model, local_dir, categorical_columns, ref)
    return pipe, ref, input_columns, categorical_columns


def _parse_skip_versions(raw: str) -> frozenset[tuple[str, str]]:
    """``"name:version,name:version"`` -> ``{(name, version), ...}``. Blank
    entries (a trailing comma, a blank env var) are ignored rather than
    raising -- this is an ops escape hatch, not a validated config file."""
    pairs: set[tuple[str, str]] = set()
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, _, version = entry.partition(":")
        if name and version:
            pairs.add((name, version))
    return frozenset(pairs)


def _preload_all_versions(
    settings: Settings, model_names: list[str]
) -> dict[tuple[str, str], Champion]:
    """Downloads and loads every version of every registered model name --
    not just ``settings.model.name`` -- so ``POST /switch_model`` can swap to
    *any* real model on the registry, not only versions of one configured
    champion, and never needs a network call to do it.

    A version that fails to resolve (network blip, incompatible feature
    schema, no usable signature -- see ``_require_feedable``, no loadable
    estimator at all, or one that loads but can't actually score its own
    logged input example -- see ``_smoke_score``) is logged and excluded,
    never fatal on its own; only an **empty** resulting pool -- nothing
    loadable across the *entire* registry -- is fatal.

    ``settings.serving.scorer_skip_versions`` (``C3_SCORER_SKIP_VERSIONS``)
    is checked *before* attempting to load each version -- confirmed against
    the real registry that a version's artifact can crash the process
    outright (a native segfault while unpickling, not a raised Python
    exception), which the ``except Exception`` below can never catch since
    the process doing the catching is what dies.

    Pool keyed by ``(name, version)``, not version alone: two different
    registered models can legitimately share a version number (confirmed on
    the real registry: ``paysim-fraud-lightgbm`` v3 and ``paysim_fraud_clf``
    v3 both exist).
    """
    skip = _parse_skip_versions(settings.serving.scorer_skip_versions)
    pool: dict[tuple[str, str], Champion] = {}
    for name in model_names:
        for info in list_model_versions(name, settings=settings):
            if (name, info.version) in skip:
                _logger.warning(
                    "pre-load: %s v%s in C3_SCORER_SKIP_VERSIONS, skipping without loading",
                    name,
                    info.version,
                )
                continue
            try:
                pyfunc_model, ref = resolve_version(name, info.version, settings=settings)
                local_dir = cached_model_dir(settings.model, ref.name, ref.version)
                pipe = _load_estimator(local_dir, pyfunc_model)
                input_columns = model_input_columns(pyfunc_model)
                _require_feedable(input_columns, ref)
                categorical_columns = model_string_input_columns(pyfunc_model)
                _smoke_score(pipe, pyfunc_model, local_dir, categorical_columns, ref)
            except Exception:
                _logger.warning(
                    "pre-load: %s v%s failed to resolve, skipping",
                    name,
                    info.version,
                    exc_info=True,
                )
                continue
            pool[(ref.name, ref.version)] = Champion(
                pipe=pipe,
                ref=ref,
                input_columns=input_columns,
                categorical_columns=categorical_columns,
            )

    if not pool:
        raise RuntimeError(
            f"no version of any registered model ({model_names}) could be resolved -- "
            "refusing to boot with nothing loadable."
        )
    return pool


def _pick_default_version(settings: Settings, pool: dict[tuple[str, str], Champion]) -> Champion:
    """Prefers ``settings.model.version`` when pinned and present in the pool;
    else whichever pooled version of ``settings.model.name`` currently holds
    the "champion" alias; else the highest version number of
    ``settings.model.name`` in the pool.

    Scoped to ``settings.model.name`` throughout, never to the pool at large:
    the pool now spans every registered model, and "highest version number"
    is meaningless across different model families (comparing one model's v6
    to an unrelated model's v1 as if they were commensurate). Raises clearly
    if the *configured* default model has nothing loadable at all -- a real,
    actionable boot failure, never silently papered over by serving some
    other, unrelated model instead.
    """
    own_versions = {
        version: champion
        for (name, version), champion in pool.items()
        if name == settings.model.name
    }
    if not own_versions:
        raise RuntimeError(
            f"configured default model {settings.model.name!r} has no loadable version in "
            "the pool -- refusing to pick an unrelated model as the default champion."
        )
    if settings.model.version and settings.model.version in own_versions:
        return own_versions[settings.model.version]
    for info in list_model_versions(settings.model.name, settings=settings):
        if settings.model.alias in info.aliases and info.version in own_versions:
            return own_versions[info.version]
    return own_versions[max(own_versions, key=int)]


def _log_model_swap(trigger: str, old_ref: ModelRef, new_ref: ModelRef) -> None:
    """Single, verbose log line for every model-swap event -- the automatic
    champion-hot-reload poll and a manual ``POST /switch_model`` call both
    route through here, so the two are never logged inconsistently, and every
    field useful for after-the-fact debugging (which worker, which run,
    degraded status, tags) is always present, not just the version numbers.
    """
    worker_index = bentoml.server_context.worker_index or 0
    _logger.info(
        "model swap [%s] worker=%s :: %s@%s v%s (run_id=%s, degraded=%s) -> "
        "%s@%s v%s (run_id=%s, degraded=%s, tags=%s)",
        trigger,
        worker_index,
        old_ref.name,
        old_ref.alias,
        old_ref.version,
        old_ref.run_id,
        old_ref.degraded,
        new_ref.name,
        new_ref.alias,
        new_ref.version,
        new_ref.run_id,
        new_ref.degraded,
        new_ref.tags,
    )


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
        # Pre-load every version of every registered model -- not just
        # settings.model.name's "champion" -- so POST /switch_model can swap
        # to any real model on the registry and never needs a network call.
        # Default active pick still prefers settings.model.name's champion,
        # so behavior is unchanged for anyone who never calls /switch_model.
        model_names = list_registered_models(settings=_settings)
        self._pool = _preload_all_versions(_settings, model_names)
        default = _pick_default_version(_settings, self._pool)
        self._scorer = FraudScorer(
            pipe=default.pipe,
            ref=default.ref,
            threshold=_settings.serving.decision_threshold,
            state=RedisStateStore(redis_settings=_settings.redis, state_settings=_settings.state),
            sink=JsonlEventSink(
                event_settings=_settings.event,
                worker_id=bentoml.server_context.worker_index or 0,
            ),
            input_columns=default.input_columns,
            categorical_columns=default.categorical_columns,
        )
        # Set by POST /switch_model: a manual pick must never be silently
        # reverted by the next automatic poll tick below.
        self._reload_paused = threading.Event()
        threading.Thread(
            target=self._champion_reload_loop,
            daemon=True,
            name="champion-hot-reload",
        ).start()

    def _champion_reload_loop(self) -> None:
        """Hot-swaps the in-memory champion in place, in this worker process, on
        every ``champion_poll_s`` tick -- no restart, no connection-refused window.
        Complements (does not replace) the supervisor's own poll+restart, which
        still fires independently on a version change it observes; this loop is
        what lets a version change actually land with zero downtime, and is what
        gives a standalone-started worker (no supervisor at all) live reload it
        would otherwise never get.

        A resolve failure (MLflow down, network blip) is never allowed to kill
        this thread or the worker -- log and try again next tick, exactly like
        supervisor.py's own champion-poll thread.

        Permanently skipped once ``POST /switch_model`` has been called on this
        worker (``self._reload_paused``) -- a manual pick lasts until the
        worker restarts, never silently reverted by this loop's next tick.
        """
        while True:
            time.sleep(_settings.serving.champion_poll_s)
            if self._reload_paused.is_set():
                continue
            try:
                pipe, ref, input_columns, categorical_columns = _resolve_sklearn_champion(_settings)
            except Exception:
                _logger.warning(
                    "champion hot-reload poll failed; keeping current model", exc_info=True
                )
                continue
            current = self._scorer.ref
            if should_reload(current, ref):
                self._scorer.reload(
                    pipe=pipe,
                    ref=ref,
                    input_columns=input_columns,
                    categorical_columns=categorical_columns,
                )
                _log_model_swap("auto-hot-reload", current, ref)

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

    @bentoml.api(route="/models")
    def models(self) -> list[LoadedModelInfo]:
        """Every version of every registered model this worker pre-loaded at
        startup, and which one is currently active -- what POST /switch_model
        lets you pick between."""
        active = (self._scorer.ref.name, self._scorer.ref.version)
        return [
            LoadedModelInfo(
                **dataclasses.asdict(champion.ref),
                active=(champion.ref.name, champion.ref.version) == active,
            )
            for champion in self._pool.values()
        ]

    @bentoml.api(route="/switch_model")
    def switch_model(self, name: str, version: str) -> ModelInfoResponse:
        """Activates a pre-loaded (name, version) for this worker's future
        /predict calls -- no network call, it's already in memory from
        startup. Not limited to settings.model.name's own versions: the pool
        spans every registered model, so this can switch to an entirely
        different model family too.

        Manual, per-worker only: with C3_SCORER_WORKERS > 1, this affects only
        the worker process that handles this request, not the whole service --
        see POST /models on each worker to check. Also permanently pauses this
        worker's automatic champion-hot-reload polling (see
        _champion_reload_loop) so an unrelated background poll can never
        silently revert the choice; only a worker restart resumes it.
        """
        champion = self._pool.get((name, version))
        if champion is None:
            raise bentoml.exceptions.NotFound(
                f"{name!r} v{version!r} is not in this worker's pre-loaded pool "
                f"({sorted(self._pool)}) -- see POST /models"
            )
        old_ref = self._scorer.ref
        self._scorer.reload(
            pipe=champion.pipe,
            ref=champion.ref,
            input_columns=champion.input_columns,
            categorical_columns=champion.categorical_columns,
        )
        self._reload_paused.set()
        _log_model_swap("manual-switch-model", old_ref, champion.ref)
        return ModelInfoResponse(**dataclasses.asdict(champion.ref))

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
