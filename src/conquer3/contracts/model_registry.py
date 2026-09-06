"""The model contract: publish (Colab, Layer 8) and resolve (BentoML, Layer 5).

MLflow is the single system of record -- tracking, artifacts, and registry. Tags on
each registered version: feature_schema_version, sklearn_version, python_version,
code_sha, decision_threshold. The alias "champion" marks the deployed version.

Aliases are mutable; ModelRef always names the immutable version actually resolved,
so an audit trail (db.ops.record_model_deployment, Layer 5) can record what was
really loaded, not just which alias pointed where at the time.
"""

from __future__ import annotations

import json
import logging
import platform
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from conquer3.config.settings import ModelSettings, Settings, get_settings
from conquer3.core.schema import FEATURE_SCHEMA_VERSION

if TYPE_CHECKING:
    from mlflow.pyfunc import PyFuncModel

# mlflow warns whenever the runtime's installed package versions differ from
# what was pinned into an artifact's requirements.txt at publish time -- purely
# informational (the load still succeeds; the failure modes that actually
# matter are the bounded timeouts _bound_mlflow_calls sets up below) and
# unavoidably noisy every time this repo's mlflow version moves ahead of an
# already-published artifact's pin. Every mlflow.sklearn/mlflow.pyfunc load in
# this file and in serving/service.py's _resolve_sklearn_champion routes
# through here at import time, so setting it once here covers both.
logging.getLogger("mlflow.utils.requirements_utils").setLevel(logging.ERROR)

__all__ = [
    "ChampionResolutionError",
    "IncompatibleModelError",
    "ModelRef",
    "ModelRegistryError",
    "ModelVersionInfo",
    "cached_model_dir",
    "list_model_versions",
    "list_registered_models",
    "load_native_estimator",
    "model_input_columns",
    "model_string_input_columns",
    "promote_champion",
    "publish_model",
    "resolve_champion",
    "resolve_version",
    "should_reload",
    "verify_compatible",
]

# publish_model always logs through mlflow.sklearn, but a version registered
# directly against MLflow by training code this repo doesn't own can carry any
# flavor -- checked in this order, first match wins.
_NATIVE_FLAVORS = ("sklearn", "lightgbm", "xgboost")


class ModelRegistryError(Exception):
    """Base for every error this module raises."""


class IncompatibleModelError(ModelRegistryError):
    """A model version's feature_schema_version tag doesn't match core.schema's."""


class ChampionResolutionError(ModelRegistryError):
    """Neither a live MLflow resolve nor a cached fallback could produce a model."""


@dataclass(frozen=True, slots=True)
class ModelRef:
    name: str
    version: str
    run_id: str
    alias: str
    tags: dict[str, str] = field(default_factory=dict)
    degraded: bool = False


@dataclass(frozen=True, slots=True)
class ModelVersionInfo:
    """One registered version, for the UI's model-listing table (Layer 9). A read
    projection over MLflow's own ``ModelVersion`` -- never a second source of truth
    for what "champion" means; ``aliases`` simply echoes what the registry reports.
    """

    version: str
    run_id: str
    created_at_ms: int
    tags: dict[str, str]
    aliases: tuple[str, ...]
    compatible: bool


def verify_compatible(tags: dict[str, str]) -> None:
    """Raises IncompatibleModelError unless tags['feature_schema_version'] matches
    core.schema.FEATURE_SCHEMA_VERSION exactly. Called on both the live and the
    cached path in resolve_champion -- an incompatible cached champion combined
    with a dead tracking server is unrecoverable and must still raise.
    """
    got = tags.get("feature_schema_version")
    want = str(FEATURE_SCHEMA_VERSION)
    if got != want:
        raise IncompatibleModelError(
            f"model's feature_schema_version tag is {got!r}, core.schema says {want!r}"
        )


def model_input_columns(model: PyFuncModel) -> tuple[str, ...] | None:
    """The named input columns a resolved pyfunc model declares, in signature
    order -- ``None`` if it has no signature or its inputs aren't named (e.g. a
    raw tensor spec).

    Lets the scorer feed a model whose own feature set only partially overlaps
    conquer3's (a version registered outside ``publish_model``, e.g. a
    different training pipeline) the intersection of what it declares and what
    ``core.features`` actually computes, instead of requiring an exact match.
    """
    schema = model.metadata.get_input_schema()  # type: ignore[no-untyped-call]
    if schema is None or not schema.has_input_names():
        return None
    return tuple(schema.input_names())


def model_string_input_columns(model: PyFuncModel) -> frozenset[str]:
    """The subset of ``model_input_columns(model)`` MLflow declares as a
    ``string``-typed ColSpec -- empty if the model has no signature, or none
    of its columns are string-typed.

    A native LightGBM/XGBoost booster trained with a pandas ``category``-dtype
    column strictly validates the *count* of categorical-dtype columns it's
    handed at predict time; the scorer otherwise builds every column as plain
    ``float64``, which such a booster rejects outright even though the value
    itself (a string, or NaN for a column ``core.features`` never computes
    live) is exactly what the column always held -- confirmed empirically
    against ``paysim-fraud-lightgbm`` v4's real ``type`` column. Casting these
    columns to ``category`` dtype before scoring (``serving/scorer.py``'s
    ``_predict_proba``) is what actually fixes it, flavor-agnostically: this
    reacts only to what MLflow itself declared, never to the loaded
    estimator's concrete Python type.
    """
    schema = model.metadata.get_input_schema()  # type: ignore[no-untyped-call]
    if schema is None or not schema.has_input_names():
        return frozenset()
    from mlflow.types import DataType

    return frozenset(col.name for col in schema.inputs if col.type == DataType.string)


def list_registered_models(*, settings: Settings | None = None) -> list[str]:
    """Every registered model name in the MLflow registry, not just
    ``settings.model.name`` -- lets the serving pool and the UI's registry
    panel span every model family a training pipeline has ever registered,
    not only the one this deployment happens to be pinned/aliased to.
    """
    from mlflow.tracking import MlflowClient

    settings = settings or get_settings()
    _bound_mlflow_calls(settings)

    client = MlflowClient()
    return sorted(rm.name for rm in client.search_registered_models())


def publish_model(
    sk_model: Any,
    x_sample: Any,
    proba_sample: Any,
    *,
    sklearn_version: str,
    code_sha: str,
    decision_threshold: float,
    model_name: str | None = None,
    alias_as_champion: bool = False,
    extra_pip_requirements: list[str] | None = None,
    settings: Settings | None = None,
) -> ModelRef:
    """Logs + registers sk_model, tags the new version, optionally aliases it
    "champion". x_sample/proba_sample feed mlflow.models.infer_signature -- an
    explicit signature is what makes column-order/dtype drift fail loudly at load
    instead of silently mis-scoring.
    """
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient

    settings = settings or get_settings()
    model_name = model_name or settings.model.name
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.set_experiment(settings.mlflow.experiment_name)

    signature = infer_signature(x_sample, proba_sample)
    pip_requirements = [
        f"mlflow=={mlflow.__version__}",
        f"scikit-learn=={sklearn_version}",
        *(extra_pip_requirements or []),
    ]

    with mlflow.start_run() as run:
        info = mlflow.sklearn.log_model(
            sk_model=sk_model,
            name="model",
            signature=signature,
            input_example=x_sample[:5] if hasattr(x_sample, "__getitem__") else x_sample,
            pip_requirements=pip_requirements,
            registered_model_name=model_name,
        )
        run_id = run.info.run_id

    assert info.registered_model_version is not None
    version = str(info.registered_model_version)

    tags = {
        "feature_schema_version": str(FEATURE_SCHEMA_VERSION),
        "sklearn_version": sklearn_version,
        "python_version": platform.python_version(),
        "code_sha": code_sha,
        "decision_threshold": str(decision_threshold),
    }
    client = MlflowClient()
    for key, value in tags.items():
        client.set_model_version_tag(model_name, version, key, value)

    alias = ""
    if alias_as_champion:
        client.set_registered_model_alias(model_name, settings.model.alias, version)
        alias = settings.model.alias

    return ModelRef(name=model_name, version=version, run_id=run_id, alias=alias, tags=tags)


def resolve_champion(
    model_name: str | None = None, *, settings: Settings | None = None
) -> tuple[PyFuncModel, ModelRef]:
    """Resolves the "champion" alias to a concrete version, loads it, and caches
    both the ref and the artifact. If MLflow is unreachable, falls back to the
    last successfully cached ref + artifact, marks the result degraded, and emits
    the c3_model_resolution_degraded gauge -- MLflow is never on the request path,
    only boot/explicit reload, so an outage here must not take scoring down.
    """
    settings = settings or get_settings()
    model_name = model_name or settings.model.name
    alias = settings.model.alias

    try:
        ref, model = _resolve_live(model_name, alias, settings)
    except IncompatibleModelError:
        # A live-but-incompatible model is a deliberate rejection, not a
        # connectivity failure -- "fail at load, not at inference" applies to a
        # bad live model too, so this must never be mistaken for an outage and
        # fall through to a possibly-stale cache.
        raise
    except Exception:
        cached = _read_cache(settings.model)
        if cached is None:
            raise ChampionResolutionError(
                f"live resolve of {model_name!r}@{alias} failed and no cached "
                "champion exists to fall back to"
            ) from None
        verify_compatible(cached.tags)
        import mlflow.pyfunc

        local_dir = cached_model_dir(settings.model, cached.name, cached.version)
        model = mlflow.pyfunc.load_model(str(local_dir))
        degraded_ref = ModelRef(**{**asdict(cached), "degraded": True})
        _emit_degraded_gauge(True)
        return model, degraded_ref
    else:
        _write_cache(settings.model, ref)
        _emit_degraded_gauge(False)
        return model, ref


def should_reload(current: ModelRef, candidate: ModelRef) -> bool:
    """Whether a newly resolved ``candidate`` should replace the currently-served
    ``current`` ref -- the decision a worker-side hot-reload loop makes every poll
    tick.

    Never regresses an already-healthy ref to a stale degraded fallback: if MLflow
    is unreachable on a later poll, ``resolve_champion`` still returns *something*
    (the last cached ref, marked degraded) rather than raising, and blindly
    reloading that would downgrade a worker that already has a good model loaded.
    Does reload on a genuine version change, and also on recovery -- same version,
    but the candidate is no longer degraded while the current one was.
    """
    if candidate.degraded and not current.degraded:
        return False
    return candidate.version != current.version or candidate.degraded != current.degraded


def _bound_mlflow_calls(settings: Settings) -> None:
    """Sets the env vars mlflow itself reads to bound every way a sick tracking
    server can block a caller. Deliberate exception to "config/settings.py is the
    only place env vars are read" -- these are mlflow's own knobs, not ours.

    Two distinct failure modes, two distinct guards:

    MLFLOW_HTTP_REQUEST_TIMEOUT / _MAX_RETRIES bound a dead tracking server --
    confirmed empirically that mlflow's defaults (120s timeout + 7 retries with
    exponential backoff) can block for minutes even though each individual
    connection attempt is refused instantly.

    MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false forces artifacts to stream
    through the tracking server, instead of the presigned-URL path mlflow 3.x
    auto-enables whenever the server advertises it in /server-info. That path
    hands the client a signed object-store URL and pulls chunks straight from it,
    bypassing the tracking server entirely -- silently adding a second network
    dependency the deployment contract never promised. Confirmed empirically: a
    remote registry returned presigned URLs on host `storage:9000` (its own
    compose-internal MinIO), so every chunk failed DNS resolution while the
    server's access log showed nothing but 200s for the presigned-URL requests
    themselves. The tracking URI is the only endpoint a client is guaranteed to
    reach, and these artifacts are a few MB, so proxying them costs nothing worth
    that failure mode.

    MLFLOW_DOWNLOAD_CHUNK_TIMEOUT covers the remaining chunked path, but note what
    it does *not* cover: mlflow reads it only in CloudArtifactRepository (a direct
    `s3://`-style artifact root), never in the proxied `mlflow-artifacts:`
    repository this deployment uses, whose chunk timeout is hardcoded to 10s. It
    is set for the `s3://` case only -- it is not, and never was, a bound on the
    presigned path disabled above.

    Called by every entry point that talks to MLflow (resolve, list, promote) so
    none of them can drift from this bound by a copy-paste omission.
    """
    import os

    import mlflow

    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = str(settings.model.resolve_timeout_s)
    os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = str(settings.model.resolve_max_retries)
    os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD"] = "false"
    os.environ["MLFLOW_DOWNLOAD_CHUNK_TIMEOUT"] = str(settings.model.resolve_timeout_s)
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)


def _download_and_load(
    *,
    model_name: str,
    version: str,
    model_uri: str,
    alias: str,
    tags: dict[str, str],
    run_id: str,
    settings: Settings,
) -> tuple[PyFuncModel, ModelRef]:
    """Shared by the alias-based and version-based resolve paths: download
    ``model_uri`` into the local artifact cache and build the matching
    :class:`ModelRef`.

    Downloads into a private, per-attempt temp directory and publishes it into
    the real (shared, deterministic) cache path with a single atomic rename,
    never writing into that shared path directly. Two BentoML worker processes
    resolving the same ``(model_name, version)`` concurrently -- the normal
    case at boot, with every worker pre-loading the same pool -- would
    otherwise both write file-by-file into the identical directory, letting
    one observe the other's partial write (confirmed empirically:
    ``zipfile.BadZipFile`` and an empty, unparseable ``MLmodel`` file from the
    same real boot). ``Path.rename`` is atomic on POSIX when the destination
    doesn't exist yet, so a reader can only ever see "not there" or
    "complete", never partial. If the destination already exists (a
    concurrent caller published first), the rename raises ``OSError`` --
    harmless here, since the two downloads are of the exact same immutable
    model version: discard this attempt's copy and use the one already
    published.
    """
    import mlflow.pyfunc

    local_dir = cached_model_dir(settings.model, model_name, version)
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    mlflow.pyfunc.get_model_dependencies(model_uri)

    tmp_dir = local_dir.parent / f".{local_dir.name}.tmp-{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True)
    try:
        model = mlflow.pyfunc.load_model(model_uri, dst_path=str(tmp_dir))
        try:
            tmp_dir.rename(local_dir)
        except OSError:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    ref = ModelRef(name=model_name, version=version, run_id=run_id, alias=alias, tags=tags)
    return model, ref


def _resolve_live(model_name: str, alias: str, settings: Settings) -> tuple[ModelRef, PyFuncModel]:
    from mlflow.tracking import MlflowClient

    _bound_mlflow_calls(settings)

    client = MlflowClient()
    mv = client.get_model_version_by_alias(model_name, alias)
    verify_compatible(mv.tags)  # before downloading the artifact, not after

    assert mv.run_id is not None
    model, ref = _download_and_load(
        model_name=model_name,
        version=mv.version,
        model_uri=f"models:/{model_name}@{alias}",
        alias=alias,
        tags=dict(mv.tags),
        run_id=mv.run_id,
        settings=settings,
    )
    return ref, model


def resolve_version(
    model_name: str, version: str, *, settings: Settings | None = None
) -> tuple[PyFuncModel, ModelRef]:
    """Resolves one specific registered version by number -- not the "champion"
    alias. Used by the pre-load-all-versions boot step and the manual
    model-swap endpoint (``serving/service.py``); ``resolve_champion``'s
    alias-based resolution and degraded-cache fallback are unrelated and
    unaffected by this.

    No degraded-cache fallback here: a caller pre-loading many versions treats
    one that fails to resolve as "skip it", not "serve something stale
    instead" -- that recovery story only makes sense for the single champion
    the service must always have *something* to serve.
    """
    from mlflow.tracking import MlflowClient

    settings = settings or get_settings()
    _bound_mlflow_calls(settings)

    client = MlflowClient()
    mv = client.get_model_version(model_name, version)

    # verify_compatible(mv.tags)

    assert mv.run_id is not None
    return _download_and_load(
        model_name=model_name,
        version=version,
        model_uri=f"models:/{model_name}/{version}",
        alias="",
        tags=dict(mv.tags),
        run_id=mv.run_id,
        settings=settings,
    )


def list_model_versions(
    model_name: str | None = None, *, settings: Settings | None = None
) -> list[ModelVersionInfo]:
    """Every registered version of ``model_name``, newest first -- the UI's model
    table (Layer 9). Read-only: never downloads an artifact, only registry
    metadata, so a slow/broken artifact store can't make this hang (same
    reasoning as ``test_log_current_champion_from_registry``).
    """
    from mlflow.tracking import MlflowClient

    settings = settings or get_settings()
    model_name = model_name or settings.model.name
    _bound_mlflow_calls(settings)

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")

    aliases_by_version: dict[str, list[str]] = {}
    for alias, version in client.get_registered_model(model_name).aliases.items():
        aliases_by_version.setdefault(version, []).append(alias)

    infos = [
        ModelVersionInfo(
            version=mv.version,
            run_id=mv.run_id or "",
            created_at_ms=mv.creation_timestamp,
            tags=dict(mv.tags),
            aliases=tuple(sorted(aliases_by_version.get(mv.version, []))),
            compatible=mv.tags.get("feature_schema_version") == str(FEATURE_SCHEMA_VERSION),
        )
        for mv in versions
    ]
    return sorted(infos, key=lambda info: int(info.version), reverse=True)


def promote_champion(
    version: str, *, model_name: str | None = None, settings: Settings | None = None
) -> ModelRef:
    """Re-aliases ``version`` as the champion. Refuses an incompatible
    ``feature_schema_version`` *before* re-aliasing -- the scorer's own poll would
    otherwise pick up an incompatible model at the next tick with no chance to
    reject it here first.

    This only moves the alias; it does not restart or notify the scorer. The
    scorer's own poll thread (``serving/supervisor.py``, ``C3_CHAMPION_POLL_S``)
    is what actually picks up the change and restarts -- by design, MLflow stays
    off the request path (see ``resolve_champion``'s docstring).
    """
    from mlflow.tracking import MlflowClient

    settings = settings or get_settings()
    model_name = model_name or settings.model.name
    _bound_mlflow_calls(settings)

    client = MlflowClient()
    mv = client.get_model_version(model_name, version)
    verify_compatible(mv.tags)

    alias = settings.model.alias
    client.set_registered_model_alias(model_name, alias, version)
    assert mv.run_id is not None
    return ModelRef(
        name=model_name, version=version, run_id=mv.run_id, alias=alias, tags=dict(mv.tags)
    )


def cached_model_dir(model_settings: ModelSettings, model_name: str, version: str) -> Path:
    """Where the raw native artifact for one resolved version lives on disk.

    Public because Layer 5's wrapper build (``serving/build.py``) needs the exact
    same path to hand to :func:`load_native_estimator` -- it must never re-derive
    this independently and risk drifting from what ``resolve_champion`` actually
    downloaded into.
    """
    return Path(model_settings.cache_dir) / model_name / version


def load_native_estimator(local_dir: Path) -> Any:
    """Loads the raw native estimator (real ``predict_proba``, not the generic
    pyfunc wrapper) from an already-downloaded artifact directory, dispatching
    to whichever mlflow flavor module actually saved it.

    ``publish_model`` always logs through ``mlflow.sklearn``, so its output
    always has a "sklearn" flavor entry. A version registered directly against
    MLflow by training code this module doesn't own can carry a different one
    -- ``paysim-fraud-lightgbm`` is a real example, logged via
    ``mlflow.lightgbm.log_model``. Its ``MLmodel`` has no "sklearn" flavor
    entry at all, so unconditionally calling ``mlflow.sklearn.load_model``
    fails with ``MlflowException: Model does not have the "sklearn" flavor``
    even though the artifact loads fine and behaves like any other sklearn
    estimator (``LGBMClassifier`` has a real ``predict_proba``).
    """
    import importlib

    import yaml

    mlmodel_path = local_dir / "MLmodel"
    try:
        mlmodel = yaml.safe_load(mlmodel_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ModelRegistryError(f"{mlmodel_path} is missing or unreadable: {exc}") from exc
    if not isinstance(mlmodel, dict):
        raise ModelRegistryError(
            f"{mlmodel_path} is empty or not a valid MLflow model file -- "
            "this artifact was not uploaded correctly, refusing to load it"
        )
    flavors = mlmodel.get("flavors", {})
    for flavor_name in _NATIVE_FLAVORS:
        if flavor_name in flavors:
            module = importlib.import_module(f"mlflow.{flavor_name}")
            estimator: Any = module.load_model(str(local_dir))
            return estimator
    raise ModelRegistryError(
        f"{local_dir / 'MLmodel'} has none of the supported native flavors "
        f"{_NATIVE_FLAVORS} -- flavors present: {sorted(flavors)}"
    )


def _write_cache(model_settings: ModelSettings, ref: ModelRef) -> None:
    path = Path(model_settings.champion_cache_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(ref)), encoding="utf-8")


def _read_cache(model_settings: ModelSettings) -> ModelRef | None:
    path = Path(model_settings.champion_cache_file)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("degraded", None)
    return ModelRef(**data)


def _emit_degraded_gauge(degraded: bool) -> None:
    from conquer3.telemetry.otel import get_meter

    meter = get_meter(__name__)
    gauge = meter.create_gauge(
        "c3_model_resolution_degraded",
        description="1 if serving loaded a cached champion instead of resolving live",
    )
    gauge.set(1 if degraded else 0)
