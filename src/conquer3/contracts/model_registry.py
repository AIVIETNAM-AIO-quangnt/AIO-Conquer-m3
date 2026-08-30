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
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from conquer3.config.settings import ModelSettings, Settings, get_settings
from conquer3.core.schema import FEATURE_SCHEMA_VERSION

if TYPE_CHECKING:
    from mlflow.pyfunc import PyFuncModel

__all__ = [
    "ChampionResolutionError",
    "IncompatibleModelError",
    "ModelRef",
    "ModelRegistryError",
    "cached_model_dir",
    "publish_model",
    "resolve_champion",
    "verify_compatible",
]


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


def _resolve_live(model_name: str, alias: str, settings: Settings) -> tuple[ModelRef, PyFuncModel]:
    import os

    import mlflow
    import mlflow.pyfunc
    from mlflow.tracking import MlflowClient

    # Deliberate exception to "config/settings.py is the only place env vars are
    # read" -- this *writes* env vars mlflow itself reads, to bound how long a dead
    # tracking server can block boot (confirmed empirically: mlflow's defaults of
    # 120s timeout + 7 retries with exponential backoff can take minutes even
    # though each individual connection attempt is refused instantly).
    #
    # The two below address a *different* failure mode from the registry call: a
    # tracking server that answers registry API calls fine but cannot actually
    # deliver artifact bytes.
    #
    # MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false forces artifacts to stream
    # through the tracking server, instead of the presigned-URL path mlflow 3.x
    # auto-enables whenever the server advertises it in /server-info. That path
    # hands the client a signed object-store URL and pulls chunks straight from
    # it, bypassing the tracking server entirely -- silently adding a second
    # network dependency the deployment contract never promised. Confirmed
    # empirically: a remote registry returned presigned URLs on host
    # `storage:9000` (its own compose-internal MinIO), so every chunk failed DNS
    # resolution while the server's access log showed nothing but 200s for the
    # presigned-URL requests themselves. The tracking URI is the only endpoint a
    # client is guaranteed to reach, and these artifacts are a few MB, so
    # proxying them costs nothing worth that failure mode.
    #
    # MLFLOW_DOWNLOAD_CHUNK_TIMEOUT covers the remaining chunked path, but note
    # what it does *not* cover: mlflow reads it only in CloudArtifactRepository
    # (a direct `s3://`-style artifact root), never in the proxied
    # `mlflow-artifacts:` repository this deployment uses, whose chunk timeout is
    # hardcoded to 10s. It is set for the `s3://` case only -- it is not, and
    # never was, a bound on the presigned path disabled above.
    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = str(settings.model.resolve_timeout_s)
    os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = str(settings.model.resolve_max_retries)
    os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD"] = "false"
    os.environ["MLFLOW_DOWNLOAD_CHUNK_TIMEOUT"] = str(settings.model.resolve_timeout_s)
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)

    client = MlflowClient()
    mv = client.get_model_version_by_alias(model_name, alias)
    verify_compatible(mv.tags)  # before downloading the artifact, not after

    assert mv.run_id is not None
    local_dir = cached_model_dir(settings.model, model_name, mv.version)
    local_dir.mkdir(parents=True, exist_ok=True)
    model = mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}", dst_path=str(local_dir))
    ref = ModelRef(
        name=model_name, version=mv.version, run_id=mv.run_id, alias=alias, tags=dict(mv.tags)
    )
    return ref, model


def cached_model_dir(model_settings: ModelSettings, model_name: str, version: str) -> Path:
    """Where the raw sklearn artifact for one resolved version lives on disk.

    Public because Layer 5's wrapper build (``serving/build.py``) needs the exact
    same path to hand to ``mlflow.sklearn.load_model`` -- it must never re-derive
    this independently and risk drifting from what ``resolve_champion`` actually
    downloaded into.
    """
    return Path(model_settings.cache_dir) / model_name / version


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
