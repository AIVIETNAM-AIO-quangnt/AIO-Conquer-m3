"""Champion activation (supervisor side) and loading (worker side).

``resolve_champion()`` (Layer 4, untouched -- bounded timeouts, degraded-mode
cache) is the only call that ever talks to remote MLflow, and only the supervisor
makes it. It downloads the artifact into ``C3_MODEL_CACHE_DIR`` and this module
then records *which exact version is live* in a small JSON pointer file.

Workers call :func:`load_active_champion`, which reads that pointer and loads the
already-downloaded artifact off local disk. **A worker never contacts MLflow.**
That split does two things the previous symlink-and-pyfunc-wrapper design could
not: it removes the boot race where a restarting worker could independently
resolve a *different* champion than the supervisor just recorded, and it makes
the "scorer is the inference endpoint, not a proxy" claim structural rather than
merely observed -- there is no ``models:/…`` URI anywhere in a worker process.

No pyfunc wrapper is built. MLflow's scoring server needed one because it could
only serve a model URI; BentoML serves a Python class, so the raw sklearn
artifact Layer 4 already caches is loaded directly. Keeping the registry model
pure is what lets one artifact serve Colab, batch, the skew audit, and production.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any

from conquer3.config.settings import Settings, get_settings
from conquer3.contracts.model_registry import ModelRef, cached_model_dir, resolve_champion

__all__ = ["activate_champion", "load_active_champion", "read_active_ref"]


def activate_champion(settings: Settings | None = None) -> ModelRef:
    """Resolve the champion and record it as the active version.

    Idempotent: calling this again with an unchanged champion rewrites the same
    pointer, so the supervisor can call it unconditionally on every poll tick
    without tracking "did the version actually change" itself.
    """
    settings = settings or get_settings()
    _model, ref = resolve_champion(settings.model.name, settings=settings)
    _write_active_ref(ref, settings)
    return ref


def read_active_ref(settings: Settings | None = None) -> ModelRef:
    settings = settings or get_settings()
    path = Path(settings.serving.active_champion_file)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RuntimeError(
            f"no active champion at {path} -- a worker was started without the "
            "supervisor (`conquer3 serve`) having resolved a champion first."
        ) from None
    return ModelRef(**payload)


def load_active_champion(settings: Settings | None = None) -> tuple[Any, ModelRef]:
    """Load the active champion's raw sklearn estimator from the local cache.

    ``mlflow.sklearn``, not ``mlflow.pyfunc``: the scorer needs a real
    ``predict_proba``, and a pyfunc wrapper's ``.predict()`` returns class labels.
    """
    import mlflow.sklearn

    settings = settings or get_settings()
    ref = read_active_ref(settings)
    local_dir = cached_model_dir(settings.model, ref.name, ref.version)
    return mlflow.sklearn.load_model(str(local_dir)), ref


def _write_active_ref(ref: ModelRef, settings: Settings) -> None:
    """Atomic pointer write: a worker booting concurrently reads either the whole
    old ref or the whole new one, never a half-written file. `ref.degraded` is a
    property of *this resolution*, not of the version, so this always overwrites
    rather than skipping when the version is unchanged."""
    path = Path(settings.serving.active_champion_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(dataclasses.asdict(ref)), encoding="utf-8")
    os.replace(str(tmp), str(path))
