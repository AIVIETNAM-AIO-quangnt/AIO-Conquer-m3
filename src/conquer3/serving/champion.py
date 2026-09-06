"""Champion activation (supervisor side) and loading (worker side).

Historically the only call that ever talked to remote MLflow, made only by the
supervisor -- no longer true: ``serving/service.py``'s worker-side hot-reload
(``_resolve_sklearn_champion``) resolves independently too, and both this
module's :func:`activate_champion` and that one now branch on
``settings.model.version``, pinning to an exact registered version instead of
resolving the "champion" alias when it's set. What's still true: this module
records, via ``activate_champion``, *which exact version the supervisor last
resolved* in a small JSON pointer file, downloaded into ``C3_MODEL_CACHE_DIR``.

Workers call :func:`load_active_champion`, which reads that pointer and loads the
already-downloaded artifact off local disk -- though ``service.py``'s own boot
path resolves live itself now rather than calling this, so it's mainly a
building block for anything that specifically wants the supervisor-cached
pointer rather than its own live resolve.

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
from conquer3.contracts.model_registry import (
    ModelRef,
    cached_model_dir,
    load_native_estimator,
    resolve_champion,
    resolve_version,
)

__all__ = ["activate_champion", "load_active_champion", "read_active_ref"]


def activate_champion(settings: Settings | None = None) -> ModelRef:
    """Resolve the model to serve and record it as the active version.

    If ``settings.model.version`` is set, pins to that exact registered
    version via ``resolve_version`` instead of resolving the "champion" alias
    -- for a model that isn't (or isn't yet) aliased. Mirrors
    ``serving/service.py``'s ``_resolve_sklearn_champion`` branching exactly,
    so the supervisor and every worker always agree on which one is active.

    Idempotent: calling this again with an unchanged champion (or an unchanged
    pin) rewrites the same pointer, so the supervisor can call it
    unconditionally on every poll tick without tracking "did the version
    actually change" itself.
    """
    settings = settings or get_settings()
    if settings.model.version:
        _model, ref = resolve_version(
            settings.model.name, settings.model.version, settings=settings
        )
    else:
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
    """Load the active champion's raw native estimator from the local cache.

    Via :func:`load_native_estimator`, not ``mlflow.pyfunc``: the scorer needs
    a real ``predict_proba``, and a pyfunc wrapper's ``.predict()`` returns
    class labels.
    """
    settings = settings or get_settings()
    ref = read_active_ref(settings)
    local_dir = cached_model_dir(settings.model, ref.name, ref.version)
    return load_native_estimator(local_dir), ref


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
