"""Boot-time (and every champion-poll-time) wrapper build.

``resolve_champion()`` (Layer 4, untouched -- bounded timeouts, degraded-mode
cache) is the only call that ever talks to remote MLflow here. Everything after
that happens on ``scorer``'s own disk: build a local pyfunc wrapper around the
downloaded artifact, then atomically repoint the live serving symlink at it. No
``models:/…`` URI exists anywhere past this function -- there is nothing left for
MLflow to be re-resolved *through* once it returns (plan §8.1).

The wrapper is never logged back to the registry: doing so would make the
registered artifact depend on Redis/event-sink config and would invalidate Layer
4's already-passing gate. Keeping the registry model pure is what lets one
artifact serve Colab, batch, the skew audit, and production.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import tempfile
from pathlib import Path

import mlflow.pyfunc

from conquer3.config.settings import Settings, get_settings
from conquer3.contracts.model_registry import ModelRef, cached_model_dir, resolve_champion
from conquer3.serving.pyfunc_model import FraudScorerModel
from conquer3.serving.signature import build_signature

__all__ = ["build_and_activate_champion"]


def build_and_activate_champion(settings: Settings | None = None) -> ModelRef:
    """Resolves the champion, builds (or reuses) its wrapper directory, then
    atomically repoints the live symlink at it. Idempotent: calling this again
    with an unchanged champion is cheap (the wrapper directory for that version
    already exists) and always re-activates the same symlink target, so the
    supervisor can call it unconditionally on every poll tick without tracking
    "did the version actually change" itself.
    """
    settings = settings or get_settings()
    _model, ref = resolve_champion(settings.model.name, settings=settings)
    wrapper_dir = _build_wrapper_dir(ref, settings)
    _activate(wrapper_dir, settings)
    return ref


def _build_wrapper_dir(ref: ModelRef, settings: Settings) -> Path:
    wrapped_root = Path(settings.serving.wrapped_model_dir)
    dest = wrapped_root / ref.version
    if dest.is_dir():
        # Already built for this exact version -- skip the expensive
        # save_model(), but `ref.degraded` is a property of *this resolution*,
        # not of the version, and can differ from what an earlier build for the
        # same version baked in (e.g. a live resolve reusing a wrapper an
        # earlier degraded-cache fallback built, or vice versa). `ref.json`
        # always lands at exactly this path -- a plain-file artifact copies to
        # `artifacts/<basename>`, confirmed against the installed mlflow by
        # inspecting a built wrapper's MLmodel file -- so it's safe to overwrite
        # in place without re-running save_model.
        _write_ref_artifact(ref, dest / "artifacts")
        return dest

    wrapped_root.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f"{ref.version}.build-", dir=wrapped_root))
    try:
        artifacts = {
            "champion": str(cached_model_dir(settings.model, ref.name, ref.version)),
            "ref": str(_write_ref_artifact(ref, scratch)),
        }
        build_path = scratch / "model"
        mlflow.pyfunc.save_model(
            path=str(build_path),
            python_model=FraudScorerModel(),
            artifacts=artifacts,
            signature=build_signature(),
            pip_requirements=["mlflow", "pandas", "redis", "scikit-learn"],
        )
        os.replace(str(build_path), str(dest))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return dest


def _write_ref_artifact(ref: ModelRef, dest_dir: Path) -> Path:
    path = dest_dir / "ref.json"
    path.write_text(json.dumps(dataclasses.asdict(ref)), encoding="utf-8")
    return path


def _activate(wrapper_dir: Path, settings: Settings) -> None:
    """Atomic symlink swap: build a new link under a scratch name, then rename it
    over the live one. `os.replace` on a symlink replaces the link itself (never
    its target), so a reader can never observe a symlink that points nowhere."""
    symlink = Path(settings.serving.current_model_symlink)
    symlink.parent.mkdir(parents=True, exist_ok=True)

    tmp_link = symlink.with_name(f"{symlink.name}.tmp-{os.getpid()}")
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    tmp_link.symlink_to(wrapper_dir, target_is_directory=True)
    os.replace(str(tmp_link), str(symlink))
