"""Pure-logic tests for contracts.model_registry -- no server, no network."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from conquer3.contracts.model_registry import (
    IncompatibleModelError,
    ModelRef,
    ModelRegistryError,
    load_native_estimator,
    should_reload,
    verify_compatible,
)
from conquer3.core.schema import FEATURE_SCHEMA_VERSION


def _ref(*, version: str, degraded: bool) -> ModelRef:
    return ModelRef(
        name="paysim_fraud_clf", version=version, run_id="r", alias="champion", degraded=degraded
    )


def test_should_reload_on_a_genuine_version_change() -> None:
    assert should_reload(_ref(version="1", degraded=False), _ref(version="2", degraded=False))


def test_should_not_reload_when_nothing_changed() -> None:
    assert not should_reload(_ref(version="1", degraded=False), _ref(version="1", degraded=False))


def test_should_not_regress_a_healthy_ref_to_a_degraded_candidate() -> None:
    assert not should_reload(_ref(version="1", degraded=False), _ref(version="1", degraded=True))
    # Even a different version -- still a stale fallback, not a real promotion.
    assert not should_reload(_ref(version="1", degraded=False), _ref(version="2", degraded=True))


def test_should_reload_on_recovery_from_degraded_at_the_same_version() -> None:
    assert should_reload(_ref(version="1", degraded=True), _ref(version="1", degraded=False))


def test_verify_compatible_accepts_matching_feature_schema_version() -> None:
    verify_compatible({"feature_schema_version": str(FEATURE_SCHEMA_VERSION)})


def test_verify_compatible_rejects_mismatched_feature_schema_version() -> None:
    with pytest.raises(IncompatibleModelError):
        verify_compatible({"feature_schema_version": "999"})


def test_verify_compatible_rejects_missing_tag() -> None:
    with pytest.raises(IncompatibleModelError):
        verify_compatible({})


def test_load_native_estimator_dispatches_to_the_artifacts_actual_flavor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A version registered directly against MLflow (not through
    ``publish_model``) can carry any flavor its own training code used --
    ``paysim-fraud-lightgbm`` is a real example, logged via
    ``mlflow.lightgbm.log_model``. Its ``MLmodel`` has no "sklearn" flavor
    entry at all, so this must not assume sklearn.
    """
    import mlflow.lightgbm

    (tmp_path / "MLmodel").write_text("flavors:\n  lightgbm:\n    lgb_version: 4.6.0\n")
    sentinel = object()
    monkeypatch.setattr(mlflow.lightgbm, "load_model", lambda path: sentinel)

    assert load_native_estimator(tmp_path) is sentinel


def test_load_native_estimator_prefers_sklearn_when_both_flavors_are_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mlflow.lightgbm
    import mlflow.sklearn

    (tmp_path / "MLmodel").write_text("flavors:\n  sklearn: {}\n  lightgbm: {}\n")
    sentinel = object()
    monkeypatch.setattr(mlflow.sklearn, "load_model", lambda path: sentinel)

    def _fail(path: str) -> None:
        raise AssertionError("should not have dispatched to mlflow.lightgbm")

    monkeypatch.setattr(mlflow.lightgbm, "load_model", _fail)

    assert load_native_estimator(tmp_path) is sentinel


def test_load_native_estimator_rejects_an_unsupported_flavor(tmp_path: Path) -> None:
    (tmp_path / "MLmodel").write_text("flavors:\n  pytorch: {}\n")

    with pytest.raises(ModelRegistryError):
        load_native_estimator(tmp_path)


def test_load_native_estimator_rejects_a_missing_mlmodel_file(tmp_path: Path) -> None:
    """A concurrent-download race (or a genuinely incomplete upload) can leave
    the artifact directory without an ``MLmodel`` file at all -- must raise a
    clean, categorized error, not a bare ``FileNotFoundError``, so the caller's
    "logged and excluded" pattern reports it the same way as every other bad
    registration.
    """
    with pytest.raises(ModelRegistryError):
        load_native_estimator(tmp_path)


def test_load_native_estimator_rejects_an_empty_mlmodel_file(tmp_path: Path) -> None:
    """A concurrent-download race can leave a truncated, empty ``MLmodel``
    file behind -- ``yaml.safe_load`` returns ``None`` for it, which must not
    reach the unguarded ``.get("flavors", ...)`` call as a bare
    ``AttributeError``.
    """
    (tmp_path / "MLmodel").write_text("")

    with pytest.raises(ModelRegistryError):
        load_native_estimator(tmp_path)


def test_load_native_estimator_rejects_a_non_mapping_mlmodel_file(tmp_path: Path) -> None:
    (tmp_path / "MLmodel").write_text("- not\n- a\n- mapping\n")

    with pytest.raises(ModelRegistryError):
        load_native_estimator(tmp_path)


def test_download_and_load_publishes_atomically_under_concurrent_callers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two BentoML worker processes resolving the same (name, version)
    concurrently must never let one observe the other's partial write --
    confirmed as a real bug from a live boot with scorer_workers=2
    (zipfile.BadZipFile; a truncated, unparseable empty MLmodel file).

    Simulates the race with two threads, each downloading into its own
    private temp dir (real file I/O releases the GIL, so this exercises real
    OS-level interleaving) before both race to publish into the identical
    shared ``local_dir``. Both calls must return without raising, and the
    published directory must end up a single, complete copy -- never a torn
    mix of the two -- with no leftover temp directories.
    """
    import threading

    import mlflow.pyfunc

    from conquer3.config.settings import ModelSettings, Settings
    from conquer3.contracts import model_registry

    settings = Settings(model=ModelSettings(cache_dir=str(tmp_path)))
    barrier = threading.Barrier(2)

    def fake_get_model_dependencies(model_uri: str) -> None:
        return None

    def fake_load_model(model_uri: str, dst_path: str) -> object:
        barrier.wait()  # maximize the race window right at the publish step
        (Path(dst_path) / "MLmodel").write_text("flavors:\n  sklearn: {}\n")
        (Path(dst_path) / "model.pkl").write_bytes(b"x" * 1000)
        return object()

    monkeypatch.setattr(mlflow.pyfunc, "get_model_dependencies", fake_get_model_dependencies)
    monkeypatch.setattr(mlflow.pyfunc, "load_model", fake_load_model)

    errors: list[BaseException] = []

    def _run() -> None:
        try:
            model_registry._download_and_load(
                model_name="m",
                version="1",
                model_uri="models:/m/1",
                alias="",
                tags={},
                run_id="r",
                settings=settings,
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    local_dir = tmp_path / "m" / "1"
    assert (local_dir / "MLmodel").read_text() == "flavors:\n  sklearn: {}\n"
    assert (local_dir / "model.pkl").stat().st_size == 1000
    # Exactly the published version, no orphaned .tmp-* directories left behind.
    assert sorted(p.name for p in (tmp_path / "m").iterdir()) == ["1"]


def test_model_ref_round_trips_through_json() -> None:
    ref = ModelRef(
        name="paysim_fraud_clf",
        version="3",
        run_id="abc123",
        alias="champion",
        tags={"feature_schema_version": "1"},
        degraded=True,
    )
    restored = ModelRef(**json.loads(json.dumps(asdict(ref))))
    assert restored == ref


def test_log_current_champion_from_registry() -> None:
    """Diagnostic only, no assertions: prints whatever MLFLOW_TRACKING_URI's
    (from the real .env, if configured) "champion"-aliased model currently
    looks like. Run with ``pytest -s`` to actually see the output -- pytest
    captures stdout by default.

    Deliberately queries only registry metadata (MlflowClient), never the
    artifact itself: artifact downloads have their own, separate failure modes
    (see resolve_champion's docstring) that this test does not exercise, so a
    slow/broken artifact store can never make this test hang. Bounded with a
    short, hardcoded timeout so an unreachable tracking server fails (and
    prints why) in seconds rather than blocking the suite.
    """
    from conquer3.config.settings import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    if not settings.mlflow.tracking_uri:
        print("MLFLOW_TRACKING_URI is not set -- nothing to log.")
        return

    import os

    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "10"
    os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "1"

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    model_name = settings.model.name
    alias = settings.model.alias

    print(f"tracking_uri = {settings.mlflow.tracking_uri}")
    print(f"model_name   = {model_name}")
    print(f"alias        = {alias}")

    try:
        mv = MlflowClient().get_model_version_by_alias(model_name, alias)
    except Exception as exc:
        print(f"could not resolve {model_name!r}@{alias!r}: {type(exc).__name__}: {exc}")
        return

    print(f"version      = {mv.version}")
    print(f"run_id       = {mv.run_id}")
    print(f"source       = {mv.source}")
    print(f"status       = {mv.status}")
    print(f"tags         = {mv.tags}")
