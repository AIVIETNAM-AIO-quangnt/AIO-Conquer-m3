"""Pure-logic tests for serving.service's model-pool assembly -- no BentoML
server, no Redis, no network. Covers the pre-load pool spanning *every*
registered model name (not just settings.model.name), the pre-load exclusion
of models this scorer can never feed a well-defined row, and the pyfunc
fallback for a model with no native estimator flavor at all -- none of which
had unit coverage before (tests/integration/test_serving_e2e.py only ever
seeds one, always-signed, single-model champion).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from conquer3.config.settings import ModelSettings, Settings
from conquer3.contracts.model_registry import ModelRef, ModelRegistryError, ModelVersionInfo
from conquer3.serving import service
from conquer3.serving.scorer import Champion


def _settings(*, name: str = "modelA", version: str = "", alias: str = "champion") -> Settings:
    return Settings(model=ModelSettings(name=name, version=version, alias=alias))


def _info(version: str, *, aliases: tuple[str, ...] = ()) -> ModelVersionInfo:
    return ModelVersionInfo(
        version=version,
        run_id=f"r{version}",
        created_at_ms=0,
        tags={},
        aliases=aliases,
        compatible=True,
    )


class _FakeMetadata:
    """Stands in for ``PyFuncModel.metadata`` -- specifically its
    ``load_input_example`` method, which ``_smoke_score`` reads. Defaults to
    "no example logged" (``None``), matching most fakes in this file that
    don't care about the smoke-score gate at all."""

    def __init__(self, input_example: Any = None, *, error: Exception | None = None) -> None:
        self._input_example = input_example
        self._error = error

    def load_input_example(self, path: str) -> Any:
        if self._error is not None:
            raise self._error
        return self._input_example


class _FakePyfunc:
    """Stands in for the generic pyfunc wrapper resolve_version returns --
    carries the owning model's name so the fake model_input_columns below can
    look up per-model behavior; never actually scored in these tests unless
    an ``input_example`` is supplied."""

    def __init__(
        self,
        name: str,
        *,
        input_example: Any = None,
        load_input_example_error: Exception | None = None,
    ) -> None:
        self.name = name
        self.metadata = _FakeMetadata(input_example, error=load_input_example_error)


def _patch_registry(
    monkeypatch: pytest.MonkeyPatch,
    *,
    versions_by_name: dict[str, list[ModelVersionInfo]],
    input_columns_by_name: dict[str, tuple[str, ...] | None],
    pipes_by_name: dict[str, Any] | None = None,
    input_examples_by_name: dict[str, Any] | None = None,
) -> None:
    """Wires service.py's imported registry functions to fakes driven purely
    by the model *name* -- version-specific detail isn't needed for these
    tests, every version of a given fake model behaves the same way. The fake
    pyfunc carries its owning name so model_input_columns can look it up.

    ``pipes_by_name``/``input_examples_by_name`` default to "loads fine, no
    input example logged" (a bare ``object()``, ``metadata.load_input_example``
    returns ``None``) -- the smoke-score gate is a no-op unless a test
    deliberately opts a model into it via these.
    """
    pipes_by_name = pipes_by_name or {}
    input_examples_by_name = input_examples_by_name or {}

    def fake_list_model_versions(name: str, *, settings: Settings) -> list[ModelVersionInfo]:
        return versions_by_name.get(name, [])

    def fake_resolve_version(
        name: str, version: str, *, settings: Settings
    ) -> tuple[Any, ModelRef]:
        pyfunc = _FakePyfunc(name, input_example=input_examples_by_name.get(name))
        ref = ModelRef(name=name, version=version, run_id=f"r{version}", alias="", tags={})
        return pyfunc, ref

    def fake_cached_model_dir(model_settings: ModelSettings, name: str, version: str) -> Path:
        return Path(f"/fake/{name}/{version}")

    def fake_model_input_columns(pyfunc_model: _FakePyfunc) -> tuple[str, ...] | None:
        return input_columns_by_name[pyfunc_model.name]

    def fake_load_native_estimator(local_dir: Path) -> Any:
        return pipes_by_name.get(local_dir.parent.name, object())

    monkeypatch.setattr(service, "list_model_versions", fake_list_model_versions)
    monkeypatch.setattr(service, "resolve_version", fake_resolve_version)
    monkeypatch.setattr(service, "cached_model_dir", fake_cached_model_dir)
    monkeypatch.setattr(service, "load_native_estimator", fake_load_native_estimator)
    monkeypatch.setattr(service, "model_string_input_columns", lambda pyfunc_model: frozenset())
    monkeypatch.setattr(service, "model_input_columns", fake_model_input_columns)


def test_preload_all_versions_pools_across_multiple_model_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two different registered model names, each with one loadable version,
    must both end up in the pool -- keyed by (name, version), not merely
    version, since two different models can share a version number (real
    example: paysim-fraud-lightgbm v3 and paysim_fraud_clf v3 both exist)."""
    _patch_registry(
        monkeypatch,
        versions_by_name={"modelA": [_info("3")], "modelB": [_info("3")]},
        input_columns_by_name={"modelA": ("amount",), "modelB": ("amount",)},
    )

    pool = service._preload_all_versions(_settings(), ["modelA", "modelB"])

    assert set(pool) == {("modelA", "3"), ("modelB", "3")}
    assert pool[("modelA", "3")].ref.name == "modelA"
    assert pool[("modelB", "3")].ref.name == "modelB"


def test_preload_all_versions_excludes_a_model_with_no_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model with no declared MLflow signature at all (model_input_columns
    returns None) is excluded, not pooled -- real example: paysim-fraud-
    lightgbm v1, which 500s on every /predict because this scorer has no way
    to know what row it actually expects."""
    _patch_registry(
        monkeypatch,
        versions_by_name={"good": [_info("1")], "no_signature": [_info("1")]},
        input_columns_by_name={"good": ("amount",), "no_signature": None},
    )

    pool = service._preload_all_versions(_settings(), ["good", "no_signature"])

    assert set(pool) == {("good", "1")}


def test_preload_all_versions_excludes_zero_gold_schema_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model declaring only columns gold.txn_features has no room for at
    all is excluded up front, at pre-load time -- not left in the pool to
    guarantee a 500 the first time someone switches to it."""
    _patch_registry(
        monkeypatch,
        versions_by_name={"foreign": [_info("1")]},
        input_columns_by_name={"foreign": ("totally_unrelated_column",)},
    )

    with pytest.raises(RuntimeError, match="nothing loadable"):
        service._preload_all_versions(_settings(), ["foreign"])


def test_preload_all_versions_raises_when_nothing_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_registry(monkeypatch, versions_by_name={}, input_columns_by_name={})
    with pytest.raises(RuntimeError, match="nothing loadable"):
        service._preload_all_versions(_settings(), ["modelA"])


def test_load_estimator_falls_back_to_pyfunc_when_no_native_flavor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model logged with only a `python_function` flavor (real examples:
    paysim-fraud-xgb-baseline/-enhanced/-optimal) has no sklearn/lightgbm/
    xgboost estimator load_native_estimator can find -- falling back to the
    already-loaded pyfunc wrapper, instead of re-raising, is what makes such
    a model loadable and swappable at all."""

    def fail(local_dir: Path) -> Any:
        raise ModelRegistryError("no supported native flavor")

    monkeypatch.setattr(service, "load_native_estimator", fail)
    pyfunc = _FakePyfunc("x")

    assert service._load_estimator(Path("/fake/x/1"), pyfunc) is pyfunc  # type: ignore[arg-type]


def test_load_estimator_prefers_the_native_estimator_when_one_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = object()
    monkeypatch.setattr(service, "load_native_estimator", lambda local_dir: native)
    pyfunc = _FakePyfunc("x")
    assert service._load_estimator(Path("/fake/x/1"), pyfunc) is native  # type: ignore[arg-type]


def test_require_feedable_rejects_no_signature() -> None:
    ref = ModelRef(name="m", version="1", run_id="r", alias="", tags={})
    with pytest.raises(ModelRegistryError, match="no declared MLflow signature"):
        service._require_feedable(None, ref)


def test_require_feedable_rejects_zero_gold_schema_overlap() -> None:
    ref = ModelRef(name="m", version="1", run_id="r", alias="", tags={})
    with pytest.raises(ModelRegistryError, match="not a gold-schema model"):
        service._require_feedable(("totally_unrelated_column",), ref)


def test_require_feedable_accepts_partial_overlap() -> None:
    ref = ModelRef(name="m", version="1", run_id="r", alias="", tags={})
    service._require_feedable(("amount", "totally_unrelated_column"), ref)  # must not raise


class _FakePipe:
    def __init__(self, *, predict_proba_error: Exception | None = None) -> None:
        self._predict_proba_error = predict_proba_error

    def predict_proba(self, row: pd.DataFrame) -> Any:
        if self._predict_proba_error is not None:
            raise self._predict_proba_error
        return [[0.1, 0.9]]


def test_smoke_score_skips_when_no_input_example_was_logged() -> None:
    """Not every registration captures an input example -- its absence must
    not be treated as a failure, only as "unverifiable this way"."""
    ref = ModelRef(name="m", version="1", run_id="r", alias="", tags={})
    pyfunc = _FakePyfunc("m")

    service._smoke_score(_FakePipe(), pyfunc, Path("/fake/m/1"), frozenset(), ref)  # no raise


def test_smoke_score_rejects_a_version_that_fails_to_score_its_own_example() -> None:
    """The exact shape of the real paysim-fraud-lightgbm v4 bug: a model that
    loads fine (passes _require_feedable) but whose predict_proba call itself
    breaks on the row it declares it expects."""
    ref = ModelRef(name="m", version="1", run_id="r", alias="", tags={})
    example = pd.DataFrame({"amount": [1.0]})
    pyfunc = _FakePyfunc("m", input_example=example)
    pipe = _FakePipe(predict_proba_error=ValueError("categorical_feature mismatch"))

    with pytest.raises(ModelRegistryError, match="failed to score its own logged input example"):
        service._smoke_score(pipe, pyfunc, Path("/fake/m/1"), frozenset(), ref)


def test_smoke_score_casts_declared_categorical_columns_before_scoring() -> None:
    ref = ModelRef(name="m", version="1", run_id="r", alias="", tags={})
    example = pd.DataFrame({"amount": [1.0], "type": ["TRANSFER"]})
    seen: dict[str, str] = {}

    class _RecordingPipe:
        def predict_proba(self, row: pd.DataFrame) -> Any:
            seen["dtype"] = str(row["type"].dtype)
            return [[0.1, 0.9]]

    pyfunc = _FakePyfunc("m", input_example=example)

    service._smoke_score(_RecordingPipe(), pyfunc, Path("/fake/m/1"), frozenset({"type"}), ref)

    assert seen["dtype"] == "category"


def test_smoke_score_rejects_when_the_logged_example_cannot_be_read() -> None:
    ref = ModelRef(name="m", version="1", run_id="r", alias="", tags={})
    pyfunc = _FakePyfunc("m", load_input_example_error=FileNotFoundError("input_example.json"))

    with pytest.raises(ModelRegistryError, match="could not be read"):
        service._smoke_score(_FakePipe(), pyfunc, Path("/fake/m/1"), frozenset(), ref)


def test_preload_all_versions_excludes_a_version_that_fails_its_smoke_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wired end-to-end through _preload_all_versions: a version that loads
    fine but can't score its own logged input example must be excluded from
    the pool, not left to break the first real /predict after a swap."""

    class _BrokenPipe:
        def predict_proba(self, row: pd.DataFrame) -> Any:
            raise ValueError("categorical_feature mismatch")

    _patch_registry(
        monkeypatch,
        versions_by_name={"good": [_info("1")], "broken": [_info("1")]},
        input_columns_by_name={"good": ("amount",), "broken": ("amount",)},
        pipes_by_name={"broken": _BrokenPipe()},
        input_examples_by_name={"broken": pd.DataFrame({"amount": [1.0]})},
    )

    pool = service._preload_all_versions(_settings(), ["good", "broken"])

    assert set(pool) == {("good", "1")}


def _pool(*refs: ModelRef) -> dict[tuple[str, str], Champion]:
    return {(ref.name, ref.version): Champion(pipe=object(), ref=ref) for ref in refs}


def test_pick_default_version_prefers_the_pinned_version_of_the_default_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(name="modelA", version="2")
    pool = _pool(
        ModelRef(name="modelA", version="1", run_id="r1", alias="", tags={}),
        ModelRef(name="modelA", version="2", run_id="r2", alias="", tags={}),
        ModelRef(name="modelB", version="9", run_id="r9", alias="", tags={}),
    )
    monkeypatch.setattr(service, "list_model_versions", lambda name, *, settings: [])

    assert service._pick_default_version(settings, pool).ref.version == "2"


def test_pick_default_version_prefers_the_champion_alias_when_not_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(name="modelA", version="")
    pool = _pool(
        ModelRef(name="modelA", version="1", run_id="r1", alias="", tags={}),
        ModelRef(name="modelA", version="2", run_id="r2", alias="", tags={}),
    )
    monkeypatch.setattr(
        service,
        "list_model_versions",
        lambda name, *, settings: [_info("1"), _info("2", aliases=("champion",))],
    )

    assert service._pick_default_version(settings, pool).ref.version == "2"


def test_pick_default_version_falls_back_to_highest_version_of_the_default_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(name="modelA", version="")
    pool = _pool(
        ModelRef(name="modelA", version="1", run_id="r1", alias="", tags={}),
        ModelRef(name="modelA", version="3", run_id="r3", alias="", tags={}),
        ModelRef(name="modelB", version="99", run_id="r99", alias="", tags={}),
    )
    monkeypatch.setattr(service, "list_model_versions", lambda name, *, settings: [])

    # Highest version of modelA (3), never modelB's 99 -- version numbers
    # aren't comparable across different model families.
    assert service._pick_default_version(settings, pool).ref.version == "3"


def test_pick_default_version_raises_when_the_default_model_has_nothing_loadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(name="modelA", version="")
    pool = _pool(ModelRef(name="modelB", version="1", run_id="r1", alias="", tags={}))
    monkeypatch.setattr(service, "list_model_versions", lambda name, *, settings: [])

    with pytest.raises(RuntimeError, match="modelA"):
        service._pick_default_version(settings, pool)
