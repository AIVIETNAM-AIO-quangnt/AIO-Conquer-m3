"""Layer 4 gate: publish/resolve/re-alias/degraded-fallback against a real,
ephemeral local MLflow tracking server (sqlite backend, local artifact root, no
Docker needed) -- the same "ephemeral instead of touching real/absent shared
infra" philosophy Layer 3b used for testcontainers Postgres/Redis, just without
needing a container at all.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from conquer3.config.settings import get_settings
from conquer3.contracts.model_registry import (
    ChampionResolutionError,
    IncompatibleModelError,
    publish_model,
    resolve_champion,
)

pytestmark = [pytest.mark.mlflow]

_STARTUP_TIMEOUT_S = 30.0


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _dummy_model_and_sample() -> tuple[Any, Any, Any]:
    import numpy as np
    import pandas as pd
    from sklearn.dummy import DummyClassifier

    from conquer3.core.schema import CATEGORICAL_FEATURES, FEATURE_NAMES, NUMERIC_FEATURES

    rng = np.random.default_rng(0)
    n = 20
    data: dict[str, object] = {name: rng.normal(size=n) for name in NUMERIC_FEATURES}
    for name in CATEGORICAL_FEATURES:
        data[name] = rng.choice(["a", "b"], size=n)
    x_sample = pd.DataFrame(data, columns=list(FEATURE_NAMES))
    y = rng.integers(0, 2, size=n)
    clf = DummyClassifier(strategy="prior").fit(x_sample, y)
    return clf, x_sample, clf.predict_proba(x_sample)


def _sklearn_version() -> str:
    import sklearn

    return sklearn.__version__


@pytest.fixture
def mlflow_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[str]:
    """Spawns a real local ``mlflow server`` (sqlite backend, local artifact root),
    points MLFLOW_TRACKING_URI + the champion cache at tmp_path, yields the
    tracking URI."""
    port = _free_port()
    backend = tmp_path / "mlflow.db"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    uri = f"http://127.0.0.1:{port}"

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--backend-store-uri",
            f"sqlite:///{backend}",
            "--default-artifact-root",
            str(artifacts),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + _STARTUP_TIMEOUT_S
        healthy = False
        while time.monotonic() < deadline:
            try:
                urllib.request.urlopen(uri + "/health", timeout=1)
                healthy = True
                break
            except Exception:
                time.sleep(0.3)
        if not healthy:
            raise RuntimeError("mlflow server did not become healthy in time")

        monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
        monkeypatch.setenv("C3_MODEL_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.setenv("C3_MODEL_CHAMPION_CACHE_FILE", str(tmp_path / "champion.json"))
        get_settings.cache_clear()
        try:
            yield uri
        finally:
            get_settings.cache_clear()
    finally:
        proc.terminate()
        proc.wait(timeout=15)


def test_publish_and_resolve_happy_path(mlflow_server: str) -> None:
    clf, x_sample, proba = _dummy_model_and_sample()
    ref = publish_model(
        clf,
        x_sample,
        proba,
        sklearn_version=_sklearn_version(),
        code_sha="deadbeef",
        decision_threshold=0.5,
        model_name="gate_model_happy",
        alias_as_champion=True,
    )
    assert ref.alias == "champion"

    model, resolved = resolve_champion("gate_model_happy")
    assert resolved.version == ref.version
    assert resolved.degraded is False
    preds = model.predict(x_sample)
    assert len(preds) == len(x_sample)


def test_re_alias_resolver_follows(mlflow_server: str) -> None:
    clf, x_sample, proba = _dummy_model_and_sample()
    ref1 = publish_model(
        clf,
        x_sample,
        proba,
        sklearn_version=_sklearn_version(),
        code_sha="v1",
        decision_threshold=0.5,
        model_name="gate_model_realias",
        alias_as_champion=True,
    )
    _, resolved1 = resolve_champion("gate_model_realias")
    assert resolved1.version == ref1.version

    ref2 = publish_model(
        clf,
        x_sample,
        proba,
        sklearn_version=_sklearn_version(),
        code_sha="v2",
        decision_threshold=0.7,
        model_name="gate_model_realias",
        alias_as_champion=True,
    )
    assert ref2.version != ref1.version

    _, resolved2 = resolve_champion("gate_model_realias")
    assert resolved2.version == ref2.version


def test_conda_yaml_names_full_mlflow_not_skinny(mlflow_server: str) -> None:
    clf, x_sample, proba = _dummy_model_and_sample()
    ref = publish_model(
        clf,
        x_sample,
        proba,
        sklearn_version=_sklearn_version(),
        code_sha="x",
        decision_threshold=0.5,
        model_name="gate_model_conda",
        alias_as_champion=True,
    )
    resolve_champion("gate_model_conda")

    settings = get_settings()
    conda_yaml = Path(settings.model.cache_dir) / "gate_model_conda" / ref.version / "conda.yaml"
    text = conda_yaml.read_text(encoding="utf-8")
    assert "mlflow==" in text
    assert "mlflow-skinny" not in text


def test_degraded_path_falls_back_to_cache_and_emits_gauge(
    mlflow_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    clf, x_sample, proba = _dummy_model_and_sample()
    publish_model(
        clf,
        x_sample,
        proba,
        sklearn_version=_sklearn_version(),
        code_sha="x",
        decision_threshold=0.5,
        model_name="gate_model_degraded",
        alias_as_champion=True,
    )
    _, ref = resolve_champion("gate_model_degraded")
    assert ref.degraded is False  # populates the cache for the fallback below

    reader = InMemoryMetricReader()
    metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"http://127.0.0.1:{_free_port()}")
    get_settings.cache_clear()

    model2, ref2 = resolve_champion("gate_model_degraded")
    assert ref2.degraded is True
    assert ref2.version == ref.version
    assert len(model2.predict(x_sample)) == len(x_sample)

    values = [
        dp.value
        for rm in reader.get_metrics_data().resource_metrics
        for sm in rm.scope_metrics
        for m in sm.metrics
        if m.name == "c3_model_resolution_degraded"
        for dp in m.data.data_points
    ]
    assert values[-1] == 1


def test_no_cache_and_dead_server_raises(
    mlflow_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"http://127.0.0.1:{_free_port()}")
    get_settings.cache_clear()

    with pytest.raises(ChampionResolutionError):
        resolve_champion("gate_model_never_published")


def test_incompatible_model_rejected(mlflow_server: str) -> None:
    from mlflow.tracking import MlflowClient

    clf, x_sample, proba = _dummy_model_and_sample()
    ref = publish_model(
        clf,
        x_sample,
        proba,
        sklearn_version=_sklearn_version(),
        code_sha="x",
        decision_threshold=0.5,
        model_name="gate_model_bad",
        alias_as_champion=True,
    )

    client = MlflowClient()
    client.set_model_version_tag("gate_model_bad", ref.version, "feature_schema_version", "999")

    with pytest.raises(IncompatibleModelError):
        resolve_champion("gate_model_bad")
