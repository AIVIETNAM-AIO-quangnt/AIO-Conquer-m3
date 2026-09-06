"""Settings load with defaults and pick up env overrides via the documented prefixes."""

from __future__ import annotations

from conquer3.config.settings import Settings


def test_defaults_load_without_any_env_vars(monkeypatch) -> None:
    for var in list(__import__("os").environ):
        if var.startswith(("C3_", "POSTGRES_", "REDIS_", "MLFLOW_", "OTEL_", "KAGGLE_")):
            monkeypatch.delenv(var, raising=False)
    settings = Settings(_env_file=None)
    assert settings.pg.host == "localhost"
    assert settings.pg.sslmode == "require"
    assert settings.pg.channel_binding == "require"
    assert settings.otel.enabled is False
    assert settings.pathway.pg_sink == "auto"


def test_nested_env_prefix_overrides(monkeypatch) -> None:
    monkeypatch.setenv("POSTGRES_HOST", "warehouse.internal")
    monkeypatch.setenv("PATHWAY_LICENSE_KEY", "abc123")
    settings = Settings(_env_file=None)
    assert settings.pg.host == "warehouse.internal"
    assert settings.pathway.license_key == "abc123"


def test_model_alias_is_yaml_only_and_ignores_its_similarly_named_env_var(monkeypatch) -> None:
    """C3_MODEL_ALIAS/C3_MODEL_NAME look like real overrides but are deliberately
    inert -- see settings.py's module docstring ("two config sources, split by
    kind, never both for the same field"). model.name/alias are yaml-sourced
    only; this guards against that ever silently changing."""
    monkeypatch.setenv("C3_MODEL_ALIAS", "shadow")
    settings = Settings(_env_file=None)
    assert settings.model.alias == "champion"


def test_otel_disabled_without_endpoint(monkeypatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    settings = Settings(_env_file=None)
    assert settings.otel.enabled is False


def test_otel_enabled_with_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317")
    settings = Settings(_env_file=None)
    assert settings.otel.enabled is True
