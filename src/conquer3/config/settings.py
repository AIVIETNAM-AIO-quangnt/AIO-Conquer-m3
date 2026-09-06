"""The single place environment variables are read, and (for non-secret tuning
knobs) the single place ``configs/default.yaml`` is read.

No other module should call ``os.getenv`` / ``os.environ`` directly -- that
invariant is what lets `.env.example` double as documentation of every knob the
system has, and what makes each subsystem's config testable without real env vars.
Grep for ``os.getenv`` outside this file to audit it. The one deliberate exception
is ``C3_CONFIG_PATH`` below: it doesn't set a tuning value, it only says which YAML
file to load, which Docker images with differing ``WORKDIR``s need.

Every setting has a ``C3_`` prefix except the third-party ones (``POSTGRES_*``,
``MLFLOW_*``, ``REDIS_*``, ``AIRFLOW_*``, ``OTEL_*``) that follow each tool's own
convention, since those are frequently read by the tool's own code too.

**Two config sources, split by kind, never both for the same field:**

* ``.env`` / real env vars -- secrets and per-deployment connection info (hosts,
  ports, credentials, URLs), and host-vs-container-path fields that must differ
  between a host-side ``uv run conquer3 ...`` invocation and a Docker container
  (e.g. ``ModelSettings.cache_dir``).
* ``configs/default.yaml`` -- non-secret business/tuning knobs (thresholds,
  timeouts, retry counts, worker counts, TTLs), versioned with the code. These
  fields read **only** the YAML; setting the equivalent ``C3_*`` env var has no
  effect on them.

A settings class whose fields are entirely one kind is a plain ``BaseSettings``
(env-only, e.g. :class:`PgSettings`) or a plain ``BaseModel`` (yaml-only, e.g.
:class:`StateSettings`). A class split between the two kinds (e.g.
:class:`ModelSettings`) is a plain ``BaseModel`` fed by a private, env-only
``BaseSettings`` half (e.g. ``_ModelEnv``) -- see :func:`_yaml_section` and
``Settings.__init__`` below for how the two get merged into one object per
section, so every consumer still does a single ``settings.model.whatever``
regardless of which file that particular field came from.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Says which configs/default.yaml to load -- not itself a tuning value, see the
# module docstring's "one deliberate exception" note. Docker images whose final
# WORKDIR isn't where `configs/` was copied to (airflow.Dockerfile) set this.
_CONFIG_PATH_ENV: Final = "C3_CONFIG_PATH"
_DEFAULT_CONFIG_PATH: Final = Path("configs/default.yaml")


@lru_cache(maxsize=1)
def _load_yaml_defaults() -> Mapping[str, Any]:
    """The whole of ``configs/default.yaml``, or ``{}`` if it isn't present --
    missing is tolerated (falls back to each field's Python-level default)
    rather than an error, so a bare ``pip install`` outside this repo doesn't
    hard-fail just for constructing ``Settings``."""
    path = Path(os.environ.get(_CONFIG_PATH_ENV, _DEFAULT_CONFIG_PATH))
    if not path.is_file():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _yaml_section(name: str) -> dict[str, Any]:
    """The ``name:`` top-level section of ``configs/default.yaml``, as kwargs
    for the matching settings class."""
    return dict(_load_yaml_defaults().get(name, {}))


class PgSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POSTGRES_", extra="ignore")

    host: str = "localhost"
    port: int = 5432
    db: str = "conquer3"
    user: str = "conquer3"
    password: str = "change-me"
    # Neon's own connection-string defaults -- Neon requires TLS, and channel
    # binding pins the SCRAM handshake to the TLS channel it rode in on. Local
    # docker-compose Postgres has no SSL configured, so local dev overrides both
    # to "disable" via .env (see .env.example).
    sslmode: str = "require"
    channel_binding: str = "require"

    @property
    def libpq_dsn(self) -> str:
        """DSN in libpq keyword form, for DuckDB's ATTACH ... (TYPE postgres)."""
        return (
            f"host={self.host} port={self.port} dbname={self.db} "
            f"user={self.user} password={self.password} "
            f"sslmode={self.sslmode} channel_binding={self.channel_binding}"
        )

    @property
    def sqlalchemy_url(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"
            f"?sslmode={self.sslmode}&channel_binding={self.channel_binding}"
        )


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDIS_", extra="ignore")

    host: str = "localhost"
    username: str = "default"
    port: int = 6379
    db: int = 0
    password: str | None = None
    # Managed Redis providers commonly require TLS (plaintext connections get
    # closed immediately, no protocol-level error). Local docker-compose Redis
    # has no TLS configured, so local dev overrides this to false via .env (see
    # .env.example) -- same reasoning as PgSettings.sslmode/channel_binding.
    tls: bool = True


class StateSettings(BaseModel):
    """Entirely yaml-sourced -- see configs/default.yaml's `state:` section."""

    key_prefix: str = "c3"
    ttl_days: int = 90
    cas_retries: int = 3
    pg_fallback: bool = False


class _MlflowEnv(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MLFLOW_", extra="ignore")

    tracking_uri: str = ""
    tracking_username: str | None = None
    tracking_password: str | None = None


class MlflowSettings(BaseModel):
    tracking_uri: str = ""
    tracking_username: str | None = None
    tracking_password: str | None = None
    experiment_name: str = "paysim-fraud"  # yaml-sourced, see configs/default.yaml


class _ModelEnv(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="C3_MODEL_", extra="ignore")

    cache_dir: str = "/models"
    champion_cache_file: str = "/models/champion.json"


class ModelSettings(BaseModel):
    # Host-vs-container paths, env-sourced (see .env.example).
    cache_dir: str = "/models"
    champion_cache_file: str = "/models/champion.json"
    # Everything else is yaml-sourced -- see configs/default.yaml's `model:` section.
    name: str = "paysim_fraud_clf"
    alias: str = "champion"
    # Pins the scorer to this exact registered version (name + version), for a
    # model that isn't (or isn't yet) aliased -- resolve_version, not
    # resolve_champion, in serving/service.py's _resolve_sklearn_champion and
    # _pick_default_version. "" (the default) keeps the usual alias-based
    # behavior; resolve_champion/promote_champion/list_model_versions are
    # untouched by this either way, still operating on the alias. Plain str,
    # not str | None: an empty yaml key parses as None and breaks every other
    # field here the same way (see the incident that added this field) --
    # "" is representable and unambiguous in yaml, None isn't worth risking again.
    version: str = ""
    resolve_timeout_s: int = 10
    resolve_max_retries: int = 2


class _PathwayEnv(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="C3_PATHWAY_", extra="ignore")

    license_key: str = Field(default="", alias="PATHWAY_LICENSE_KEY")


class PathwaySettings(BaseModel):
    license_key: str = ""  # secret, env-sourced (see .env.example)
    # Everything else is yaml-sourced -- see configs/default.yaml's `pathway:` section.
    pg_sink: str = "auto"  # auto | licensed | psycopg
    mode: str = "streaming"  # streaming | static
    persist_dir: str = "/pathway-state"
    autocommit_ms: int = 500
    refresh_ms: int = 1000


class EventSettings(BaseModel):
    """Entirely yaml-sourced -- see configs/default.yaml's `event:` section."""

    dir: str = "/events"
    rotate_seconds: int = 3600
    fsync_interval_ms: int = 1000
    staging_dir: str = "/staging"


class _DuckEnv(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="C3_DUCKDB_", extra="ignore")

    path: str = "/duckdb/analytics.duckdb"
    temp_dir: str = "/duckdb/tmp"


class DuckSettings(BaseModel):
    # Host-vs-container paths, env-sourced (see .env.example).
    path: str = "/duckdb/analytics.duckdb"
    temp_dir: str = "/duckdb/tmp"
    # Yaml-sourced -- see configs/default.yaml's `duck:` section.
    memory_limit: str = "4GB"
    threads: int = 4


class _ServingEnv(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="C3_", extra="ignore")

    scorer_host: str = "0.0.0.0"
    scorer_port: int = 3000
    # Pointer file naming the champion version the workers must load -- a
    # host-vs-container path, same reasoning as ModelSettings.cache_dir above.
    active_champion_file: str = "/models/active.json"
    # Dev-mode toggle, not a tuning knob -- env-sourced like the paths above,
    # never yaml (a real deployment must never pick this up from a versioned
    # default). See supervisor.py's _spawn().
    scorer_reload: bool = False


class ServingSettings(BaseModel):
    scorer_host: str = "0.0.0.0"
    scorer_port: int = 3000
    active_champion_file: str = "/models/active.json"
    scorer_reload: bool = False
    # Everything else is yaml-sourced -- see configs/default.yaml's `serving:` section.
    decision_threshold: float = 0.5
    scorer_workers: int = 2
    scorer_timeout_s: int = 60
    champion_poll_s: int = 300


class OtelSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OTEL_", extra="ignore")

    exporter_otlp_endpoint: str | None = Field(default=None, alias="OTEL_EXPORTER_OTLP_ENDPOINT")
    exporter_otlp_protocol: str = "grpc"
    resource_attributes: str = "service.namespace=conquer3"

    @property
    def enabled(self) -> bool:
        """No-op providers when unset -- tests and Colab never need a collector."""
        return bool(self.exporter_otlp_endpoint)


class _UiEnv(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="C3_UI_", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8501
    # Alias escape hatch, same reasoning as PathwaySettings.license_key and
    # KaggleSettings.csv_path above -- a bare C3_UI_SCORER_URL would read oddly
    # next to the existing C3_SCORER_* block this points at.
    scorer_url: str = Field(default="http://localhost:3000", alias="C3_SCORER_URL")


class UiSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8501
    scorer_url: str = "http://localhost:3000"
    # Yaml-sourced -- see configs/default.yaml's `ui:` section.
    history_max_rows: int = 50000
    history_max_files: int = 200
    request_batch_size: int = 500
    request_timeout_s: int = 120


class _KaggleEnv(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KAGGLE_", extra="ignore")

    username: str | None = None
    key: str | None = None
    # conquer3's own knob, not Kaggle's -- alias overrides the class's KAGGLE_ prefix
    # the same way PathwaySettings.license_key does for PATHWAY_LICENSE_KEY above.
    # The one place `conquer3 ingest download/bronze` and scripts/smoke/layer2_
    # warehouse.sh all agree on where the raw PaySim1 CSV lives.
    csv_path: str = Field(default="data/raw/paysim1.csv", alias="C3_PAYSIM_CSV_PATH")


class KaggleSettings(BaseModel):
    username: str | None = None
    key: str | None = None
    csv_path: str = "data/raw/paysim1.csv"
    dataset: str = "ealaxi/paysim1"  # yaml-sourced, see configs/default.yaml


# Matches pydantic-settings' own `DotenvType | None` (the type of `_env_file` on
# `BaseSettings.__init__` and of `SettingsConfigDict["env_file"]`) -- narrowing this
# to `str | None` would reject the `Path`/list-of-paths forms pydantic-settings
# itself accepts for `env_file`.
_EnvFile = Path | str | Sequence[Path | str] | None


def _build_state(_env_file: _EnvFile) -> StateSettings:
    return StateSettings(**_yaml_section("state"))


def _build_mlflow(env_file: _EnvFile) -> MlflowSettings:
    return MlflowSettings(**_MlflowEnv(_env_file=env_file).model_dump(), **_yaml_section("mlflow"))


def _build_model(env_file: _EnvFile) -> ModelSettings:
    return ModelSettings(**_ModelEnv(_env_file=env_file).model_dump(), **_yaml_section("model"))


def _build_pathway(env_file: _EnvFile) -> PathwaySettings:
    return PathwaySettings(
        **_PathwayEnv(_env_file=env_file).model_dump(), **_yaml_section("pathway")
    )


def _build_event(_env_file: _EnvFile) -> EventSettings:
    return EventSettings(**_yaml_section("event"))


def _build_duck(env_file: _EnvFile) -> DuckSettings:
    return DuckSettings(**_DuckEnv(_env_file=env_file).model_dump(), **_yaml_section("duck"))


def _build_serving(env_file: _EnvFile) -> ServingSettings:
    return ServingSettings(
        **_ServingEnv(_env_file=env_file).model_dump(), **_yaml_section("serving")
    )


def _build_ui(env_file: _EnvFile) -> UiSettings:
    return UiSettings(**_UiEnv(_env_file=env_file).model_dump(), **_yaml_section("ui"))


def _build_kaggle(env_file: _EnvFile) -> KaggleSettings:
    return KaggleSettings(**_KaggleEnv(_env_file=env_file).model_dump(), **_yaml_section("kaggle"))


# Every nested settings class above is its own independent construction, so
# building one via `default_factory` (the plain pydantic way) never sees
# Settings.model_config's env_file="." + "env" -- pydantic-settings does not cascade
# a parent's env_file into a nested BaseSettings field. Left alone, every nested
# class is blind to .env and only ever reads real process env vars, silently
# falling back to its class defaults (which are container paths/hosts for several
# of them) whenever .env alone was supposed to supply the override. Settings.__init__
# below closes that gap by building each nested section with the *same* env_file
# Settings itself resolved to -- so `Settings()` makes every nested section read .env
# too, and `Settings(_env_file=None)` (as tests/unit/test_settings.py uses for
# isolation) correctly keeps every nested section blind to .env as well. The yaml
# half of a mixed/yaml-only section is unaffected by env_file either way -- it
# only ever reads configs/default.yaml (see _yaml_section).
_NESTED_SETTINGS_FACTORIES: Final[Mapping[str, Callable[[_EnvFile], BaseModel]]] = {
    "pg": lambda env_file: PgSettings(_env_file=env_file),
    "redis": lambda env_file: RedisSettings(_env_file=env_file),
    "state": _build_state,
    "mlflow": _build_mlflow,
    "model": _build_model,
    "pathway": _build_pathway,
    "event": _build_event,
    "duck": _build_duck,
    "serving": _build_serving,
    "otel": lambda env_file: OtelSettings(_env_file=env_file),
    "ui": _build_ui,
    "kaggle": _build_kaggle,
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="C3_", env_file=".env", extra="ignore")

    env: str = "local"
    log_level: str = "INFO"
    sim_epoch_iso: str = "2024-01-01T00:00:00Z"

    pg: PgSettings = Field(default_factory=PgSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    state: StateSettings = Field(default_factory=StateSettings)
    mlflow: MlflowSettings = Field(default_factory=MlflowSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    pathway: PathwaySettings = Field(default_factory=PathwaySettings)
    event: EventSettings = Field(default_factory=EventSettings)
    duck: DuckSettings = Field(default_factory=DuckSettings)
    serving: ServingSettings = Field(default_factory=ServingSettings)
    otel: OtelSettings = Field(default_factory=OtelSettings)
    ui: UiSettings = Field(default_factory=UiSettings)
    kaggle: KaggleSettings = Field(default_factory=KaggleSettings)

    def __init__(self, **kwargs: Any) -> None:
        # Match pydantic-settings' own rule: an *explicitly passed* _env_file
        # (including None) wins; otherwise fall back to this class's configured
        # env_file -- never a bare `default_factory()` call.
        env_file = (
            kwargs["_env_file"] if "_env_file" in kwargs else self.model_config.get("env_file")
        )
        for field_name, factory in _NESTED_SETTINGS_FACTORIES.items():
            kwargs.setdefault(field_name, factory(env_file))
        super().__init__(**kwargs)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton. Tests that need different env vars should call
    ``get_settings.cache_clear()`` after mutating the environment. Tests that
    need different configs/default.yaml values should also call
    ``_load_yaml_defaults.cache_clear()`` after setting C3_CONFIG_PATH."""
    return Settings()
