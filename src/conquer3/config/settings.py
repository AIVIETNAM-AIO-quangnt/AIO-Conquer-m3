"""The single place environment variables are read.

No other module should call ``os.getenv`` / ``os.environ`` directly -- that
invariant is what lets `.env.example` double as documentation of every knob the
system has, and what makes each subsystem's config testable without real env vars.
Grep for ``os.getenv`` outside this file to audit it.

Every setting has a ``C3_`` prefix except the third-party ones (``POSTGRES_*``,
``MLFLOW_*``, ``REDIS_*``, ``AIRFLOW_*``, ``OTEL_*``) that follow each tool's own
convention, since those are frequently read by the tool's own code too.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PgSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POSTGRES_", extra="ignore")

    host: str = "localhost"
    port: int = 5432
    db: str = "conquer3"
    user: str = "conquer3"
    password: str = "change-me"

    @property
    def libpq_dsn(self) -> str:
        """DSN in libpq keyword form, for DuckDB's ATTACH ... (TYPE postgres)."""
        return (
            f"host={self.host} port={self.port} dbname={self.db} "
            f"user={self.user} password={self.password}"
        )

    @property
    def sqlalchemy_url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDIS_", extra="ignore")

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None


class StateSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="C3_STATE_", extra="ignore")

    key_prefix: str = "c3"
    ttl_days: int = 90
    cas_retries: int = 3
    pg_fallback: bool = False  # keep Postgres out of the request hot path by default


class MlflowSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MLFLOW_", extra="ignore")

    tracking_uri: str = ""
    tracking_username: str | None = None
    tracking_password: str | None = None
    experiment_name: str = "paysim-fraud"


class ModelSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="C3_MODEL_", extra="ignore")

    name: str = "paysim_fraud_clf"
    alias: str = "champion"
    cache_dir: str = "/models"
    champion_cache_file: str = "/models/champion.json"


class PathwaySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="C3_PATHWAY_", extra="ignore")

    license_key: str = Field(default="", alias="PATHWAY_LICENSE_KEY")
    pg_sink: str = "auto"  # auto | licensed | psycopg
    mode: str = "streaming"  # streaming | static
    persist_dir: str = "/pathway-state"
    autocommit_ms: int = 500
    refresh_ms: int = 1000


class EventSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="C3_EVENT_", extra="ignore")

    dir: str = "/events"
    rotate_seconds: int = 3600
    fsync_interval_ms: int = 1000
    staging_dir: str = "/staging"


class DuckSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="C3_DUCKDB_", extra="ignore")

    path: str = "/duckdb/analytics.duckdb"
    memory_limit: str = "4GB"
    threads: int = 4
    temp_dir: str = "/duckdb/tmp"


class ServingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="C3_", extra="ignore")

    decision_threshold: float = 0.5
    admin_token: str = "change-me"


class OtelSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OTEL_", extra="ignore")

    exporter_otlp_endpoint: str | None = Field(default=None, alias="OTEL_EXPORTER_OTLP_ENDPOINT")
    exporter_otlp_protocol: str = "grpc"
    resource_attributes: str = "service.namespace=conquer3"

    @property
    def enabled(self) -> bool:
        """No-op providers when unset -- tests and Colab never need a collector."""
        return bool(self.exporter_otlp_endpoint)


class KaggleSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KAGGLE_", extra="ignore")

    username: str | None = None
    key: str | None = None
    dataset: str = "ealaxi/paysim1"


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
    kaggle: KaggleSettings = Field(default_factory=KaggleSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton. Tests that need different env vars should call
    ``get_settings.cache_clear()`` after mutating the environment."""
    return Settings()
