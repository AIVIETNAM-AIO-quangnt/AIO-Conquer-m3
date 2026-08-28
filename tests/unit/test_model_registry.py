"""Pure-logic tests for contracts.model_registry -- no server, no network."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from conquer3.contracts.model_registry import IncompatibleModelError, ModelRef, verify_compatible
from conquer3.core.schema import FEATURE_SCHEMA_VERSION


def test_verify_compatible_accepts_matching_feature_schema_version() -> None:
    verify_compatible({"feature_schema_version": str(FEATURE_SCHEMA_VERSION)})


def test_verify_compatible_rejects_mismatched_feature_schema_version() -> None:
    with pytest.raises(IncompatibleModelError):
        verify_compatible({"feature_schema_version": "999"})


def test_verify_compatible_rejects_missing_tag() -> None:
    with pytest.raises(IncompatibleModelError):
        verify_compatible({})


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
