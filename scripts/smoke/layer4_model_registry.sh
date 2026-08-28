#!/usr/bin/env bash
# Layer 4 gate: the MLflow model contract (contracts/model_registry.py) -- publish
# registers a version with tags + signature, pyfunc.load_model downloads and
# predicts, re-aliasing to another version makes the resolver follow, a dead
# MLFLOW_TRACKING_URI falls back to the cached champion and flips the
# c3_model_resolution_degraded gauge, and the logged conda.yaml names mlflow, never
# mlflow-skinny. Do not start Layer 5 (serving) until this passes.
#
# No Docker, no docker-compose profile, no .env needed: unlike layer2_warehouse.sh,
# this gate never touches the real (deliberately absent) MLflow service. Each
# integration test spins up its OWN ephemeral local `mlflow server` (sqlite
# backend, local artifact root, a free port) -- same "ephemeral instead of
# touching real/absent shared infra" philosophy layer3b_pathway.sh uses for
# testcontainers Postgres/Redis, just without needing a container at all.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

echo "== ruff / mypy / import-linter =="
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run lint-imports

echo "== model_registry unit tests: verify_compatible, ModelRef round-trip (no server) =="
uv run pytest -s tests/unit/test_model_registry.py 

echo "== model_registry integration tests (ephemeral local mlflow server, no Docker) =="
echo "   -- publish registers a version with all tags + a logged signature"
echo "   -- pyfunc.load_model downloads through the tracking server and predicts"
echo "   -- re-alias to another version -> resolver follows"
echo "   -- dead MLFLOW_TRACKING_URI -> cached-champion degraded path loads, gauge reads 1"
echo "   -- conda.yaml names mlflow, never mlflow-skinny"
uv run pytest -s tests/integration/test_model_registry_e2e.py

echo
echo "Layer 4 gate: PASS"
