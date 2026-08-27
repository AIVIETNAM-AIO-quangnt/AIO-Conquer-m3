#!/usr/bin/env bash
# Layer 3b gate: the Pathway feature engine -- static backfill row-count parity,
# licensed-vs-psycopg-fallback output parity, streaming pickup latency, and
# kill/restart deduplication. Do not start Layer 4 (model contract) until this
# passes.
#
# No shared Docker warehouse needed: unlike scripts/smoke/layer2_warehouse.sh,
# this gate does NOT seed synthetic rows into the real docker-compose Postgres.
# `export_staging` (pipelines/transforms/export_staging.py) is a full-refresh
# export -- it always processes the *entire* silver.txn -- so testing it against
# the shared dev Postgres (which, once Layer 2 has run, holds the real 6.3M-row
# PaySim dataset) means every run re-exports and re-backfills all of it, which is
# both slow and pollutes shared dev state if interrupted. The pytest integration
# tests below spin up their OWN ephemeral, empty Postgres+Redis via testcontainers
# instead, so a tiny synthetic seed stays tiny. This mirrors
# scripts/smoke/layer3_feature_core.sh's lint+pytest style, not layer2's.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

echo "== ruff / mypy / import-linter =="
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run lint-imports

echo "== conquer3.core stays untouched by this layer (Colab install path) =="
uv run pytest tests/contract/test_core_is_dependency_light.py

echo "== pathway unit tests (no Docker, pw.debug only) =="
uv run pytest tests/unit/test_pathway_schemas.py tests/unit/test_pathway_accumulator.py

echo "== pathway integration tests (ephemeral testcontainers Postgres+Redis) =="
echo "   -- static backfill row-count parity"
echo "   -- licensed connector vs psycopg fallback: byte-identical output"
echo "   -- streaming: hand-written JSONL picked up in under 2s"
echo "   -- kill/restart: resumes without duplicating"
uv run pytest tests/integration/test_pathway_backfill.py tests/integration/test_pathway_streaming.py -m pathway -v

echo
echo "Layer 3b gate: PASS"
