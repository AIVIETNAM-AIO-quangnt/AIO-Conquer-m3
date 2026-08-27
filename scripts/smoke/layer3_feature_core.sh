#!/usr/bin/env bash
# Layer 3 gate: the shared feature core (conquer3.core) -- golden vectors, the
# cold-start policy, the Hypothesis merge-associativity property test, and an
# exact-equality 100k-row Python/DuckDB parity sweep for derive_event_ts_us (the
# "3 Feature core" row in the plan's verification-gates table). Do not start
# Layer 3b (Pathway) until this passes.
#
# No Docker needed: conquer3.core is dependency-light by construction (see
# tests/contract/test_core_is_dependency_light.py), so this gate never touches
# infra. It re-checks the subset of Layer 0's full suite that is specifically
# this layer's own checkpoint, so it can be run standalone.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

echo "== ruff / mypy / import-linter =="
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run lint-imports

echo "== golden vectors, cold-start policy, merge associativity (tests/unit/test_features.py) =="
uv run pytest tests/unit/test_features.py

echo "== step -> timestamp determinism (tests/unit/test_timeref.py) =="
uv run pytest tests/unit/test_timeref.py

echo "== derive_event_ts_us: exact-equality parity vs DuckDB, incl. the 100k-row sweep =="
uv run pytest tests/parity/test_event_ts_us_sql.py

echo "== conquer3.core pulls in zero heavy dependencies (the Colab install path) =="
uv run pytest tests/contract/test_core_is_dependency_light.py

echo
echo "Layer 3 gate: PASS"
