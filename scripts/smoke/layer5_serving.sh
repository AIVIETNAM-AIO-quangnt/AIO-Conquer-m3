#!/usr/bin/env bash
# Layer 5 gate: the scoring service (conquer3.serving) -- `scorer` IS the
# inference endpoint, remote MLflow is storage only. Back-to-back and batched
# same-account calls carry state correctly, dry_run leaves Redis/events
# untouched, op=model_info reports the resolved version, concurrent
# same-account requests never corrupt state, a champion promotion reloads
# within one poll interval with zero non-2xx responses, a boot against a dead
# remote MLflow falls back to the cached champion (degraded=true on every
# response), and -- the property the whole plan is built around -- killing
# remote MLflow entirely after boot leaves /invocations serving at full
# correctness. Do not start Layer 6 (Airflow DAGs) until this passes.
#
# No docker-compose profile needed: each integration test spins up its OWN
# ephemeral local `mlflow server` (sqlite backend, no Docker) exactly like
# Layer 4's gate, plus its OWN ephemeral Redis via testcontainers exactly like
# Layer 3b's gate. Needs Docker only for that ephemeral Redis container.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

echo "== ruff / mypy / import-linter =="
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run lint-imports

echo "== serving unit tests: signature generation, event sink, FraudScorerModel logic (no infra) =="
uv run pytest -q \
  tests/unit/test_signature.py \
  tests/unit/test_event_sink.py \
  tests/unit/test_pyfunc_model.py

echo "== state store integration tests (ephemeral testcontainers Redis) =="
echo "   -- get/commit round-trip, monotonic-CAS rejects a stale write, hit/miss/rejected counters"
uv run pytest -q tests/integration/test_state_store.py -m integration

echo "== serving end-to-end gate (ephemeral local mlflow server + ephemeral Redis + the real scoring server) =="
echo "   -- back-to-back same-account calls carry state; the same two txns as one batch match exactly"
echo "   -- op=model_info reports the resolved version; dry_run leaves Redis and the event dir untouched"
echo "   -- concurrent same-account requests never corrupt state (real asyncio.to_thread + monotonic CAS)"
echo "   -- not-a-proxy: killing remote MLflow after boot leaves /invocations at full correctness"
echo "   -- a dead-MLflow boot falls back to the cached champion, degraded=true on every response"
echo "   -- promoting a new champion reloads within one poll interval with zero non-2xx responses"
uv run pytest -s tests/integration/test_serving_e2e.py -m integration -v

echo
echo "Layer 5 gate: PASS"
