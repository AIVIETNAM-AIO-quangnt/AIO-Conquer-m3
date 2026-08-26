#!/usr/bin/env bash
# Layer 1 gate: core infra (postgres, redis, otel-collector) and the Airflow
# pipeline profile come up healthy, and a hello-world DAG runs end to end.
# Do not start Layer 2 (the medallion warehouse) until this passes.
#
# Requires Docker and Docker Compose v2. Not runnable in a sandbox without a Docker
# daemon -- run this on a machine that has one.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

if ! command -v docker >/dev/null 2>&1; then
  echo "FAIL: docker is not installed / not on PATH" >&2
  exit 1
fi
if [ ! -f .env ]; then
  echo "FAIL: .env not found. Run scripts/bootstrap.sh first." >&2
  exit 1
fi

wait_healthy() {
  local service="$1" timeout_s="${2:-90}" waited=0
  while true; do
    local status
    status="$(docker compose ps --format '{{.Health}}' "$service" 2>/dev/null || true)"
    if [ "$status" = "healthy" ]; then
      echo "  $service: healthy"
      return 0
    fi
    if [ "$waited" -ge "$timeout_s" ]; then
      echo "FAIL: $service did not become healthy within ${timeout_s}s (last status: ${status:-unknown})" >&2
      docker compose logs --tail=50 "$service" >&2 || true
      exit 1
    fi
    sleep 3
    waited=$((waited + 3))
  done
}

echo "== profile: core =="
docker compose --profile core up -d --build
wait_healthy postgres
wait_healthy redis

echo "  postgres: SELECT 1"
docker compose exec -T postgres psql -U "${POSTGRES_USER:-conquer3}" -d "${POSTGRES_DB:-conquer3}" -c "SELECT 1;" >/dev/null

echo "  redis: PING"
docker compose exec -T redis redis-cli ping | grep -q PONG

echo "  otel-collector: :13133 health_check extension"
# No in-container healthcheck is possible (see docker-compose.yaml) -- poll the
# host-mapped port instead, which is what the Layer 1 gate actually specifies.
waited=0
until curl -fsS "http://localhost:13133" >/dev/null 2>&1; do
  waited=$((waited + 2))
  if [ "$waited" -ge 60 ]; then
    echo "FAIL: otel-collector :13133 did not return 200 within 60s" >&2
    docker compose logs --tail=50 otel-collector >&2 || true
    exit 1
  fi
  sleep 2
done
echo "  otel-collector: OK"

echo
echo "== profile: pipeline =="
docker compose --profile pipeline up -d --build
echo "  waiting for airflow-init to complete..."
docker compose wait airflow-init
wait_healthy airflow-apiserver 180
wait_healthy airflow-scheduler 120
wait_healthy airflow-dag-processor 120
wait_healthy airflow-triggerer 120

echo "  checking for zero DAG import errors..."
import_errors="$(docker compose exec -T airflow-apiserver airflow dags list-import-errors 2>&1 || true)"
if echo "$import_errors" | grep -qi "traceback\|error"; then
  echo "FAIL: DAG import errors present:" >&2
  echo "$import_errors" >&2
  exit 1
fi
echo "  zero DAG import errors"

echo "  running the hello_world DAG..."
docker compose exec -T airflow-apiserver airflow dags unpause hello_world >/dev/null 2>&1 || true
run_id="smoke-$(date +%s)"
docker compose exec -T airflow-apiserver airflow dags trigger hello_world --run-id "$run_id" >/dev/null

waited=0
while true; do
  state="$(docker compose exec -T airflow-apiserver airflow dags list-runs -d hello_world --output json 2>/dev/null \
    | python3 -c "
import json, sys
runs = json.load(sys.stdin)
match = [r for r in runs if r.get('run_id') == '$run_id']
print(match[0]['state'] if match else 'unknown')
")"
  case "$state" in
    success) echo "  hello_world run $run_id: success"; break ;;
    failed) echo "FAIL: hello_world run $run_id failed" >&2; docker compose logs --tail=80 airflow-scheduler >&2; exit 1 ;;
  esac
  waited=$((waited + 3))
  if [ "$waited" -ge 120 ]; then
    echo "FAIL: hello_world run $run_id did not finish within 120s (state: $state)" >&2
    exit 1
  fi
  sleep 3
done

echo
echo "Layer 1 gate: PASS"
echo "Airflow UI: http://localhost:8080  (user/pass from _AIRFLOW_WWW_USER_* in .env)"
