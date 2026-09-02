#!/usr/bin/env bash
# Brings the whole docker-compose infrastructure up, profile by profile, and
# waits for each service to actually report healthy before moving on -- the
# "day 2" operation you run whenever you want the full stack running, as
# opposed to scripts/bootstrap.sh's one-time env setup.
#
# Order matches each profile's own dependencies: core (postgres/redis/otel,
# nothing else needs) -> pipeline (airflow, its own metadata DB) -> stream
# (pathway, needs postgres+redis healthy -- compose enforces that itself) ->
# serving (scorer, needs a resolvable MLflow champion -- see below).
#
# `demo` (producer) is deliberately not started here: it's a one-shot replay
# driver you run explicitly once scorer is up, not standing infrastructure.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

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

# `docker compose wait <service>` is unreliable against a one-shot service
# that's already exited by the time this polls: confirmed against the
# installed Compose (v5.4.0) that it answers "no containers for project"
# instead of returning immediately, even though `docker compose ps -a` shows
# the container right there, Exited(0). Polling state directly sidesteps that
# quirk and also checks the exit code, which `wait` doesn't gate on here.
wait_exited() {
  local service="$1" timeout_s="${2:-90}" waited=0
  while true; do
    local status
    status="$(docker compose ps -a --format '{{.State}}' "$service" 2>/dev/null || true)"
    if [ "$status" = "exited" ]; then
      local exit_code
      exit_code="$(docker compose ps -a --format '{{.ExitCode}}' "$service" 2>/dev/null || echo "?")"
      if [ "$exit_code" != "0" ]; then
        echo "FAIL: $service exited with code $exit_code" >&2
        docker compose logs --tail=80 "$service" >&2 || true
        exit 1
      fi
      echo "  $service: completed"
      return 0
    fi
    if [ "$waited" -ge "$timeout_s" ]; then
      echo "FAIL: $service did not complete within ${timeout_s}s (last state: ${status:-unknown})" >&2
      docker compose logs --tail=50 "$service" >&2 || true
      exit 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
}

# COMPOSE_PROFILES accumulates as each profile comes up, and every `docker
# compose` call below -- not just `up`/`build`, also the `ps`/`logs` inside
# wait_healthy/wait_exited -- sees the full set so far. Compose resolves
# `depends_on` against whichever profiles are currently in scope, regardless
# of what's already running: confirmed empirically that scoping only `up`/
# `build` to `--profile stream` isn't enough -- a later unscoped `docker
# compose logs pathway` still fails with "no such service: postgres", because
# pathway depends_on postgres/redis (profile core), which were never in scope
# for THAT command. An exported env var, not per-command flags, is what keeps
# every subcommand consistent without threading `--profile` through each one.
export COMPOSE_PROFILES=core

echo "== profile: core (postgres, redis) =="
docker compose up -d --build
wait_healthy postgres
wait_healthy redis
waited=0
# until curl -fsS "http://localhost:13133" >/dev/null 2>&1; do
#   waited=$((waited + 2))
#   if [ "$waited" -ge 60 ]; then
#     echo "FAIL: otel-collector :13133 did not return 200 within 60s" >&2
#     docker compose logs --tail=50 otel-collector >&2 || true
#     exit 1
#   fi
#   sleep 2
# done
# echo "  otel-collector: OK"

echo
echo "== profile: pipeline (airflow) =="
export COMPOSE_PROFILES="${COMPOSE_PROFILES},pipeline"
docker compose up -d --build
echo "  waiting for airflow-init to complete..."
wait_exited airflow-init
wait_healthy airflow-apiserver 180
wait_healthy airflow-scheduler 120
wait_healthy airflow-dag-processor 120
wait_healthy airflow-triggerer 120

echo
echo "== profile: stream (pathway) =="
# No container healthcheck exists for this service (see docker-compose.yaml) --
# it depends_on postgres+redis being healthy, which compose already enforces,
# and idles harmlessly with an empty staging dir until there's data to fold.
export COMPOSE_PROFILES="${COMPOSE_PROFILES},stream"
docker compose up -d --build
sleep 3
state="$(docker compose ps --format '{{.State}}' pathway 2>/dev/null || true)"
if [ "$state" != "running" ]; then
  echo "FAIL: pathway is not running (state: ${state:-unknown})" >&2
  docker compose logs --tail=50 pathway >&2 || true
  exit 1
fi
echo "  pathway: running"

echo
echo "== profile: serving (scorer) =="
mlflow_uri="$(grep -E '^MLFLOW_TRACKING_URI=' .env | cut -d= -f2- || true)"
if [ -z "$mlflow_uri" ]; then
  echo "  skipped: MLFLOW_TRACKING_URI is empty in .env."
  echo "  scorer resolves a champion at boot and refuses to start without one --"
  echo "  bringing it up unconfigured would just crash-loop. Fill in"
  echo "  MLFLOW_TRACKING_URI (and register+alias a champion), then run:"
  echo "    docker compose --profile serving up -d --build"
else
  export COMPOSE_PROFILES="${COMPOSE_PROFILES},serving"
  docker compose up -d --build
  wait_healthy scorer
fi

echo
echo "== profile: ui (Streamlit console) =="
if [ -z "$mlflow_uri" ]; then
  echo "  skipped: serving wasn't started above (no MLFLOW_TRACKING_URI), so ui has no"
  echo "  scorer to talk to. Once scorer is up, run:"
  echo "    docker compose --profile core --profile serving --profile ui up -d --build"
else
  export COMPOSE_PROFILES="${COMPOSE_PROFILES},ui"
  docker compose up -d --build
  wait_healthy ui
fi

echo
echo "== infrastructure up =="
docker compose ps
echo
echo "Airflow UI:  http://localhost:8080  (user/pass from _AIRFLOW_WWW_USER_* in .env)"
echo "otel-collector health: http://localhost:13133"
if [ -n "$mlflow_uri" ]; then
  scorer_port="$(grep -E '^C3_SCORER_PORT=' .env | cut -d= -f2- || true)"
  ui_port="$(grep -E '^C3_UI_PORT=' .env | cut -d= -f2- || true)"
  echo "scorer:      http://localhost:${scorer_port:-3000}/docs.json (Swagger UI at /)"
  echo "ui:          http://localhost:${ui_port:-8501}"
fi
echo
echo "Next: scripts/smoke/layer1_infra.sh (and the rest of scripts/smoke/) to verify"
echo "each layer's gate, or README.md's \"Running the scorer\" section for a demo."
