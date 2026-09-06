#!/usr/bin/env bash
# Runs the scorer + ui pair -- what the Makefile calls the "core" group -- as
# plain host processes against `.env`, with no Docker/Compose involved.
#
# Unlike scripts/startup.sh (which brings up the full docker-compose stack),
# this assumes Postgres/Redis/MLflow are already reachable however `.env`
# points at them (external managed services, or an instance you started
# yourself) and only launches the two Python processes: `conquer3 serve`
# (backgrounded, since it blocks) and `conquer3 ui` (foreground -- Ctrl+C
# stops both, mirroring scripts/eval/run_registry_eval.sh's trap pattern).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "FAIL: uv is not installed / not on PATH" >&2
  exit 1
fi
if [ ! -f .env ]; then
  echo "FAIL: .env not found. Run scripts/bootstrap.sh first." >&2
  exit 1
fi

# Nested settings classes (PgSettings, ServingSettings, ...) only read real
# process env vars -- only the top-level Settings declares env_file=".env",
# and that doesn't cascade into them (see README's "conquer3 CLI reference").
set -a
# shellcheck disable=SC1091
source .env
set +a

if [ -z "${MLFLOW_TRACKING_URI:-}" ]; then
  echo "FAIL: MLFLOW_TRACKING_URI is empty in .env -- the scorer resolves a champion" >&2
  echo "at boot and refuses to guess. Fill it in (and register+alias a champion)" >&2
  echo "before running this script." >&2
  exit 1
fi

# At boot the scorer preloads every version of every registered model (see
# service.py's _preload_all_versions), and mlflow 3.x auto-enables a
# presigned-URL artifact download path from /server-info. Confirmed against
# this deployment: that URL points at the registry's own compose-internal
# MinIO (e.g. http://storage:9000), unresolvable from a bare host outside
# that docker network -- CLAUDE.md documents this exact failure mode, costing
# ~90s of retries *per model version* before falling back. Forcing artifacts
# through the tracking server instead avoids it entirely (confirmed here:
# per-version load time drops from ~90s to ~1s). Only defaulted, not forced,
# so an explicit `.env` value (e.g. once this deployment's MinIO is reachable
# from wherever this script runs) still wins.
export MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD="${MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD:-false}"

SCORER_PORT="${C3_SCORER_PORT:-3000}"
UI_PORT="${C3_UI_PORT:-8501}"
# Explicit, not relied-on-default: keeps the ui pointed at whichever port this
# script actually started the scorer on, the same reason docker-compose.yaml
# sets C3_SCORER_URL for the ui service rather than trusting its own default.
export C3_SCORER_URL="http://127.0.0.1:${SCORER_PORT}"

SCORER_LOG="$(mktemp -t conquer3-scorer.XXXXXX.log)"
SCORER_PID=""

cleanup() {
  if [ -n "$SCORER_PID" ] && kill -0 "$SCORER_PID" 2>/dev/null; then
    echo
    echo "== stopping scorer (pid $SCORER_PID) =="
    kill -TERM "$SCORER_PID" 2>/dev/null || true
    wait "$SCORER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "== starting scorer: conquer3 serve (log: $SCORER_LOG) =="
uv run conquer3 serve >"$SCORER_LOG" 2>&1 &
SCORER_PID=$!

# --max-time bounds each individual probe: confirmed against this deployment
# that the port can accept a connection before the app finishes preloading,
# in which case an unbounded curl just hangs on that one request rather than
# failing fast -- without it, one slow probe would silently eat into how
# promptly a died-process is noticed below.
#
# No overall timeout: cold-start preloads every version of every registered
# model (see the MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD note above), and on
# this registry that has consistently taken longer than any fixed budget is
# worth guessing at -- confirmed in practice, not just in theory. The only
# thing that ends this loop early is the scorer process itself exiting.
echo "== waiting for http://127.0.0.1:${SCORER_PORT}/readyz (no timeout -- cold start can take a while) =="
until curl -fsS --max-time 3 -o /dev/null "http://127.0.0.1:${SCORER_PORT}/readyz" 2>/dev/null; do
  if ! kill -0 "$SCORER_PID" 2>/dev/null; then
    echo "FAIL: scorer exited before becoming ready -- log:" >&2
    cat "$SCORER_LOG" >&2
    exit 1
  fi
  sleep 2
done
echo "  scorer: ready"

echo
echo "scorer: http://127.0.0.1:${SCORER_PORT}/docs.json (Swagger UI at /)"
echo "ui:     http://127.0.0.1:${UI_PORT}  (starting now, foreground -- Ctrl+C stops both)"
echo

echo "== starting ui: conquer3 ui =="
uv run conquer3 ui
