#!/usr/bin/env bash
# Runs tests/eval/test_registry_model_eval.py end-to-end against the REAL
# remote MLflow registry configured in .env -- one command instead of a
# by-hand "start the scorer, remember to pin one worker, run pytest,
# remember to kill it" sequence.
#
# Not a correctness gate (see the eval suite's own module docstring) and
# deliberately not numbered into scripts/smoke/layer*.sh -- this exercises
# whatever is really registered on the remote registry today, not a fixed,
# reproducible fixture the way every layer*.sh gate does.
#
# Forces a single worker: POST /switch_model is per-worker (see the eval
# suite's docstring), so a multi-worker scorer would let BentoML's own load
# balancing route /switch_model and the /predict calls that follow it to
# different processes, silently mixing models into one model's reported
# numbers.
#
# `export C3_SCORER_WORKERS=1` alone does NOT do this: configs/default.yaml's
# `serving.scorer_workers` is yaml-sourced, and ServingSettings applies yaml
# values *after* (so overriding) the env-derived ones -- confirmed
# empirically (see config/settings.py's _build_serving). Instead, point
# C3_CONFIG_PATH at a scratch copy of the tracked yaml with just that one
# field forced to 1, so this script never edits configs/default.yaml itself.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

CONFIG_OVERRIDE="$(mktemp -t conquer3-eval-config.XXXXXX.yaml)"
sed -E 's/^([[:space:]]*scorer_workers:)[[:space:]]*[0-9]+/\1 1/' \
  configs/default.yaml >"$CONFIG_OVERRIDE"
export C3_CONFIG_PATH="$CONFIG_OVERRIDE"

SCORER_PORT="${C3_SCORER_PORT:-3000}"
SCORER_LOG="$(mktemp -t conquer3-eval-scorer.XXXXXX.log)"
SCORER_PID=""

cleanup() {
  if [ -n "$SCORER_PID" ] && kill -0 "$SCORER_PID" 2>/dev/null; then
    echo "== stopping scorer (pid $SCORER_PID) =="
    kill -TERM "$SCORER_PID" 2>/dev/null || true
    wait "$SCORER_PID" 2>/dev/null || true
  fi
  rm -f "$CONFIG_OVERRIDE"
}
trap cleanup EXIT

echo "== starting conquer3 serve (single worker, log: $SCORER_LOG) =="
uv run conquer3 serve >"$SCORER_LOG" 2>&1 &
SCORER_PID=$!

echo "== waiting for http://localhost:${SCORER_PORT}/readyz =="
for _ in $(seq 1 60); do
  if curl -fsS -o /dev/null "http://localhost:${SCORER_PORT}/readyz" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$SCORER_PID" 2>/dev/null; then
    echo "scorer exited before becoming ready -- log:"
    cat "$SCORER_LOG"
    exit 1
  fi
  sleep 2
done
if ! curl -fsS -o /dev/null "http://localhost:${SCORER_PORT}/readyz" 2>/dev/null; then
  echo "scorer never became ready within timeout -- log:"
  cat "$SCORER_LOG"
  exit 1
fi

echo "== running the registry-wide evaluation suite (real MLflow registry) =="
uv run pytest tests/eval -m eval -v

OUT_DIR="${C3_EVAL_OUT_DIR:-data/eval}"
echo
echo "Reports written to: ${OUT_DIR}/all_versions_summary.json (and per-model CSV/JSON alongside it)"
