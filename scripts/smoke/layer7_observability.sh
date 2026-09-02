#!/usr/bin/env bash
# Layer 7 gate: conquer3.telemetry.otel actually reaches a real collector, and a
# broken remote config fails loudly in the *collector's* log, never the app's.
# Do not start Layer 8 (the Colab notebook) until this passes.
#
# There is no local otel-collector container (docker-compose.yaml) -- app
# processes push OTLP straight to whatever OTEL_EXPORTER_OTLP_ENDPOINT in .env
# points at, a remote LGTM stack's own collector. This gate's infra check is
# therefore against that real remote stack: it's optional (skips cleanly if
# GRAFANA_URL isn't set in .env -- lint/tests alone still run), sends one real
# probe through .env's current OTEL config, and checks Tempo/Loki/Prometheus
# directly for it. Metrics not landing in Prometheus is reported as a WARN, not a
# FAIL -- a known, external, remote-side gap (see the note in .env), not
# something broken here.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

if [ ! -f .env ]; then
  echo "FAIL: .env not found. Run scripts/bootstrap.sh first." >&2
  exit 1
fi

echo "== ruff / mypy / import-linter =="
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run lint-imports

echo
echo "== unit tests: init_telemetry no-op/enabled/idempotency, exporter protocol dispatch,"
echo "   scorer + event-sink instruments actually record (no infra) =="
uv run pytest -q tests/unit/test_otel.py tests/unit/test_telemetry_instruments.py

echo
echo "== live remote LGTM stack reachability (optional -- only runs if GRAFANA_URL"
echo "   is set in .env; this is how to verify the actual deployed stack yourself) =="
set -a
# shellcheck disable=SC1091
source .env
set +a
if [ -z "${GRAFANA_URL:-}" ]; then
  echo "  GRAFANA_URL not set in .env -- skipping (nothing remote configured yet)"
else
  check_tcp() {  # check_tcp <label> <host:port>
    local label="$1" hostport="$2" host="${2%%:*}" port="${2##*:}"
    if timeout 3 bash -c "echo > /dev/tcp/${host}/${port}" 2>/dev/null; then
      echo "  TCP ${label} (${hostport}): reachable"
    else
      echo "FAIL: TCP ${label} (${hostport}) unreachable" >&2
      exit 1
    fi
  }
  check_tcp "otel endpoint" "${OTEL_EXPORTER_OTLP_ENDPOINT#*://}"
  check_tcp "grafana" "${GRAFANA_URL#*://}"

  echo "  sending a fresh probe through .env's CURRENT OTEL config (traces+metrics+logs)"
  uv run python "$(dirname "$0")/_layer7_emit_spans.py"

  if [ -n "${C3_TEMPO_QUERY_URL:-}" ]; then
    echo "  Tempo: searching for conquer3-scorer-layer7-gate (allowing for indexing lag)"
    waited=0
    while true; do
      result="$(curl -s -G --data-urlencode 'q={}' --data-urlencode 'limit=10' \
        "${C3_TEMPO_QUERY_URL}/api/search")"
      if echo "$result" | grep -q "conquer3-scorer-layer7-gate"; then
        echo "    OK -- trace found"
        break
      fi
      waited=$((waited + 3))
      if [ "$waited" -ge 30 ]; then
        echo "FAIL: no conquer3-scorer-layer7-gate trace found in Tempo within 30s" >&2
        echo "$result" >&2
        exit 1
      fi
      sleep 3
    done
  else
    echo "  C3_TEMPO_QUERY_URL not set -- skipping Tempo check"
  fi

  if [ -n "${GRAFANA_USER:-}" ] && [ -n "${GRAFANA_PASSWORD:-}" ]; then
    echo "  Grafana auth"
    auth_code="$(curl -s -o /dev/null -w '%{http_code}' -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" \
      "${GRAFANA_URL}/api/datasources")"
    if [ "$auth_code" != "200" ]; then
      echo "FAIL: Grafana auth returned $auth_code (expected 200)" >&2
      exit 1
    fi
    echo "    OK ($auth_code)"

    echo "  Loki (via Grafana's datasource proxy -- its own port usually isn't public),"
    echo "  allowing for indexing lag:"
    waited=0
    while true; do
      now_ns=$(($(date +%s%N)))
      start_ns=$((now_ns - 300000000000))
      result="$(curl -s -G -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" \
        --data-urlencode 'query={service_name="conquer3-scorer-layer7-gate"}' \
        --data-urlencode "start=${start_ns}" --data-urlencode "end=${now_ns}" \
        "${GRAFANA_URL}/api/datasources/proxy/uid/loki/loki/api/v1/query_range")"
      if echo "$result" | grep -q "LAYER7_GATE_LOG_PROBE"; then
        echo "    OK -- log line found"
        break
      fi
      waited=$((waited + 3))
      if [ "$waited" -ge 30 ]; then
        echo "FAIL: no LAYER7_GATE_LOG_PROBE log line found in Loki within 30s" >&2
        echo "$result" >&2
        exit 1
      fi
      sleep 3
    done
  else
    echo "  GRAFANA_USER/PASSWORD not set -- skipping Loki check (needs Grafana's proxy)"
  fi

  if [ -n "${C3_PROMETHEUS_QUERY_URL:-}" ]; then
    echo "  Prometheus: checking for c3_score_latency_ms (best-effort)"
    result="$(curl -s -G --data-urlencode 'query=c3_score_latency_ms' \
      "${C3_PROMETHEUS_QUERY_URL}/api/v1/query")"
    if echo "$result" | grep -q '"result":\[\]'; then
      echo "    WARN -- metric not visible yet. Known gap, not a bug here: the remote"
      echo "    Prometheus needs --web.enable-remote-write-receiver=true (or a scrape"
      echo "    target for its collector), see the note in .env. Not failing the gate"
      echo "    on this -- it's an external dependency this repo can't fix."
    else
      echo "    OK -- metric visible"
    fi
  else
    echo "  C3_PROMETHEUS_QUERY_URL not set -- skipping Prometheus check"
  fi
fi

echo
echo "Layer 7 gate: PASS"
