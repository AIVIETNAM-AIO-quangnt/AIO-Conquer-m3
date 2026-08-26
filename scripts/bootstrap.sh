#!/usr/bin/env bash
# One-time local setup: python deps, .env, and the secrets Airflow 3 requires to
# start (a blank AIRFLOW__CORE__FERNET_KEY / AIRFLOW__API_AUTH__JWT_SECRET works,
# but a generated one is what you actually want even for local dev).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "== uv sync =="
uv sync

if [ ! -f .env ]; then
  echo "== creating .env from .env.example =="
  cp .env.example .env

  FERNET_KEY="$(python3 -c 'import base64, os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())')"
  JWT_SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"

  # BSD and GNU sed both accept -i '' vs -i differently; use a portable perl instead.
  perl -pi -e "s#^AIRFLOW__CORE__FERNET_KEY=\$#AIRFLOW__CORE__FERNET_KEY=${FERNET_KEY}#" .env
  perl -pi -e "s#^AIRFLOW__API_AUTH__JWT_SECRET=\$#AIRFLOW__API_AUTH__JWT_SECRET=${JWT_SECRET}#" .env

  echo "Generated AIRFLOW__CORE__FERNET_KEY and AIRFLOW__API_AUTH__JWT_SECRET in .env."
  echo "Fill in MLFLOW_TRACKING_URI, PATHWAY_LICENSE_KEY, and the remote OTel/Grafana"
  echo "endpoints in .env before starting the pipeline/serving/observability profiles."
else
  echo "== .env already exists, leaving it alone =="
fi

echo "== starting core infra (postgres, redis, otel-collector) =="
docker compose --profile core up -d

echo
echo "Bootstrap complete. Next:"
echo "  scripts/smoke/layer1_infra.sh   # verify core + pipeline infra is healthy"
