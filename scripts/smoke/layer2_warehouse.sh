#!/usr/bin/env bash
# Layer 2 gate: the medallion schema applies to a running Postgres, and the full
# ingest -> bronze -> silver -> gold pipeline runs end to end over the real PaySim1
# dataset. Do not start Layer 3b (Pathway) until this passes.
#
# Requires: Layer 1's `core` profile (postgres) up and healthy. The PaySim1 CSV
# (C3_PAYSIM_CSV_PATH in .env) is fetched automatically if missing -- see below.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

if [ ! -f .env ]; then
  echo "FAIL: .env not found. Run scripts/bootstrap.sh first." >&2
  exit 1
fi

# `conquer3`'s nested settings classes (PgSettings, DuckSettings, ...) only read
# real process env vars, not .env's content directly -- only the top-level Settings
# class has env_file=".env", and pydantic-settings doesn't cascade that into nested
# BaseSettings fields built via default_factory. Inside docker-compose containers
# this is a non-issue (env_file: .env injects real container env vars), but this
# script runs `conquer3` on the host, so export .env's contents for real here.
set -a
# shellcheck disable=SC1091
source .env
set +a

CSV_PATH="${C3_PAYSIM_CSV_PATH:-data/raw/paysim1.csv}"
ZIP_PATH="${CSV_PATH%.csv}.zip"

if [ ! -f "$CSV_PATH" ]; then
  echo "WARN: $CSV_PATH not found." >&2

  if [ -f "$ZIP_PATH" ]; then
    echo "  Found $ZIP_PATH -- extracting..." >&2
    uv run python -c "
import sys
from conquer3.pipelines.ingest.kaggle import extract_single_csv
extract_single_csv(sys.argv[1], sys.argv[2])
" "$ZIP_PATH" "$CSV_PATH"
  fi

  if [ ! -f "$CSV_PATH" ]; then
    echo "  Downloading from Kaggle (PaySim1 is public -- no KAGGLE_USERNAME/KAGGLE_KEY needed)..." >&2
    if ! uv run conquer3 ingest download --dest "$CSV_PATH"; then
      echo "FAIL: could not obtain $CSV_PATH." >&2
      echo "  Download it manually from https://www.kaggle.com/datasets/ealaxi/paysim1" >&2
      echo "  and place the CSV (or its .zip) at $CSV_PATH (or $ZIP_PATH)." >&2
      exit 1
    fi
  fi
fi
echo "  using $CSV_PATH"

echo "== ruff / mypy / import-linter =="
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run lint-imports

echo "== db/ddl/30_gold.sql matches core.schema =="
uv run conquer3 db gen-gold-ddl --check

echo "== pytest (unit + parity + contract; integration needs Docker) =="
uv run pytest tests/unit tests/parity tests/contract
uv run pytest tests/integration -m integration

echo "== db migrate =="
uv run conquer3 db migrate

echo "== ingest bronze =="
uv run conquer3 ingest bronze --csv "$CSV_PATH"

echo "== transform bronze-to-silver =="
uv run conquer3 transform bronze-to-silver

echo "== transform silver-to-gold =="
uv run conquer3 transform silver-to-gold

echo "== consistency checks =="
row_counts="$(docker compose exec -T postgres psql -U "${POSTGRES_USER:-conquer3}" -d "${POSTGRES_DB:-conquer3}" -tA -c "
  SELECT (SELECT count(*) FROM bronze.txn_raw)
       || ',' || (SELECT count(*) FROM silver.txn)
       || ',' || (SELECT count(*) FROM gold.txn_features);
")"
bronze_n="$(echo "$row_counts" | cut -d, -f1)"
silver_n="$(echo "$row_counts" | cut -d, -f2)"
gold_n="$(echo "$row_counts" | cut -d, -f3)"
echo "  bronze.txn_raw=$bronze_n silver.txn=$silver_n gold.txn_features=$gold_n"
if [ "$bronze_n" != "$silver_n" ] || [ "$silver_n" != "$gold_n" ]; then
  echo "FAIL: row counts disagree across bronze/silver/gold" >&2
  exit 1
fi

failed_runs="$(docker compose exec -T postgres psql -U "${POSTGRES_USER:-conquer3}" -d "${POSTGRES_DB:-conquer3}" -tA -c "
  SELECT count(*) FROM ops.pipeline_runs WHERE status != 'success';
")"
if [ "$failed_runs" != "0" ]; then
  echo "FAIL: $failed_runs non-success row(s) in ops.pipeline_runs" >&2
  exit 1
fi
echo "  ops.pipeline_runs: all runs succeeded"

echo
echo "Layer 2 gate: PASS ($gold_n rows through bronze -> silver -> gold)"
