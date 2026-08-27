# conquer3

End-to-end credit-fraud detection platform on the [PaySim1](https://www.kaggle.com/datasets/ealaxi/paysim1)
dataset — a Postgres medallion warehouse, a Pathway feature engine shared by batch
and streaming paths, a BentoML scoring service, Airflow orchestration, and a Colab
training notebook, all designed around one rule: **one pure Python module computes
every feature, and every consumer (live requests, batch backfill, notebook training)
calls the same code.**

This file tracks what's actually built and how to run it, layer by layer.

## Status

Built layer by layer; each layer has a gate script under `scripts/smoke/` that must
pass before the next one starts.

| Layer | What | Status |
|---|---|---|
| 0 — Skeleton | `conquer3.core` feature engine, import boundaries, tooling | ✅ Done — `scripts/smoke/layer0_skeleton.sh` |
| 1 — Infra | Docker Compose: Postgres, Redis, OTel Collector, Airflow | ✅ Running — `scripts/smoke/layer1_infra.sh` |
| 2 — Warehouse | Postgres medallion schema (bronze/silver/gold), DuckDB+Ibis transforms | ✅ Done — `scripts/smoke/layer2_warehouse.sh` |
| 3 — Feature core | (built as part of Layer 0; Pathway wiring is Layer 3b) | ✅ Done |
| 3b — Pathway | Batch backfill + streaming state repair | ⬜ Not started |
| 4 — Model contract | MLflow publish/resolve (`contracts/model_registry.py`) | ⬜ Not started |
| 5 — Serving | BentoML service, Redis state store, event sink | ⬜ Not started |
| 6 — Airflow DAGs | Bootstrap/ingest/medallion/DQ/skew-audit/champion-watch DAGs | ⬜ Not started (only the `hello_world` smoke DAG exists) |
| 7 — Observability | Local OTel Collector wired; remote Grafana endpoints | 🟡 Collector running locally; remote endpoints not yet supplied |
| 8 — Colab notebook | Training template | ⬜ Not started |

**What works right now:** `conquer3.core` (the feature engine — 34 features, cold-start
handling, the associativity-verified state merge that keeps streaming and batch in
sync), the full local infra stack (`core` + `pipeline` Compose profiles) including
a working Airflow install that successfully parses and runs a smoke-test DAG, and the
Layer 2 warehouse -- `conquer3 ingest bronze` / `conquer3 transform bronze-to-silver` /
`conquer3 transform silver-to-gold` load a PaySim1 CSV through the full medallion
pipeline into `gold.txn_features`, computing every feature via `conquer3.core.features`
(never in SQL) so it's provably the same code path serving and Colab use.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) (Python package/venv manager)
- Docker + Docker Compose v2 (`docker compose version`)
- Python 3.12 (pinned via `.python-version`)

## Quickstart

```bash
git clone <this repo> && cd conquer3

# 1. Python env + .env (generates a real Fernet key and JWT secret into .env)
./scripts/bootstrap.sh

# 2. Core infra: Postgres, Redis, OTel Collector
docker compose --profile core up -d

# 3. Airflow (separate profile — its own metadata DB, 5 services)
docker compose --profile pipeline up -d --build

# 4. Verify everything end-to-end (see "Verifying" below)
./scripts/smoke/layer1_infra.sh
```

Airflow UI: **http://localhost:8080** — login from `_AIRFLOW_WWW_USER_USERNAME` /
`_AIRFLOW_WWW_USER_PASSWORD` in `.env` (defaults to `airflow` / `change-me`).

## Running the Python side without Docker

```bash
uv sync --all-extras   # or a subset -- see "Dependency extras" below
uv run pytest tests/
uv run conquer3 version
```

## Compose profiles

Nothing starts unless you pick a profile — there's no default "just run
`docker compose up`" here, on purpose, since most services aren't built yet.

| Profile | Services | Purpose |
|---|---|---|
| `core` | `postgres`, `redis`, `otel-collector` | Needed by everything else |
| `pipeline` | `airflow-postgres`, `airflow-{init,apiserver,scheduler,dag-processor,triggerer}` | Orchestration (own metadata DB, separate from the `postgres` warehouse) |
| `stream` | `pathway` | Feature engine (Layer 3b — builds today, nothing to run yet) |
| `serving` | `bentoml` | Scoring API (Layer 5 — builds today, nothing to serve yet) |
| `demo` | `producer` | Transaction replay driver (Layer 5) |
| `tools` | `adminer` | Postgres UI at http://localhost:8081 |

Combine profiles freely: `docker compose --profile core --profile pipeline up -d`.

MLflow and Grafana/Prometheus/Loki/Tempo are **remote** — never in this file. Point
at them via `MLFLOW_TRACKING_URI` and the `C3_PROM_*` / `C3_LOKI_*` /
`C3_OTLP_TEMPO_ENDPOINT` vars in `.env` once you have addresses.

## Verifying

```bash
# Fastest check after editing docker-compose.yaml: resolves all interpolation,
# no containers started. Catches config errors in ~1 second.
docker compose config --quiet && echo "config OK"

# Full Layer 1 gate: brings up core + pipeline, waits for health, triggers the
# hello_world DAG, and confirms it actually runs.
./scripts/smoke/layer1_infra.sh

# Layer 0 gate: lint, type-check, import boundaries, full test suite, and a
# bare-venv install proving conquer3.core has zero dependencies (the Colab path).
./scripts/smoke/layer0_skeleton.sh

# Layer 2 gate: applies the medallion DDL, then runs ingest -> bronze -> silver ->
# gold over the real PaySim1 dataset end to end. Needs Layer 1's `core` profile up.
# Fetches the CSV itself if C3_PAYSIM_CSV_PATH (.env) isn't already there --
# extracts a sibling .zip if one's present, else downloads it (PaySim1 is public,
# no Kaggle credentials needed).
./scripts/smoke/layer2_warehouse.sh
```

Ad hoc:

```bash
docker compose ps                        # health status
docker compose logs -f airflow-init       # watch db migrate + admin user creation
docker compose logs -f otel-collector
curl http://localhost:13133               # collector health_check extension
```

## Repo layout

```
src/conquer3/
├── core/          # Tier 0: the feature engine. stdlib + typing only -- this is
│                  # what ships into Google Colab via `pip install conquer3[train]`.
│                  # timeref.py, schema.py, types.py, features.py, serde.py.
├── contracts/      # events.py (JSONL scored-event layout, stdlib-only),
│                  # model_registry.py (MLflow contract -- Layer 4, not built yet)
├── config/         # settings.py -- the ONLY place env vars are read
├── telemetry/      # otel.py -- no-op unless OTEL_EXPORTER_OTLP_ENDPOINT is set
├── db/             # Postgres + DuckDB/Ibis engine, and db/ddl/*.sql -- Layer 2
├── pipelines/       # ingest/ + transforms/ (Layer 2, done); pathway/ (Layer 3b, not built yet)
├── serving/        # BentoML service -- Layer 5, not built yet
├── producer/       # transaction replay driver -- Layer 5, not built yet
└── cli.py          # `conquer3` console script; every subcommand imports lazily

airflow/
├── dags/           # bind-mounted into every airflow-* container. Currently just
│                  # dag_hello_world.py, the Layer 1 smoke-test DAG.
├── plugins/        # empty
└── logs/           # gitignored runtime output

docker/             # one Dockerfile per service family + docker/otel/*.yaml
                    # (docker/airflow.Dockerfile exists but isn't wired into
                    # docker-compose.yaml yet -- see "Known gaps" below)
scripts/
├── bootstrap.sh    # one-time setup
└── smoke/          # one gate script per layer

tests/{unit,parity,integration,contract}/
```

## Dependency extras

`conquer3`'s base install (`pip install conquer3`) has **zero dependencies** —
that's what lets Colab install just the feature engine. Everything else is an
extra:

| Extra | Pulls in | Used by |
|---|---|---|
| `train` | scikit-learn, pandas, kagglehub, mlflow | Colab notebook (Layer 8) |
| `serving` | bentoml, redis, mlflow, scikit-learn | `serving/` (Layer 5) |
| `pipeline` | ibis, duckdb, polars, psycopg | `db/`, `pipelines/` (Layer 2) |
| `stream` | pathway, redis, psycopg | `pipelines/pathway/` (Layer 3b) |
| `registry` | mlflow (full, not `-skinny`) | `contracts/model_registry.py` |

`uv sync --all-extras` installs everything for local development. Each Docker image
installs only what it needs — see `docker/*.Dockerfile`.

## Known gaps / things to watch

- **`docker/airflow.Dockerfile` isn't wired in.** The `airflow-*` services currently
  run the *stock* `apache/airflow:3.3.1` image, so `conquer3` isn't importable
  inside them yet. `dag_hello_world.py`'s `import conquer3` is inside the task body
  (not at module level), so DAG *parsing* works fine — but the task will fail if you
  actually trigger and run it. Restoring the custom build (`build: {context: .,
  dockerfile: docker/airflow.Dockerfile}` on the `x-airflow-common` anchor) is a
  Layer 6 concern, once there's real pipeline code worth installing.
- **`AIRFLOW_UID` must stay `50000`.** The base Airflow image bakes in a real
  `airflow` user at exactly that UID; `airflow-init` bypasses the image's own
  `/entrypoint` for its migrate+create-user step, so any other UID breaks Python's
  ability to find the installed `airflow` package (`ModuleNotFoundError: No module
  named 'airflow'`). Full explanation is in the comment above `user:` in
  `docker-compose.yaml`.
- **`AIRFLOW__CORE__FERNET_KEY` must be empty or a real 32-byte urlsafe-base64
  Fernet key** — never a placeholder string. `scripts/bootstrap.sh` generates a
  real one into `.env` on first run.
- Pathway (`stream` profile) and BentoML (`serving` profile) containers build and
  install cleanly today but have no entry point yet — Layers 3b and 5.
- **`.env.example` defaults to host-side values** (`POSTGRES_HOST`/`REDIS_HOST`/
  `OTEL_EXPORTER_OTLP_ENDPOINT` at `localhost`, `C3_DUCKDB_PATH`/`C3_DUCKDB_TEMP_DIR`
  under `data/`), because every `conquer3` command today runs on the host
  (`uv run conquer3 ...`), reaching Postgres/Redis/the collector via
  docker-compose's host port mappings -- `conquer3` isn't importable inside the
  airflow-* containers yet (see above). Once Layer 6 runs `conquer3` inside a
  container on the `c3net` network, *that* container's env needs the compose
  service names instead (`postgres`, `redis`, `http://otel-collector:4317`) and a
  `/duckdb`-mounted path -- don't just copy `.env`'s current values in for it.
- **`Settings`'s nested classes (`PgSettings`, `DuckSettings`, ...) each explicitly
  get `.env` threaded down to them** by `Settings.__init__` in `config/settings.py`
  -- a plain `Field(default_factory=PgSettings)` would silently construct them with
  *no* `env_file` at all (pydantic-settings doesn't cascade a parent's `env_file`
  into a nested `BaseSettings` field), so they'd only ever see real process env
  vars and quietly fall back to class defaults whenever `.env` alone was supposed
  to supply the override. See that method's docstring for the full mechanism.
  Separately, `scripts/smoke/layer2_warehouse.sh` still does
  `set -a; source .env; set +a` before running anything -- that's for the script's
  *own* bash-level use of those values (`docker compose exec`'s `$POSTGRES_USER`,
  `$C3_PAYSIM_CSV_PATH` for locating the raw CSV), unrelated to the Python fix above.
