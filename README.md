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
| 1 — Infra | Docker Compose: Postgres, Redis, Airflow | ✅ Running — `scripts/smoke/layer1_infra.sh` |
| 2 — Warehouse | Postgres medallion schema (bronze/silver/gold), DuckDB+Ibis transforms | ✅ Done — `scripts/smoke/layer2_warehouse.sh` |
| 3 — Feature core | (built as part of Layer 0; Pathway wiring is Layer 3b) | ✅ Done — `scripts/smoke/layer3_feature_core.sh` |
| 3b — Pathway | Batch backfill + streaming state repair | ✅ Done — `scripts/smoke/layer3b_pathway.sh` |
| 4 — Model contract | MLflow publish/resolve (`contracts/model_registry.py`) | ✅ Done — `scripts/smoke/layer4_model_registry.sh` |
| 5 — Serving | `scorer` (a BentoML service: `POST /predict`, `POST /model_info`, deprecated `POST /invocations`; OpenAPI spec at `/docs.json`), Redis state store, event sink | ✅ Done — `scripts/smoke/layer5_serving.sh`. Champion promotion restarts the server (a bounded connection-refused window, budgeted ≤25s), trading the previous SIGHUP-reload's brief overlap for a real OpenAPI spec, `/readyz`/`/livez`, and Prometheus `/metrics` |
| 6 — Airflow DAGs | Bootstrap/ingest/medallion/DQ/skew-audit/champion-watch DAGs | ⬜ Not started (only the `hello_world` smoke DAG exists) |
| 7 — Observability | `telemetry/otel.py` (traces+metrics+logs), redis-get→predict→redis-set→file-append spans, `c3_*` metrics — pushed straight to a remote LGTM stack's own collector, no local collector container | 🟡 Traces and logs confirmed landing in Tempo/Loki end to end; metrics have nowhere to land until the remote Prometheus enables its remote-write receiver (or gets a scrape target) — see `.env`'s observability section — `scripts/smoke/layer7_observability.sh` |
| 8 — Colab notebook | Training template | ⬜ Not started |

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) (Python package/venv manager)
- Docker + Docker Compose v2 (`docker compose version`)
- `make` (Linux/macOS/WSL) or PowerShell (Windows) -- the `Makefile`/`make.ps1`
  pair at the repo root is the supported way to bring services up; see
  "Orchestrating services" below
- Python 3.12 (pinned via `.python-version`)

## Quickstart

```bash
git clone <this repo> && cd conquer3

# 1. Python env + .env (generates a real Fernet key and JWT secret into .env)
./scripts/bootstrap.sh

# 2. Every service, group by group, waiting for each to actually report
#    healthy before moving on -- see "Orchestrating services" below
make up            # Linux/macOS/WSL
.\make.ps1 up        # Windows

# 3. Verify everything end-to-end (see "Verifying" below)
./scripts/smoke/layer1_infra.sh
```

`make up` brings up `airflow` -> `stream` -> `core` in order (each is safe to
start with nothing else configured); `core` (the `scorer` + `ui` services)
only actually starts if `.env` already has a real `MLFLOW_TRACKING_URI` --
bringing `scorer` up without one would just crash-loop, since it resolves a
champion at boot and refuses to guess. Bring it up later with `make core` once
you have one (see "Running the scorer" below). Re-running `make up` is safe --
already-healthy services are left alone.

Airflow UI: **http://localhost:8080** — login from `_AIRFLOW_WWW_USER_USERNAME` /
`_AIRFLOW_WWW_USER_PASSWORD` in `.env` (defaults to `airflow` / `change-me`).

## Running the Python side without Docker

```bash
uv sync --all-extras   # or a subset -- see "Dependency extras" below
uv run pytest tests/
uv run conquer3 version
```

## Orchestrating services

Nothing starts unless you pick a group -- there's no default "just run `make
up`" that unconditionally starts everything, since `core` needs a real
`MLFLOW_TRACKING_URI` first (see "Quickstart" above).
`Makefile` (Linux/macOS/WSL) and `make.ps1` (Windows) wrap
`docker-compose.yaml`'s profiles into three groups, matching how their
`depends_on` chains actually resolve:

| Group (`make <name>`) | Compose profile(s) | Services | Purpose |
|---|---|---|---|
| `core` | `serving`, `ui` | `scorer`, `ui` | Scoring API (Layer 5) + Streamlit console (Layer 9). Bundled together because `ui` depends_on `scorer`, and Compose only resolves a profile-scoped `depends_on` when both profiles are active in the same command |
| `stream` | `stream` | `pathway` | Feature engine (Layer 3b -- static backfill + streaming state repair) |
| `airflow` | `pipeline` | `airflow-postgres`, `airflow-{init,apiserver,scheduler,dag-processor,triggerer}` | Orchestration (own metadata DB, unrelated to the warehouse Postgres) |

Each group also has `-down`, `-logs`, `-ps`, `-restart`, and `-build` variants
(e.g. `make core-logs`, `make airflow-build`), plus combined `make up` /
`down` / `ps` / `logs` / `restart` / `clean` / `build` across all three -- run
`make help` (or `.\make.ps1 help`) for the full list. `logs` (and its
per-group variants) takes `SERVICE=<name>` (`-Service <name>` on Windows) to
scope to one service, e.g. `make logs SERVICE=scorer`.

Postgres and Redis are **external, managed services** (Neon Postgres, a
managed Redis) -- there's no local `postgres`/`redis` container in
`docker-compose.yaml` to bring up; point `POSTGRES_*`/`REDIS_*` in `.env` at
them instead. (`compose.parity.yaml`, used only by the standalone MLflow demo
below, is a separate, fully local-native file with its own
`postgres`/`redis`/`mlflow` containers under its own `core` profile -- don't
confuse the two.)

`demo` (`producer`, a one-shot transaction replay driver, not built yet) has
no Makefile group by design -- it's an explicit one-off you run once `scorer`
is up, not standing infrastructure:
```bash
docker compose --profile demo up -d --build
```

Streamlit has no auth and no multi-tenant isolation -- fine for local/demo
use, not for exposing this port publicly.

MLflow and Grafana/Prometheus/Loki/Tempo are **remote** -- never in this file.
Point at them via `MLFLOW_TRACKING_URI` and `OTEL_EXPORTER_OTLP_ENDPOINT` in
`.env` once you have addresses -- app code talks OTLP straight to the remote
stack's own collector (no local collector container), which does the fan-out
into its co-located Prometheus/Tempo/Loki. Verify reachability yourself with
`scripts/smoke/layer7_observability.sh`.

## Verifying

```bash
# Fastest check after editing docker-compose.yaml: resolves all interpolation,
# no containers started. Catches config errors in ~1 second.
docker compose config --quiet && echo "config OK"

# Full Layer 1 gate: brings up local Postgres/Redis + the airflow pipeline,
# waits for health, triggers the hello_world DAG, and confirms it actually runs.
./scripts/smoke/layer1_infra.sh

# Layer 0 gate: lint, type-check, import boundaries, full test suite, and a
# bare-venv install proving conquer3.core has zero dependencies (the Colab path).
./scripts/smoke/layer0_skeleton.sh

# Layer 2 gate: applies the medallion DDL, then runs ingest -> bronze -> silver ->
# gold over the real PaySim1 dataset end to end. Needs a reachable Postgres per
# .env's POSTGRES_* (managed, e.g. Neon, or a local instance).
# Fetches the CSV itself if C3_PAYSIM_CSV_PATH (.env) isn't already there --
# extracts a sibling .zip if one's present, else downloads it (PaySim1 is public,
# no Kaggle credentials needed).
./scripts/smoke/layer2_warehouse.sh

# Layer 3 gate: golden vectors, cold-start policy, the Hypothesis
# merge-associativity property test, and an exact-equality 100k-row Python/DuckDB
# parity sweep for derive_event_ts_us. No Docker needed -- conquer3.core is
# dependency-light by construction.
./scripts/smoke/layer3_feature_core.sh

# Layer 3b gate: static backfill row-count parity, licensed-connector-vs-psycopg-
# fallback output parity, streaming pickup latency (< 2s), and kill/restart
# deduplication -- all via pytest + ephemeral testcontainers Postgres/Redis, so it
# never touches (or needs) the real docker-compose stack.
./scripts/smoke/layer3b_pathway.sh

# Layer 4 gate: publish registers a version with tags + a logged signature,
# pyfunc.load_model downloads and predicts, re-aliasing makes the resolver follow,
# a dead MLFLOW_TRACKING_URI falls back to the cached champion and flips the
# c3_model_resolution_degraded gauge, and conda.yaml names mlflow, never
# mlflow-skinny -- all via pytest against an ephemeral local `mlflow server`
# (sqlite backend, no Docker needed at all). Verified end to end (including
# against a real remote MLflow deployment, not just the ephemeral local one).
./scripts/smoke/layer4_model_registry.sh

# Layer 5 gate: scorer IS the inference endpoint, proven against a real
# ephemeral local mlflow server + a real ephemeral Redis (testcontainers) + the
# actual mlflow.pyfunc.scoring_server subprocess -- back-to-back and batched
# same-account calls carry state, dry_run leaves Redis/events untouched,
# op=model_info reports the resolved version, concurrent same-account requests
# never corrupt state, a champion promotion reloads within one poll interval
# with zero non-2xx responses, a boot against dead MLflow falls back to the
# cached champion (degraded=true on every response), and -- the property the
# whole architecture is built around -- killing remote MLflow entirely after
# boot leaves /invocations serving at full correctness. Needs Docker only for
# the ephemeral Redis container.
./scripts/smoke/layer5_serving.sh
```

To check a *real* `MLFLOW_TRACKING_URI` (not the gate's ephemeral local server) once
`.env` points at one:

```bash
uv run conquer3 model publish-dummy --alias-champion   # register + alias a throwaway model
uv run conquer3 model resolve-champion                 # resolve it back, confirm degraded=False
uv run pytest -s tests/unit/test_model_registry.py::test_log_current_champion_from_registry
#   ^ prints the live registry's current champion (name/version/run_id/tags) --
#     diagnostic only, no assertions, and never downloads the artifact (see
#     "Known gaps" below for why that matters).
```

## Running the scorer

`scorer` needs a champion to resolve at boot, so publish one first (a throwaway
`DummyClassifier` is fine for a smoke run -- swap in a real Colab-trained model
once Layer 8 exists). The service is a BentoML app: a full OpenAPI 3 document is
served at `/docs.json`, with an interactive Swagger UI at `/` -- open it in a
browser for a live, always-current description of every route instead of the
prose below.

**Without Docker** (needs `compose.parity.yaml`'s `core` profile's Redis up -- a
different, local-native `core` from the Makefile's above, see "Orchestrating
services"; plus a real `MLFLOW_TRACKING_URI` in `.env`, or `mlflow server` in a
second terminal for a fully local demo -- `export` it rather than editing
`.env`, since a real env var wins over `.env`'s value without permanently
changing it):

```bash
docker compose -f compose.parity.yaml --profile core up -d redis   # or: redis-server, if you have it locally
uv run mlflow server --host 127.0.0.1 --port 5000 &       # a throwaway local registry, for a demo
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000          # overrides .env for this shell only

uv run conquer3 model publish-dummy --alias-champion       # register + alias a throwaway champion
uv run conquer3 serve                                      # resolves it, boots on :3000
```

In another terminal:

```bash
curl http://localhost:3000/readyz                          # 500 until the champion is loaded

curl -X POST http://localhost:3000/predict \
  -H 'Content-Type: application/json' \
  -d '{"transactions": [{"event_id": "e1", "account_id": "C1", "dest_id": "M900",
        "txn_type": "TRANSFER", "amount": 181.0, "oldbalance_org": 181.0,
        "newbalance_orig": 0.0, "oldbalance_dest": 0.0, "newbalance_dest": 181.0,
        "event_ts_us": 1700000000000000, "step": 1}]}'
#   -> [{"event_id": "e1", "fraud_score": ..., "decision": "...",
#        "had_prev_state": false, "seconds_since_last_txn": null,
#        "model_version": "1", "feature_schema_version": 1, "degraded": false}]

# Send it again with a later event_ts_us for the same account_id: had_prev_state
# flips to true and seconds_since_last_txn is no longer null -- state round-
# tripped through Redis between the two calls.

curl -X POST http://localhost:3000/model_info -H 'Content-Type: application/json' -d '{}'
#   -> the resolved ModelRef (name/version/run_id/alias/degraded), typed and
#      requiring no body -- unlike the old op=model_info, which needed a
#      syntactically valid placeholder transaction row.
```

A deprecated `POST /invocations` route accepts the previous MLflow envelope
(`dataframe_records` / `params: {op, dry_run}` -> `{"predictions": [...]}`) for
callers not yet migrated; it is a thin adapter over `/predict` and
`/model_info`, so it can never disagree with them, and `/docs.json` marks it
deprecated.

Promote a new champion in another terminal (`conquer3 model publish-dummy
--alias-champion` again) and `scorer` picks it up within `C3_CHAMPION_POLL_S`
(default 300s locally; lower it in `.env` for a faster demo) -- `/model_info`
will report the new version. Unlike the previous MLflow/SIGHUP implementation,
the switch is a process restart: in-flight requests drain cleanly and no request
ever gets an error response, but there is a brief window (budgeted at 25s in the
Layer 5 gate, typically 1-3s in practice) where new connections are refused
while the replacement boots and reloads the model.

### MLflow local service (Docker-based)

`compose.parity.yaml` includes a standalone MLflow service (`mlflow` on `serving`
profile) that auto-imports the champion from `.model_artifacts/paysim/champion`
at startup. Serves registry + proxied artifacts on port 5000.

**Networking & host header validation:** Both `mlflow` and `scorer` run on the
`c3net` Docker network. Scorer reaches MLflow via service name:
`MLFLOW_TRACKING_URI=http://mlflow:5000`. MLflow's FastAPI validates incoming
`Host` headers against an allowlist to prevent DNS rebinding attacks; the
included `docker/mlflow-standalone/entrypoint.sh` sets `--allowed-hosts "*"`
for local dev (sufficient for service-name connectivity). Production
deployments should restrict this to specific hostnames.

**With Docker** (`compose.parity.yaml`'s `serving` profile, which includes this
`mlflow` service; needs its `core` profile up first, and `MLFLOW_TRACKING_URI`
in `.env` reachable *from inside the container* -- `localhost` won't resolve to
your host from in there; `compose.parity.yaml` maps `host.docker.internal` to
the host gateway for you (`extra_hosts`), so `http://host.docker.internal:<port>`
reaches a server on the host on every platform, not just Docker Desktop; a real
remote address needs no such thing):

```bash
docker compose -f compose.parity.yaml --profile core --profile serving up -d --build
docker compose -f compose.parity.yaml ps scorer             # healthy once /readyz responds
curl http://localhost:${C3_SCORER_PORT:-3000}/readyz
```

### Transaction input/output schema

The definitive schema is served live at `/docs.json` (OpenAPI 3) and rendered as
Swagger UI at `/` -- both generated from the same pydantic models the routes
validate against (`src/conquer3/serving/api_models.py`), so they cannot drift
from what the server actually accepts. The block below is a snapshot for quick
reference.

`POST /predict` accepts `transactions` with 11 required fields per row, plus an
optional `dry_run`:

```json
{
  "transactions": [
    {
      "event_id": "string",
      "account_id": "string",
      "dest_id": "string",
      "txn_type": "string (TRANSFER|PAYMENT|CASH_OUT|DEBIT|CASH_IN)",
      "amount": "float",
      "oldbalance_org": "float",
      "newbalance_orig": "float",
      "oldbalance_dest": "float",
      "newbalance_dest": "float",
      "event_ts_us": "int (Unix timestamp in microseconds)",
      "step": "int (simulation step)"
    }
  ],
  "dry_run": false
}
```

Response (a JSON array, one row per input transaction, same order):

```json
[
  {
    "event_id": "string",
    "fraud_score": "float (0.0-1.0)",
    "decision": "FRAUD|LEGIT",
    "had_prev_state": "bool",
    "seconds_since_last_txn": "float | null",
    "model_version": "string",
    "feature_schema_version": "int",
    "degraded": "bool"
  }
]
```

`POST /model_info` takes no body and returns the resolved `ModelRef`
(`name`/`version`/`run_id`/`alias`/`tags`/`degraded`) as one object.

`POST /invocations` (deprecated) reproduces the previous MLflow envelope --
`{"dataframe_records": [...], "params": {"op": "score"|"model_info", "dry_run": ...}}`
in, `{"predictions": [...]}` out -- for callers not yet migrated to `/predict`.

**State tracking:** Scores for the same `account_id` called twice return updated
state (e.g., `had_prev_state=true`, `seconds_since_last_txn` populated). State
is persisted in Redis and survives scorer restarts via `C3_STATE_TTL_DAYS`
(default 90 days). `dry_run=true` skips Redis updates and event writes.

Two things confirmed empirically if you point this at a throwaway local
`mlflow server` (as in the no-Docker demo above) instead of a real deployment:

- **MLflow's own DNS-rebinding protection rejects `host.docker.internal` by
  default.** `mlflow server` (`mlflow>=3.x`) validates the incoming `Host`
  header against an allowlist that covers `localhost`/private IPs but not
  hostnames like `host.docker.internal`; without an override it answers `403
  Invalid Host header - possible DNS rebinding attack detected` to every
  request from the container. Start the demo server with
  `--allowed-hosts 'host.docker.internal:*'` (or `MLFLOW_SERVER_ALLOWED_HOSTS`)
  to let it through. A real deployment reached by its real address doesn't hit
  this at all.
- **A `mlflow server` with a local-filesystem `--default-artifact-root` only
  serves artifacts to clients that share that filesystem.** It's a fine
  shortcut for the no-Docker demo above (host-side `conquer3` and the host-side
  `mlflow server` share one disk), but the client-side artifact repo silently
  resolves to a local path the *container* can't see, so `resolve_champion`
  fails there. Point at a real deployment (with real remote artifact storage)
  to exercise the Docker profile end to end -- this is specifically a
  same-machine-only shortcut in the demo server, not a `scorer` limitation.

Ad hoc:

```bash
make ps                                       # health status across every group
make airflow-logs SERVICE=airflow-init        # watch db migrate + admin user creation
bash scripts/smoke/layer7_observability.sh    # verify the remote LGTM stack itself
```

## Repo layout

```
src/conquer3/
├── core/          # Tier 0: the feature engine. stdlib + typing only -- this is
│                  # what ships into Google Colab via `pip install conquer3[train]`.
│                  # timeref.py, schema.py, types.py, features.py, serde.py.
├── contracts/      # events.py (JSONL scored-event layout, stdlib-only),
│                  # model_registry.py (MLflow contract -- Layer 4, done)
├── config/         # settings.py -- the ONLY place env vars are read
├── telemetry/      # otel.py -- no-op unless OTEL_EXPORTER_OTLP_ENDPOINT is set
├── db/             # Postgres + DuckDB/Ibis engine, and db/ddl/*.sql -- Layer 2
├── redis_scripts/  # monotonic_cas.lua -- shared by Pathway (Layer 3b) and the
│                  # scorer (Layer 5), so both write through the exact same CAS
│                  # semantics by construction
├── pipelines/       # ingest/ + transforms/ (Layer 2 + export_staging.py for
│                  # Layer 3b, done); pathway/ (Layer 3b, done)
├── serving/        # scorer -- Layer 5, done, a BentoML service. scorer.py
│                  # (FraudScorer: framework-free feature computation + Redis +
│                  # event sink), api_models.py (pydantic request/response models
│                  # generated from TransactionEvent -- the OpenAPI 3 spec at
│                  # /docs.json comes from these), service.py (FraudScorerService:
│                  # POST /predict, POST /model_info, deprecated POST
│                  # /invocations), champion.py (supervisor: resolve + pin a
│                  # version; workers: load from the local cache only),
│                  # supervisor.py (`conquer3 serve`: launches `bentoml serve`,
│                  # polls for champion changes, restarts on a version change),
│                  # state_store.py, event_sink.py
├── producer/       # transaction replay driver -- not built yet
├── ui/             # Streamlit console -- Layer 9. app.py (entrypoint: sidebar +
│                  # tabs), scorer_client.py (HTTP client -- never imports
│                  # conquer3.serving), history.py (reads /events JSONL),
│                  # labels.py (ops.prediction_labels), inference.py,
│                  # inspection.py. A client of `scorer`; holds no model.
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
├── bootstrap.sh    # one-time setup: uv sync, generate .env
├── startup.sh      # bring the whole docker-compose stack up, profile by
│                  # profile, waiting for each service to report healthy
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
| `serving` | bentoml, mlflow (via `registry`), redis, scikit-learn | `serving/` (Layer 5, done) |
| `pipeline` | ibis, duckdb, polars, psycopg2 | `db/`, `pipelines/` (Layer 2) |
| `stream` | pathway, redis, psycopg2 | `pipelines/pathway/` (Layer 3b, done) |
| `registry` | mlflow (full, not `-skinny`) | `contracts/model_registry.py` |

`bentoml` owns the web stack (its own starlette/uvicorn, not pinned here) and,
more to the point, the OpenAPI document it derives from `serving/api_models.py`'s
pydantic models. `mlflow` is still pulled in via `registry`, but only to resolve
and load the champion artifact -- nothing in a worker process serves HTTP
through it.

`uv sync --all-extras` installs everything for local development. Each Docker image
installs only what it needs — see `docker/*.Dockerfile`.

