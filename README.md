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
| 3 — Feature core | (built as part of Layer 0; Pathway wiring is Layer 3b) | ✅ Done — `scripts/smoke/layer3_feature_core.sh` |
| 3b — Pathway | Batch backfill + streaming state repair | ✅ Done — `scripts/smoke/layer3b_pathway.sh` |
| 4 — Model contract | MLflow publish/resolve (`contracts/model_registry.py`) | ✅ Done — `scripts/smoke/layer4_model_registry.sh` |
| 5 — Serving | `scorer` (MLflow's own scoring server as a library), Redis state store, event sink | ✅ Done — `scripts/smoke/layer5_serving.sh` |
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
(never in SQL) so it's provably the same code path serving and Colab use. The
Pathway feature engine (`conquer3 transform export-staging` / `conquer3 pathway
backfill` / `conquer3 pathway streaming`) folds the same `TransactionEvent` stream
into per-account state via a custom reducer that delegates to `core.features.
advance_state`/`merge_states`, mirroring it to Redis (through a shared monotonic-CAS
Lua script) and to `gold.account_state`, via either the licensed `pw.io.postgres.write`
connector or a CAS-guarded psycopg fallback -- both proven to agree exactly. The
model contract (`conquer3 model publish-dummy` / `conquer3 model resolve-champion`)
publishes a signed, tagged model version to MLflow and resolves the "champion"
alias back to a concrete version, with a champion cache (JSON ref + downloaded
artifact) that lets resolution degrade gracefully -- and quickly -- to the last
known-good model if the tracking server is unreachable at boot. `conquer3 serve`
(the `scorer` service, Layer 5) **is the inference endpoint** -- it resolves the
champion once at boot, builds a local `mlflow.pyfunc` wrapper around it that owns
feature computation (via `conquer3.core.features`, the same code Colab and batch
use), the Redis read-modify-write, and the JSONL event sink, then serves
`POST /invocations` entirely from local files, local Redis, and local CPU.
Remote MLflow is storage and a logbook only -- never an inference backend -- and
`scorer` is not a gateway or proxy in front of it: killing remote MLflow
entirely leaves `/invocations` serving at full correctness, and only *new*
champion promotions stop arriving (picked back up automatically once MLflow
returns, via a background poll every `C3_CHAMPION_POLL_S`).

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) (Python package/venv manager)
- Docker + Docker Compose v2 (`docker compose version`)
- Python 3.12 (pinned via `.python-version`)

## Quickstart

```bash
git clone <this repo> && cd conquer3

# 1. Python env + .env (generates a real Fernet key and JWT secret into .env)
./scripts/bootstrap.sh

# 2. The whole docker-compose stack, profile by profile, waiting for each
#    service to actually report healthy before moving on
./scripts/startup.sh

# 3. Verify everything end-to-end (see "Verifying" below)
./scripts/smoke/layer1_infra.sh
```

`scripts/startup.sh` brings up `core` → `pipeline` → `stream` in order (each is
safe to start with nothing else configured); `serving` only if `.env` already
has a real `MLFLOW_TRACKING_URI` — bringing `scorer` up without one would just
crash-loop, since it resolves a champion at boot and refuses to guess. Bring it
up later with `docker compose --profile serving up -d --build` once you have
one (see "Running the scorer" below). Re-running `startup.sh` is safe --
already-healthy services are left alone.

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
| `stream` | `pathway` | Feature engine (Layer 3b — static backfill + streaming state repair) |
| `serving` | `scorer` | Scoring API (Layer 5 — the inference endpoint; see "Running the scorer" below) |
| `demo` | `producer` | Transaction replay driver (`producer/replay.py`, not built yet) |
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
once Layer 8 exists). `C3_SCORER_WORKERS` must stay `>= 2`: with exactly one
worker, uvicorn installs no `SIGHUP` handler at all, so a champion promotion
would kill the process instead of reloading it (`conquer3 serve` refuses to
start below 2, so a misconfigured `.env` fails loudly at boot, not silently on
the first promotion).

**Without Docker** (needs `core` profile's Redis up; a real `MLFLOW_TRACKING_URI`
in `.env`, or `mlflow server` in a second terminal for a fully local demo --
`export` it rather than editing `.env`, since a real env var wins over `.env`'s
value without permanently changing it):

```bash
docker compose --profile core up -d redis                # or: redis-server, if you have it locally
uv run mlflow server --host 127.0.0.1 --port 5000 &       # a throwaway local registry, for a demo
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000          # overrides .env for this shell only

uv run conquer3 model publish-dummy --alias-champion       # register + alias a throwaway champion
uv run conquer3 serve                                      # resolves it, boots on :3000
```

In another terminal:

```bash
curl http://localhost:3000/ping                            # one of MLflow's four fixed routes

curl -X POST http://localhost:3000/invocations \
  -H 'Content-Type: application/json' \
  -d '{"dataframe_records": [{"event_id": "e1", "account_id": "C1", "dest_id": "M900",
        "txn_type": "TRANSFER", "amount": 181.0, "oldbalance_org": 181.0,
        "newbalance_orig": 0.0, "oldbalance_dest": 0.0, "newbalance_dest": 181.0,
        "event_ts_us": 1700000000000000, "step": 1}]}'
#   -> {"predictions": [{"event_id": "e1", "fraud_score": ..., "decision": "...",
#        "had_prev_state": false, "seconds_since_last_txn": null,
#        "model_version": "1", "feature_schema_version": 1, "degraded": false}]}

# Send it again with a later event_ts_us for the same account_id: had_prev_state
# flips to true and seconds_since_last_txn is no longer null -- state round-
# tripped through Redis between the two calls.

curl -X POST http://localhost:3000/invocations \
  -H 'Content-Type: application/json' \
  -d '{"dataframe_records": [{"event_id": "_", "account_id": "_", "dest_id": "_",
        "txn_type": "TRANSFER", "amount": 0, "oldbalance_org": 0, "newbalance_orig": 0,
        "oldbalance_dest": 0, "newbalance_dest": 0, "event_ts_us": 0, "step": 0}],
      "params": {"op": "model_info"}}'
#   -> the resolved ModelRef (name/version/run_id/alias/degraded) as one row.
#      The row above is ignored by op=model_info but still required -- MLflow
#      enforces the input schema before predict() ever runs, regardless of op.
```

Promote a new champion in another terminal (`conquer3 model publish-dummy
--alias-champion` again) and `scorer` picks it up within `C3_CHAMPION_POLL_S`
(default 300s locally; lower it in `.env` for a faster demo) -- `op=model_info`
will report the new version, with zero downtime across the switch.

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

**With Docker** (`serving` profile; needs `core` up first, and `MLFLOW_TRACKING_URI`
in `.env` reachable *from inside the container* -- `localhost` won't resolve to
your host from in there; `docker-compose.yaml` maps `host.docker.internal` to
the host gateway for you (`extra_hosts`), so `http://host.docker.internal:<port>`
reaches a server on the host on every platform, not just Docker Desktop; a real
remote address needs no such thing):

```bash
docker compose --profile core --profile serving up -d --build
docker compose ps scorer                                   # healthy once /ping responds
curl http://localhost:${C3_SCORER_PORT:-3000}/ping
```

### Transaction input/output schema

`POST /invocations` accepts `dataframe_records` with 11 required fields:

```json
{
  "dataframe_records": [
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
  "params": {
    "op": "score|model_info",
    "dry_run": true|false
  }
}
```

Response:

```json
{
  "predictions": [
    {
      "event_id": "string",
      "fraud_score": "float (0.0-1.0)",
      "decision": "fraud|legitimate",
      "had_prev_state": "bool",
      "seconds_since_last_txn": "float | null",
      "model_version": "string",
      "feature_schema_version": "int",
      "degraded": "bool"
    }
  ]
}
```

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
│                  # model_registry.py (MLflow contract -- Layer 4, done)
├── config/         # settings.py -- the ONLY place env vars are read
├── telemetry/      # otel.py -- no-op unless OTEL_EXPORTER_OTLP_ENDPOINT is set
├── db/             # Postgres + DuckDB/Ibis engine, and db/ddl/*.sql -- Layer 2
├── redis_scripts/  # monotonic_cas.lua -- shared by Pathway (Layer 3b) and the
│                  # scorer (Layer 5), so both write through the exact same CAS
│                  # semantics by construction
├── pipelines/       # ingest/ + transforms/ (Layer 2 + export_staging.py for
│                  # Layer 3b, done); pathway/ (Layer 3b, done)
├── serving/        # scorer -- Layer 5, done. pyfunc_model.py (FraudScorerModel:
│                  # feature computation + Redis + event sink), signature.py
│                  # (ModelSignature generated from TransactionEvent), build.py
│                  # (resolve champion -> local wrapper -> symlink swap),
│                  # supervisor.py (`conquer3 serve`: launches MLflow's own
│                  # scoring server, polls for champion changes, SIGHUPs to
│                  # reload), state_store.py, event_sink.py
├── producer/       # transaction replay driver -- not built yet
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
| `serving` | mlflow, fastapi, uvicorn, redis, scikit-learn | `serving/` (Layer 5, done) |
| `pipeline` | ibis, duckdb, polars, psycopg | `db/`, `pipelines/` (Layer 2) |
| `stream` | pathway, redis, psycopg | `pipelines/pathway/` (Layer 3b, done) |
| `registry` | mlflow (full, not `-skinny`) | `contracts/model_registry.py` |

`fastapi`/`uvicorn` are listed explicitly, not inherited from `mlflow` itself:
confirmed by reading `mlflow.pyfunc.scoring_server`'s source that it imports
them lazily, inside `scoring_server.init()`, and the base `mlflow` package does
not declare them as dependencies at all -- only mlflow's own `gateway`/`genai`
extras do (and those pull in unrelated things like `boto3`/`tiktoken`). `scorer`
depends on exactly what its own scoring server needs, directly.

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
- **`C3_SCORER_WORKERS` must be `>= 2`** — `conquer3 serve` refuses to start
  below that. Confirmed by reading `uvicorn.main` (`uvicorn==0.52.4`): the
  `SIGHUP`-handling `Multiprocess` supervisor is only installed when
  `workers > 1`; with exactly one worker, `SIGHUP` falls through to the OS
  default (terminate) instead of triggering a reload. `C3_SCORER_WORKERS=1`
  does not "skip reloads" — it kills the whole scorer on the first champion
  promotion.
- **`restart_all()`'s reload is best-effort and gives the supervisor no
  callback.** Confirmed empirically (an aborted restart under host load, mid
  test suite: `ERROR: New child process was not ready in time; keeping worker
  and aborting the restart`, uvicorn's own safe fallback) — `serving/supervisor.py`
  does not trust "SIGHUP sent" as "reload succeeded"; it probes the running
  server for the version it actually reports and only advances its own
  tracked state (and only records the deployment) once that's confirmed, so
  an aborted restart is retried on the *next* poll tick instead of silently
  stalling forever on the old version.
- Pathway (`stream` profile) is wired up (Layer 3b). `scorer` (`serving`
  profile) is wired up and has a real entry point (Layer 5, `conquer3 serve`).
- **`gold.account_state.state_json` is `TEXT`, not `JSONB`**, even though it holds
  serialized JSON. Confirmed empirically: the licensed `pw.io.postgres.write`
  connector's Rust driver refuses to write a Pathway `str` column into a `JSONB`
  destination ("declared Pathway type 'str' is not compatible with PostgreSQL type
  'jsonb'"), and there's no way to add an explicit cast through that connector's
  API. `TEXT` is what both the licensed connector and the psycopg fallback agree
  on. Costs native `->>'field'` querying; nothing reads this column except
  `core.serde.state_from_json`.
- **`pw.io.postgres.write` is not actually license-gated** in the installed
  `pathway==0.32.1` (confirmed by reading its source: the `_check_entitlements`
  call lives only in `read()`'s CDC/replication path). The `auto`/`licensed`/
  `psycopg` picker in `pipelines/pathway/sinks/postgres_sink.py` still exists and
  is still tested both ways — it's what proves the Rust snapshot-writer and the
  psycopg fallback agree exactly, and it guards against a future Pathway version
  re-adding write-side gating.
- **`MLFLOW_TRACKING_URI` is still an empty placeholder in `.env.example`.** Layer 4's
  own gate never needs one (an ephemeral local `mlflow server` per test run), but
  `conquer3 model publish-dummy`/`resolve-champion` against a real deployment need
  a real remote address supplied once available -- an "Open item" carried from the
  architecture plan, not something this layer could resolve itself.
- **MLflow's host header validation blocks container-to-container requests by default.**
  `mlflow server` (`mlflow>=3.x`) validates incoming `Host` headers against an
  allowlist covering `localhost` and private IPs but rejecting arbitrary hostnames
  and container service names. **For DevOps:** Start the MLflow server with
  `--allowed-hosts '*'` (permissive, local dev only), `--allowed-hosts 'mlflow,mlflow:5000'`
  (allow service-name connectivity), or your production hostname
  `--allowed-hosts 'mlflow.company.com'`. Environment variable alternative:
  `MLFLOW_SERVER_ALLOWED_HOSTS='mlflow.company.com'`. A client connecting from inside
  a Docker container via the service name (e.g., `http://mlflow:5000`) will be rejected
  with `403 Invalid Host header - possible DNS rebinding attack detected` unless explicitly
  allowed. This is **not** a conquer3 issue — it's MLflow's own security feature that
  requires configuration when the server is reached via non-standard hostnames.
- **Some MLflow deployments don't correctly support the HTTP Range requests
  `resolve_champion`'s artifact download relies on.** Confirmed against a real
  remote server whose reverse proxy silently ignores `Range` headers (returns `200`
  with the full body instead of `206` with the requested slice) -- MLflow's chunked
  downloader reads that as a failed chunk and retries forever. The registry API
  call and the artifact chunk-download have *separate* MLflow-side timeout knobs
  (`MLFLOW_HTTP_REQUEST_TIMEOUT` vs. `MLFLOW_DOWNLOAD_CHUNK_TIMEOUT`, the latter
  defaulting to 300s); `resolve_champion` now bounds both via
  `ModelSettings.resolve_timeout_s`, so a broken/slow artifact store degrades to
  the cached champion in seconds instead of blocking boot for minutes. A
  connection that trickles bytes just fast enough to keep each individual socket
  read from ever timing out can still outlast any client-side bound, though --
  that class of failure needs a server-side fix (check Range-header handling in
  whatever sits in front of the tracking server).
- **`.env.example` defaults to host-side values** (`POSTGRES_HOST`/`REDIS_HOST`/
  `OTEL_EXPORTER_OTLP_ENDPOINT` at `localhost`, `C3_DUCKDB_PATH`/`C3_DUCKDB_TEMP_DIR`/
  `C3_MODEL_CACHE_DIR`/`C3_MODEL_CHAMPION_CACHE_FILE` under `data/`), because every
  `conquer3` command today runs on the host
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
