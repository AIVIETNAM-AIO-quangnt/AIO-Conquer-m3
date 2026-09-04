# conquer3 docker-compose orchestration.
#
# Three independent bring-up groups, matching docker-compose.yaml's profiles.
# None of the three depends on either other at the compose level (no
# cross-group `depends_on` edges) -- but within the first group, `ui`
# depends_on `scorer`, and Compose only resolves a profile-scoped depends_on
# when both profiles are active in the *same* command (see docker-compose.yaml's
# `ui` service comment). So `core` always brings up `serving`+`ui` together;
# splitting them would leave `ui` unable to resolve its dependency.
#
#   core    -> profiles: serving, ui   (scorer = serving endpoint, ui)
#   stream  -> profile:  stream        (pathway)
#   airflow -> profile:  pipeline      (airflow-postgres, airflow-init,
#                                       airflow-apiserver, airflow-scheduler,
#                                       airflow-dag-processor, airflow-triggerer)
#
# `--wait` makes every `up` block until Compose's own `depends_on`/healthcheck
# chain actually resolves (or the timeout below trips), instead of returning as
# soon as containers are merely started -- the same guarantee
# scripts/startup.sh's hand-rolled polling gives, without duplicating it.
#
# Usage: make help

SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

COMPOSE := docker compose

CORE_PROFILES    := serving,ui
STREAM_PROFILES  := stream
AIRFLOW_PROFILES := pipeline
ALL_PROFILES     := $(CORE_PROFILES),$(STREAM_PROFILES),$(AIRFLOW_PROFILES)

# Generous margins over each group's slowest healthcheck (start_period + retries
# * interval) so a genuinely stuck container fails loud instead of hanging the
# terminal forever -- see each service's healthcheck in docker-compose.yaml.
CORE_WAIT_TIMEOUT    := 300
STREAM_WAIT_TIMEOUT  := 120
AIRFLOW_WAIT_TIMEOUT := 300

SERVICE ?=

.PHONY: help check-env \
        core core-up core-down core-logs core-ps core-restart core-build \
        stream stream-up stream-down stream-logs stream-ps stream-restart stream-build \
        airflow airflow-up airflow-down airflow-logs airflow-ps airflow-restart airflow-build \
        up down ps logs restart clean build

help:
	@echo "conquer3 docker-compose orchestration"
	@echo
	@echo "  make core             up core services: scorer (serving endpoint) + ui"
	@echo "  make stream           up streaming: pathway"
	@echo "  make airflow          up airflow: postgres, init, apiserver, scheduler,"
	@echo "                        dag-processor, triggerer"
	@echo "  make up               bring up all three groups"
	@echo "  make down             tear down every profile"
	@echo "  make ps               show status across every profile"
	@echo "  make logs [SERVICE=x] follow logs (all profiles, optionally scoped)"
	@echo "  make restart          restart every profile"
	@echo "  make clean            down -v -- deletes local volumes (airflow"
	@echo "                        metadata, events, staging, models, pathway"
	@echo "                        state, duckdb). Postgres/Redis are external/"
	@echo "                        managed and untouched by this."
	@echo
	@echo "  make build            rebuild images for every service (all profiles) --"
	@echo "                        make's own CLI parser rejects a literal '--all' flag,"
	@echo "                        so this bare form IS the --all case"
	@echo "  make build SERVICE=x  rebuild just one service's image, e.g."
	@echo "                        make build SERVICE=scorer"
	@echo "  make core-build / stream-build / airflow-build   rebuild just one group's"
	@echo "                        images (e.g. only airflow's, without touching scorer/ui)"
	@echo
	@echo "  Per-group variants (swap the prefix): core-down, core-logs, core-ps,"
	@echo "  core-restart, core-build, stream-down, ..., airflow-down, ..."

check-env:
	@test -f .env || { echo "FAIL: .env not found. Run scripts/bootstrap.sh first." >&2; exit 1; }

# ─────────────────────────────── core (serving + ui) ───────────────────────────────
core: core-up

core-up: check-env
	mlflow_uri="$$(grep -E '^MLFLOW_TRACKING_URI=' .env | cut -d= -f2- || true)"
	if [ -z "$$mlflow_uri" ]; then
		echo "SKIP: MLFLOW_TRACKING_URI is empty in .env -- scorer resolves a"
		echo "  champion at boot and refuses to start without one. Fill it in"
		echo "  (and register+alias a champion), then re-run 'make core'."
		exit 0
	fi
	COMPOSE_PROFILES=$(CORE_PROFILES) $(COMPOSE) up -d --force-recreate --wait --wait-timeout $(CORE_WAIT_TIMEOUT)

core-down: check-env
	COMPOSE_PROFILES=$(CORE_PROFILES) $(COMPOSE) down

core-logs: check-env
	COMPOSE_PROFILES=$(CORE_PROFILES) $(COMPOSE) logs -f $(SERVICE)

core-ps: check-env
	COMPOSE_PROFILES=$(CORE_PROFILES) $(COMPOSE) ps

core-restart: check-env
	COMPOSE_PROFILES=$(CORE_PROFILES) $(COMPOSE) restart

core-build: check-env
	COMPOSE_PROFILES=$(CORE_PROFILES) $(COMPOSE) build

# ─────────────────────────────────── stream ────────────────────────────────────
stream: stream-up

stream-up: check-env
	COMPOSE_PROFILES=$(STREAM_PROFILES) $(COMPOSE) up -d --build --wait --wait-timeout $(STREAM_WAIT_TIMEOUT)

stream-down: check-env
	COMPOSE_PROFILES=$(STREAM_PROFILES) $(COMPOSE) down

stream-logs: check-env
	COMPOSE_PROFILES=$(STREAM_PROFILES) $(COMPOSE) logs -f $(SERVICE)

stream-ps: check-env
	COMPOSE_PROFILES=$(STREAM_PROFILES) $(COMPOSE) ps

stream-restart: check-env
	COMPOSE_PROFILES=$(STREAM_PROFILES) $(COMPOSE) restart

stream-build: check-env
	COMPOSE_PROFILES=$(STREAM_PROFILES) $(COMPOSE) build

# ─────────────────────────────────── airflow ───────────────────────────────────
airflow: airflow-up

airflow-up: check-env
	COMPOSE_PROFILES=$(AIRFLOW_PROFILES) $(COMPOSE) up -d --build --wait --wait-timeout $(AIRFLOW_WAIT_TIMEOUT)

airflow-down: check-env
	COMPOSE_PROFILES=$(AIRFLOW_PROFILES) $(COMPOSE) down

airflow-logs: check-env
	COMPOSE_PROFILES=$(AIRFLOW_PROFILES) $(COMPOSE) logs -f $(SERVICE)

airflow-ps: check-env
	COMPOSE_PROFILES=$(AIRFLOW_PROFILES) $(COMPOSE) ps

airflow-restart: check-env
	COMPOSE_PROFILES=$(AIRFLOW_PROFILES) $(COMPOSE) restart

airflow-build: check-env
	COMPOSE_PROFILES=$(AIRFLOW_PROFILES) $(COMPOSE) build

# ─────────────────────────────── combined (all groups) ─────────────────────────────
# airflow -> stream -> core: no group depends on another (see header), this is
# just a stable order to bring everything up in one call.
up: airflow-up stream-up core-up

# Naming a service explicitly (`docker compose build <service>`) bypasses profile
# scoping entirely (confirmed: it builds even with no COMPOSE_PROFILES set) --
# `SERVICE` is the same variable `logs` already uses. Without it, all profiles
# must be active or `docker compose build` silently builds nothing.
build: check-env
	if [ -n "$(SERVICE)" ]; then
		$(COMPOSE) build $(SERVICE)
	else
		COMPOSE_PROFILES=$(ALL_PROFILES) $(COMPOSE) build
	fi

down: check-env
	COMPOSE_PROFILES=$(ALL_PROFILES) $(COMPOSE) down

ps: check-env
	COMPOSE_PROFILES=$(ALL_PROFILES) $(COMPOSE) ps

logs: check-env
	COMPOSE_PROFILES=$(ALL_PROFILES) $(COMPOSE) logs -f $(SERVICE)

restart: check-env
	COMPOSE_PROFILES=$(ALL_PROFILES) $(COMPOSE) restart

clean: check-env
	COMPOSE_PROFILES=$(ALL_PROFILES) $(COMPOSE) down -v
