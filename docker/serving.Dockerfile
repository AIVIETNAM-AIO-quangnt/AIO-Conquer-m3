# The scorer: a BentoML service (src/conquer3/serving/service.py) supervised by
# `conquer3 serve` (src/conquer3/serving/supervisor.py). This container IS the
# inference endpoint, not a proxy/gateway to the remote MLflow: the supervisor
# pulls the champion artifact out of remote MLflow at boot, pins the version in a
# pointer file, and the workers serve entirely from local files, local Redis, and
# local CPU -- a worker process never contacts MLflow at all. See README's Layer 5
# section and the architecture plan's §8 for the full boundary.
#
# Routes: POST /predict, POST /model_info, POST /invocations (deprecated MLflow
# envelope), plus BentoML's /livez, /healthz, /readyz, /metrics, the OpenAPI spec
# at /docs.json and Swagger UI at /.
FROM python:3.12-slim

# curl: the container healthcheck (`GET /readyz`, which returns 500 until the
# workers have loaded the champion).
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

# Editable install: `conquer3` resolves back to /app/src rather than a copy
# baked into site-packages, so docker-compose.yaml's ./src bind mount (local
# dev) actually takes effect without a rebuild. Behaves identically to a
# normal install when nothing is mounted over /app/src (e.g. a plain
# `docker build` with no compose bind mount).
RUN pip install --no-cache-dir -e ".[serving]"

EXPOSE 3000
CMD ["conquer3", "serve"]
