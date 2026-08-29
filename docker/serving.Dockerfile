# The scorer: MLflow's own scoring server (mlflow.pyfunc.scoring_server, i.e.
# FastAPI + uvicorn), vendored as a library and launched by our own supervisor
# (`conquer3 serve`, src/conquer3/serving/supervisor.py) -- not `mlflow models
# serve`, and not a proxy/gateway to the remote MLflow. This container IS the
# inference endpoint: it pulls the champion artifact out of remote MLflow once at
# boot, then serves entirely from local files, local Redis, and local CPU. See
# README's Layer 5 section and the architecture plan's §8 for the full boundary.
FROM python:3.12-slim

# bash: the supervisor launches uvicorn via `bash -c "exec ..."` so it owns the
# uvicorn master PID directly (see supervisor.py's module docstring). curl: the
# container healthcheck (`GET /ping`, one of the scoring server's four fixed
# routes -- there is no /readyz, that was a BentoML artifact).
RUN apt-get update && apt-get install -y --no-install-recommends curl bash \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir ".[serving]"

EXPOSE 3000
CMD ["conquer3", "serve"]
