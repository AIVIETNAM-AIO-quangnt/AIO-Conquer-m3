# Airflow, with conquer3's pipeline code layered on top.
#
# `pipeline` is deliberately NOT a plain `uv sync` extra fought against Airflow's own
# transitive pins -- instead this installs it constrained by Airflow's own published
# constraints file, which is the documented way to add packages to the official
# image without breaking it: https://airflow.apache.org/docs/docker-stack/build.html
#
# All airflow-* compose services (apiserver, scheduler, dag-processor, triggerer,
# init) build from this single image. The dag-processor in particular must be able
# to `import conquer3.pipelines...` to parse our DAG files, so partial installs
# across services would be a foot-gun for no real benefit in a local/dev stack.
ARG AIRFLOW_VERSION=3.3.1
ARG PYTHON_VERSION=3.12
FROM apache/airflow:${AIRFLOW_VERSION}-python${PYTHON_VERSION}

ARG AIRFLOW_VERSION
ARG PYTHON_VERSION

USER airflow

WORKDIR /opt/conquer3
COPY --chown=airflow:root pyproject.toml README.md ./
COPY --chown=airflow:root src ./src

RUN pip install --no-cache-dir \
      --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt" \
      ".[pipeline]"

WORKDIR /opt/airflow
