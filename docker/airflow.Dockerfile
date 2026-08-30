# Airflow, with conquer3's pipeline code layered on top.
#
# Extras are deliberately NOT a plain `uv sync` fought against Airflow's own
# transitive pins -- instead this installs them constrained by Airflow's own
# published constraints file, which is the documented way to add packages to the
# official image without breaking it: https://airflow.apache.org/docs/docker-stack/build.html
#
# All airflow-* compose services (apiserver, scheduler, dag-processor, triggerer,
# init) build from this single image. The dag-processor in particular must be able
# to `import conquer3.pipelines...` to parse our DAG files, so partial installs
# across services would be a foot-gun for no real benefit in a local/dev stack.
#
# `pipeline,registry,stream` is the full union every airflow/dags/*.py task ends up
# importing at run time, not just `pipeline`: dag_champion_watch imports
# contracts.model_registry (needs `registry`'s mlflow), dag_feature_backfill imports
# pipelines.pathway.run_backfill (needs `stream`'s pathway), and the skew/state
# audits import serving.state_store (needs `stream`'s redis). Those imports are
# deferred (inside @task bodies), so DAG *parsing* succeeds either way -- only
# installing `pipeline` alone silently defers the ModuleNotFoundError to whichever
# task happens to run first, one extra at a time.
ARG AIRFLOW_VERSION=3.3.1
ARG PYTHON_VERSION=3.12
FROM apache/airflow:${AIRFLOW_VERSION}-python${PYTHON_VERSION}

ARG AIRFLOW_VERSION
ARG PYTHON_VERSION

USER airflow

WORKDIR /opt/conquer3
COPY --chown=airflow:root pyproject.toml README.md ./
COPY --chown=airflow:root src ./src

# `stream` (pathway) is deliberately NOT installed here, in this image or any
# combination/ordering of pip invocations: every published pathway release pins
# either pyarrow<19 or sqlglot==10.6.1, both incompatible with the newer
# pyarrow/sqlglot that Airflow's own constraints file and `pipeline`'s
# ibis-framework require -- a hard ResolutionImpossible, not an ordering problem.
# This is exactly why docker/pathway.Dockerfile exists as its own image with its
# own unconstrained environment; any DAG task that needs Pathway must trigger
# that `pathway` compose service rather than `import pathway` in-process here.
# `redis` (the one other piece of `stream` a DAG task needs -- conquer3.serving.
# state_store, used by the skew/state audits) has no such conflict, so it's
# installed directly rather than pulling in all of `stream` for it.
#
# registry (mlflow) can't join the constrained `pipeline` install either: every
# mlflow 3.x release requires cryptography<47, while Airflow ${AIRFLOW_VERSION}'s
# constraints file pins cryptography==50.0.0 for python${PYTHON_VERSION}. mlflow is
# installed unconstrained afterward instead, letting its resolver settle its own
# compatible cryptography; nothing in this stack's own Fernet/JWT usage requires
# cryptography>=47 specifically (see AIRFLOW__CORE__FERNET_KEY above), so the
# resulting downgrade is safe here.
RUN pip install --no-cache-dir \
      --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt" \
      ".[pipeline]" "redis>=5.0" \
 && pip install --no-cache-dir ".[registry]"

WORKDIR /opt/airflow
