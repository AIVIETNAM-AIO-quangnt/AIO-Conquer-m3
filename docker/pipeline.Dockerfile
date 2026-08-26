# Shared image for medallion pipeline tasks invoked outside Airflow's own image
# (e.g. one-off `conquer3` CLI runs) and the transaction-replay producer.
#
# Airflow itself does NOT use this image -- see docker/airflow.Dockerfile, which
# layers the same `pipeline` extra on top of Airflow's own constraints file instead
# of this base, so Airflow's transitive pins are never fought.
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir ".[pipeline]"

# No single long-running entrypoint: docker-compose invokes this image with an
# explicit command (`producer`, Layer 5) or `conquer3` CLI subcommands (Layer 2+).
CMD ["python", "-m", "conquer3.cli", "version"]
