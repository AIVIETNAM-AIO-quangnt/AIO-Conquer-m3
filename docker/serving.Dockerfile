# BentoML fraud-scoring service.
#
# Built now (Layer 1) so the compose topology is complete end-to-end; the actual
# service code (src/conquer3/serving/service.py, bentofile.yaml) lands in Layer 5.
# Until then this image builds and installs the `serving` extra cleanly, but the CMD
# below has nothing to serve yet.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir ".[serving]"

EXPOSE 3000
CMD ["bentoml", "serve", "conquer3.serving.service:FraudScorer", "--host", "0.0.0.0", "--port", "3000"]
