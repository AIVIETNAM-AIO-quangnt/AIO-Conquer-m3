# Streamlit console (Layer 9): Inference + Inspection tabs over the existing
# scorer -- src/conquer3/ui/. A client of the scorer, never a second scorer; it
# holds no model and never imports conquer3.serving (see the "ui talks to serving
# over HTTP, never by import" import-linter contract).
FROM python:3.12-slim

# curl: the container healthcheck (Streamlit's own GET /_stcore/health).
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir ".[ui]"

EXPOSE 8501
CMD ["conquer3", "ui"]
