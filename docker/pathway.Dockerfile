# Pathway feature engine: static backfill and streaming state repair.
#
# pathway ships manylinux wheels only, which is fine here (the image is always
# Linux) but is why `stream` is kept out of the default local `uv sync` -- see
# pyproject.toml.
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir ".[stream]"

# Entry points land in Layer 3b: pipelines/pathway/run_backfill.py (static mode) and
# pipelines/pathway/run_streaming.py (streaming mode), selected by C3_PATHWAY_MODE.
CMD ["python", "-m", "conquer3.cli", "version"]
