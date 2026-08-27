-- Bookkeeping for every ingest/transform invocation. Written by conquer3.db.ops
-- (wrapping pipelines/ingest and pipelines/transforms), read later by Layer 6's
-- DQ/skew-audit DAGs.
CREATE TABLE IF NOT EXISTS ops.pipeline_runs (
    run_id        BIGSERIAL PRIMARY KEY,
    layer         TEXT NOT NULL,
    started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at   TIMESTAMPTZ,
    status        TEXT NOT NULL DEFAULT 'running',
    rows_in       BIGINT,
    rows_out      BIGINT,
    detail        TEXT
);
