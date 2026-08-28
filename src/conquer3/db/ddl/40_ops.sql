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

-- Audit trail of resolved model versions -- written by db.ops.record_model_deployment,
-- called by whichever process resolves a champion (BentoML boot, Layer 5). Aliases
-- are mutable; this table names the immutable version actually loaded, and whether
-- resolution fell back to the degraded (cached) path.
CREATE TABLE IF NOT EXISTS ops.model_deployments (
    deployment_id BIGSERIAL PRIMARY KEY,
    model_name    TEXT NOT NULL,
    version       TEXT NOT NULL,
    run_id        TEXT NOT NULL,
    alias         TEXT NOT NULL,
    degraded      BOOLEAN NOT NULL DEFAULT false,
    resolved_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
