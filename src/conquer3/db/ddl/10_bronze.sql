-- Bronze: raw PaySim1 rows, typed but otherwise unmodified. `row_num` is the
-- 1-based row order in the source CSV -- it is the intra-step tiebreaker that
-- conquer3.core.timeref.derive_event_ts_us needs, reproduced in 20_silver.sql's
-- companion transform (pipelines/transforms/bronze_to_silver.py).
CREATE TABLE IF NOT EXISTS bronze.txn_raw (
    row_num           BIGINT NOT NULL PRIMARY KEY,
    step              INTEGER NOT NULL,
    type              TEXT NOT NULL,
    amount            DOUBLE PRECISION NOT NULL,
    name_orig         TEXT NOT NULL,
    oldbalance_org    DOUBLE PRECISION NOT NULL,
    newbalance_orig   DOUBLE PRECISION NOT NULL,
    name_dest         TEXT NOT NULL,
    oldbalance_dest   DOUBLE PRECISION NOT NULL,
    newbalance_dest   DOUBLE PRECISION NOT NULL,
    is_fraud          BOOLEAN NOT NULL,
    is_flagged_fraud  BOOLEAN NOT NULL,
    source_file       TEXT NOT NULL,
    loaded_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
