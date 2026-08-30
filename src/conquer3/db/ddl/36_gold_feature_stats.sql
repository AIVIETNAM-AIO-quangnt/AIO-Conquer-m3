-- Baseline statistics per feature, refreshed by dag_feature_backfill's
-- refresh_feature_stats task. Used by monitoring/DQ for drift detection.
-- One row per feature_name, upserted in place -- not a history table.
CREATE TABLE IF NOT EXISTS gold.feature_stats (
    feature_name TEXT NOT NULL PRIMARY KEY,
    mean         double precision,
    stddev       double precision,
    p25          double precision,
    p50          double precision,
    p75          double precision
);
