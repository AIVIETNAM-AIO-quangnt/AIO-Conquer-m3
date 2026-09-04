-- Applied last, deliberately: indexing after the Layer 2 bulk load is far faster
-- than indexing before it.
--
-- No index on bronze.txn_raw(step): nothing in the current pipeline filters
-- bronze by step (bronze_to_silver.py transforms the whole table in one
-- statement) -- the dynamic-per-sim_day reprocessing dag_medallion_batch.py's
-- docstring describes was never built. Measured at 41MB on 6.36M rows and
-- costing writes for zero reads; re-add `CREATE INDEX idx_bronze_txn_raw_step
-- ON bronze.txn_raw (step)` if that reprocessing feature gets built.
CREATE INDEX IF NOT EXISTS idx_silver_txn_account_ts
    ON silver.txn (account_id, event_ts_us, event_id);

CREATE INDEX IF NOT EXISTS idx_gold_txn_features_account_ts
    ON gold.txn_features (account_id, event_ts_us, event_id);

CREATE INDEX IF NOT EXISTS idx_gold_account_state_updated_at
    ON gold.account_state (updated_at_us);
