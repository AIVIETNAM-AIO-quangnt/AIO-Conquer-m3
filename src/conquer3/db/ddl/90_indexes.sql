-- Applied last, deliberately: indexing after the Layer 2 bulk load is far faster
-- than indexing before it.
CREATE INDEX IF NOT EXISTS idx_bronze_txn_raw_step
    ON bronze.txn_raw (step);

CREATE INDEX IF NOT EXISTS idx_silver_txn_account_ts
    ON silver.txn (account_id, event_ts_us, event_id);

CREATE INDEX IF NOT EXISTS idx_gold_txn_features_account_ts
    ON gold.txn_features (account_id, event_ts_us, event_id);
