-- Silver: one row per transaction, typed and timestamped. Non-label columns map
-- 1:1 onto conquer3.core.types.TransactionEvent's fields. is_fraud/is_flagged_fraud
-- are kept here for training joins but are NEVER passed into TransactionEvent or
-- conquer3.core.features -- see pipelines/transforms/silver_to_gold.py.
CREATE TABLE IF NOT EXISTS silver.txn (
    event_id          TEXT NOT NULL PRIMARY KEY,
    account_id        TEXT NOT NULL,
    dest_id           TEXT NOT NULL,
    txn_type          TEXT NOT NULL,
    amount            DOUBLE PRECISION NOT NULL,
    oldbalance_org    DOUBLE PRECISION NOT NULL,
    newbalance_orig   DOUBLE PRECISION NOT NULL,
    oldbalance_dest   DOUBLE PRECISION NOT NULL,
    newbalance_dest   DOUBLE PRECISION NOT NULL,
    step              INTEGER NOT NULL,
    event_ts_us       BIGINT NOT NULL,
    is_fraud          BOOLEAN NOT NULL,
    is_flagged_fraud  BOOLEAN NOT NULL,
    bronze_row_num    BIGINT NOT NULL
);
