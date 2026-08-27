-- GENERATED FILE -- do not hand-edit.
-- Regenerate with `conquer3 db gen-gold-ddl` after changing core/schema.py's
-- FEATURE_NAMES or pg_column_type(), then bump FEATURE_SCHEMA_VERSION and commit
-- both. tests/unit/test_ddl_gen.py enforces this file stays in sync.
--
-- Feature columns are nullable: conquer3.core.features leaves window features
-- undefined (NULL) on an account's first transaction -- see COLD_START_NULL_FEATURES
-- in core/schema.py.
CREATE TABLE IF NOT EXISTS gold.txn_features (
    event_id                 TEXT NOT NULL PRIMARY KEY,
    account_id                TEXT NOT NULL,
    event_ts_us                BIGINT NOT NULL,
    feature_schema_version     INTEGER NOT NULL,
    is_first_txn                double precision,
    seconds_since_last_txn      double precision,
    log1p_seconds_since_last    double precision,
    steps_since_last_txn        double precision,
    amount                      double precision,
    log1p_amount                double precision,
    amount_delta_vs_last        double precision,
    amount_ratio_vs_last        double precision,
    amount_velocity             double precision,
    amount_ratio_vs_prior_mean  double precision,
    amount_ratio_vs_prior_max   double precision,
    amount_z_vs_prior           double precision,
    txn_count_prior             double precision,
    account_age_hours           double precision,
    txn_rate_per_hour           double precision,
    type_changed                double precision,
    is_fraud_capable_type       double precision,
    balance_gap_org             double precision,
    balance_gap_flag            double precision,
    error_balance_orig          double precision,
    error_balance_dest          double precision,
    balance_delta_org           double precision,
    amount_to_balance_ratio     double precision,
    drains_account              double precision,
    orig_balance_was_zero       double precision,
    dest_is_merchant            double precision,
    dest_is_new                 double precision,
    dest_balance_was_zero       double precision,
    hour_of_day                 double precision,
    sim_day_of_week             double precision,
    prev_fraud_score            double precision,
    txn_type                    text,
    prev_txn_type               text,
    type_pair                   text,
    is_fraud                   BOOLEAN NOT NULL,
    is_flagged_fraud           BOOLEAN NOT NULL
);
