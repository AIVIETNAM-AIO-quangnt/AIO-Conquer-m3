"""The feature contract: names, types, and versions.

``FEATURE_NAMES`` is the **single source of truth** for the served model's input
columns. The Postgres ``gold.txn_features`` DDL, the Pathway output projection, the
sklearn ColumnTransformer, and the model signature are all generated from these
lists -- never hand-maintained in parallel. Adding a feature is: edit this file,
regenerate the DDL, bump ``FEATURE_SCHEMA_VERSION``, retrain.

``gold.txn_features`` additionally carries ``EXTERNAL_MODEL_FEATURES`` -- reserved
columns for other MLflow-registered models' own feature names, kept deliberately
separate from ``FEATURE_NAMES`` (see that constant's docstring below).

This module imports nothing. Keep it that way.
"""

from __future__ import annotations

from typing import Final

# Bump when FEATURE_NAMES or any feature's *meaning* changes. The serving layer
# refuses to load a model whose recorded version differs from this one, which turns
# "added a feature but forgot to retrain" from a silent mis-scoring into a hard fail.
FEATURE_SCHEMA_VERSION: Final[int] = 1

# Bump when AccountState's shape changes. The Redis key embeds this, so a bump is a
# clean cutover rather than a migration -- old keys simply expire via TTL.
STATE_SCHEMA_VERSION: Final[int] = 1

# Categorical placeholder for "this account has no previous transaction". A distinct
# level, never a numeric sentinel -- see the cold-start policy in features.py.
NO_PREV_CATEGORY: Final[str] = "__NONE__"

# Denominators below this are treated as zero by _safe_ratio.
EPSILON: Final[float] = 1e-9

# Balances are currency with 2 decimals; anything under this is "zero".
BALANCE_EPSILON: Final[float] = 0.01


NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    # -- cold-start marker ------------------------------------------------
    "is_first_txn",
    # -- recency vs the account's last transaction ------------------------
    "seconds_since_last_txn",
    "log1p_seconds_since_last",
    "steps_since_last_txn",
    # -- amount, absolute --------------------------------------------------
    "amount",
    "log1p_amount",
    # -- amount vs the last transaction ------------------------------------
    "amount_delta_vs_last",
    "amount_ratio_vs_last",
    "amount_velocity",
    # -- amount vs the account's running priors ----------------------------
    "amount_ratio_vs_prior_mean",
    "amount_ratio_vs_prior_max",
    "amount_z_vs_prior",
    # -- account tenure / rate ---------------------------------------------
    "txn_count_prior",
    "account_age_hours",
    "txn_rate_per_hour",
    # -- type transition ---------------------------------------------------
    "type_changed",
    "is_fraud_capable_type",
    # -- balance consistency -----------------------------------------------
    "balance_gap_org",
    "balance_gap_flag",
    "error_balance_orig",
    "error_balance_dest",
    "balance_delta_org",
    "amount_to_balance_ratio",
    "drains_account",
    "orig_balance_was_zero",
    # -- destination -------------------------------------------------------
    "dest_is_merchant",
    "dest_is_new",
    "dest_balance_was_zero",
    # -- calendar ----------------------------------------------------------
    "hour_of_day",
    "sim_day_of_week",
    # -- feedback ----------------------------------------------------------
    "prev_fraud_score",
)

CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "txn_type",
    "prev_txn_type",
    "type_pair",
)

FEATURE_NAMES: Final[tuple[str, ...]] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Features that are undefined on an account's first transaction. Enforced by test:
# every one of these must be None (numeric) or NO_PREV_CATEGORY (categorical) when
# `prev is None`. Deliberately excludes `txn_count_prior` (legitimately 0) and
# `is_first_txn` (legitimately 1).
COLD_START_NULL_FEATURES: Final[frozenset[str]] = frozenset(
    {
        "seconds_since_last_txn",
        "log1p_seconds_since_last",
        "steps_since_last_txn",
        "amount_delta_vs_last",
        "amount_ratio_vs_last",
        "amount_velocity",
        "amount_ratio_vs_prior_mean",
        "amount_ratio_vs_prior_max",
        "amount_z_vs_prior",
        "account_age_hours",
        "txn_rate_per_hour",
        "type_changed",
        "balance_gap_org",
        "balance_gap_flag",
        "dest_is_new",
        "prev_fraud_score",
        "prev_txn_type",
        "type_pair",
    }
)

# Columns present in the source data that must NEVER become features.
#   isFlaggedFraud -- the simulator's own rule-based flag; leaks the label.
#   isFraud        -- the label.
#   nameOrig/nameDest -- account identifiers; 6.3M cardinality, and identity is not
#                        a generalisable signal. Only their derived booleans are used.
FORBIDDEN_FEATURE_SOURCES: Final[frozenset[str]] = frozenset(
    {"is_fraud", "is_flagged_fraud", "isFraud", "isFlaggedFraud", "account_id", "dest_id"}
)

# Feature columns other MLflow-registered model families declare as their own inputs
# -- paysim-fraud-decision_tree, -lightgbm, -rf, -xgb-baseline, -xgb-enhanced,
# -xgb-optimal, -xgb_vanila (checked against the live registry; none of them carry
# this codebase's feature_schema_version tag, so none can ever become champion --
# see contracts.model_registry.verify_compatible). NOT part of the model contract:
# conquer3.core.features never computes these, FEATURE_SCHEMA_VERSION does not cover
# them, and the live scorer never reads them. They exist purely so gold.txn_features
# has room for every registered model's declared columns -- a job that scores with a
# non-champion model can select the columns it needs from this table instead of
# hitting "column does not exist". Expect most rows to leave these NULL until a
# dedicated job backfills that specific model's features (see TODO.md).
#
# Deduplicated against FEATURE_NAMES and FORBIDDEN_FEATURE_SOURCES: a name already
# covered by an existing column is not repeated here (e.g. "amount", "hour_of_day"),
# and paysim-fraud-lightgbm v4's own "isFlaggedFraud" input -- a label leak
# FORBIDDEN_FEATURE_SOURCES already blocks -- is dropped in favour of the existing
# `is_flagged_fraud` label column. Where the same name surfaced with conflicting
# numeric types across models (e.g. double vs. long), the wider `double precision`
# was kept. See ``db/ddl_gen.py`` for how this renders into the DDL.
EXTERNAL_MODEL_FEATURES: Final[tuple[tuple[str, str], ...]] = (
    ("amount_log", "double precision"),
    ("amount_log10", "double precision"),
    ("amount_log_zscore_by_type_hour", "double precision"),
    ("amount_ratio", "double precision"),
    ("amount_to_dest_mean_ratio", "double precision"),
    ("amount_to_orig_mean_ratio", "double precision"),
    ("amount_zscore_by_type_hour", "double precision"),
    ("current_amount", "double precision"),
    ("day_index", "integer"),
    ("day_of_week", "integer"),
    ("dest_amount_mean", "double precision"),
    ("dest_amount_mean_hist", "double precision"),
    ("dest_amount_sum_hist", "double precision"),
    ("dest_cashout_freq", "integer"),
    ("dest_count_hist", "bigint"),
    ("dest_freq", "integer"),
    ("dest_is_frequent", "integer"),
    ("dest_risk_target_enc", "double precision"),
    ("dest_type_count", "integer"),
    ("dest_velocity_1h", "integer"),
    ("dest_velocity_24h", "integer"),
    ("dest_velocity_surge_ratio", "double precision"),
    ("edge_count_hist", "bigint"),
    ("edge_is_new", "integer"),
    ("f1", "double precision"),
    ("f2", "double precision"),
    ("f3", "double precision"),
    ("hour_cos", "double precision"),
    ("hour_day", "integer"),
    ("hour_sin", "double precision"),
    ("is_capped_10m", "integer"),
    ("is_customer_dest", "integer"),
    ("is_mule_chain", "integer"),
    ("is_night", "integer"),
    ("is_rapid_passthrough", "integer"),
    ("is_round_10k", "integer"),
    ("is_round_1k", "integer"),
    ("is_transfer", "integer"),
    ("is_weekend", "integer"),
    ("log_amount", "double precision"),
    ("mule_time_since_transfer", "double precision"),
    ("orig_amount_mean_hist", "double precision"),
    ("orig_amount_sum_hist", "double precision"),
    ("orig_count_hist", "bigint"),
    ("orig_is_first_seen", "integer"),
    ("orig_velocity_1h", "integer"),
    ("orig_velocity_24h", "integer"),
    ("pit_distinct_senders_168h", "double precision"),
    ("pit_distinct_senders_24h", "double precision"),
    ("pit_prior_amount_168h", "double precision"),
    ("pit_prior_amount_1h", "double precision"),
    ("pit_prior_amount_24h", "double precision"),
    ("pit_prior_count_1h", "double precision"),
    ("pit_prior_count_24h", "double precision"),
    ("pit_steps_since_last_event", "double precision"),
    ("step", "bigint"),
    ("step_day", "integer"),
    ("transaction_type_transfer", "double precision"),
    ("type", "text"),
    ("type_code", "integer"),
)

# Postgres column type per feature, used to generate gold.txn_features DDL.
PG_TYPE_NUMERIC: Final[str] = "double precision"
PG_TYPE_CATEGORICAL: Final[str] = "text"


def pg_column_type(feature: str) -> str:
    """Postgres type for a feature column."""
    if feature in CATEGORICAL_FEATURES:
        return PG_TYPE_CATEGORICAL
    if feature in NUMERIC_FEATURES:
        return PG_TYPE_NUMERIC
    raise KeyError(f"unknown feature: {feature!r}")


def validate() -> None:
    """Self-check invariants. Called by tests and by the DDL generator."""
    if len(set(FEATURE_NAMES)) != len(FEATURE_NAMES):
        dupes = sorted({n for n in FEATURE_NAMES if FEATURE_NAMES.count(n) > 1})
        raise ValueError(f"duplicate feature names: {dupes}")
    overlap = set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES)
    if overlap:
        raise ValueError(f"features declared both numeric and categorical: {sorted(overlap)}")
    unknown = COLD_START_NULL_FEATURES - set(FEATURE_NAMES)
    if unknown:
        raise ValueError(f"COLD_START_NULL_FEATURES names unknown features: {sorted(unknown)}")
    leaked = set(FEATURE_NAMES) & FORBIDDEN_FEATURE_SOURCES
    if leaked:
        raise ValueError(f"label-leaking columns present in FEATURE_NAMES: {sorted(leaked)}")
    ext_names = [name for name, _ in EXTERNAL_MODEL_FEATURES]
    if len(set(ext_names)) != len(ext_names):
        dupes = sorted({n for n in ext_names if ext_names.count(n) > 1})
        raise ValueError(f"duplicate names in EXTERNAL_MODEL_FEATURES: {dupes}")
    reused = (set(ext_names) & set(FEATURE_NAMES)) | (set(ext_names) & FORBIDDEN_FEATURE_SOURCES)
    if reused:
        raise ValueError(
            f"EXTERNAL_MODEL_FEATURES reuses a FEATURE_NAMES/forbidden name: {sorted(reused)}"
        )
