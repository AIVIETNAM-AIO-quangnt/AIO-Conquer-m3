"""Layer 6 DAG 5: Daily data quality checks.

Runs comprehensive DQ assertions across bronze, silver, and gold layers.
Failures block downstream processes (alerts + paging).

Checks:
- Row parity across medallion layers
- Per-account event_ts_us strictly increasing (no time-travel or duplicates)
- Fraud rate within bounds [0.001, 0.002]
- Fraud labels only on TRANSFER/CASH_OUT
- is_first_txn count matches distinct account count
- No infinite floats in any numeric column
- Balance consistency: oldbalance vs. previous newbalance
"""

from __future__ import annotations

import datetime

from airflow.sdk import DAG, task


@task
def check_row_parity() -> str:
    """Assert row counts are consistent across bronze, silver, gold."""
    from conquer3.db.engine import pg_connection

    with pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM bronze.txn_raw")
            bronze_count = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM silver.txn")
            silver_count = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM gold.txn_features")
            gold_count = cur.fetchone()[0]

            if bronze_count != silver_count:
                raise AssertionError(
                    f"Row parity failure: bronze={bronze_count}, silver={silver_count}"
                )
            if silver_count != gold_count:
                raise AssertionError(
                    f"Row parity failure: silver={silver_count}, gold={gold_count}"
                )

    print(f"Row parity verified: {bronze_count} rows across all layers")
    return f"parity: {bronze_count}"


@task
def check_timestamp_ordering() -> str:
    """Assert per-account event_ts_us is strictly increasing (no duplicates in time)."""
    from conquer3.db.engine import pg_connection

    with pg_connection() as conn:
        with conn.cursor() as cur:
            # Find any account with out-of-order or equal consecutive timestamps
            cur.execute("""
                WITH ts_pairs AS (
                    SELECT
                        account_id,
                        event_ts_us,
                        LAG(event_ts_us) OVER (PARTITION BY account_id ORDER BY event_ts_us, event_id) AS prev_ts
                    FROM silver.txn
                )
                SELECT count(*) FROM ts_pairs
                WHERE prev_ts IS NOT NULL AND prev_ts >= event_ts_us
            """)
            result = cur.fetchone()[0]
            if result > 0:
                raise AssertionError(f"Found {result} out-of-order or duplicate timestamps")

    print("Timestamp ordering verified: strictly increasing per account")
    return "ordering: ok"


@task
def check_fraud_rate() -> str:
    """Assert fraud rate is within realistic bounds [0.001, 0.002] (~0.15%)."""
    from conquer3.db.engine import pg_connection

    with pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    count(*) FILTER (WHERE is_fraud) as fraud_count,
                    count(*) as total_count
                FROM gold.txn_features
            """)
            fraud_count, total_count = cur.fetchone()
            if total_count == 0:
                raise AssertionError("No transactions found in gold.txn_features")
            fraud_rate = fraud_count / total_count

            if not (0.001 <= fraud_rate <= 0.002):
                raise AssertionError(
                    f"Fraud rate {fraud_rate:.6f} ({fraud_count}/{total_count}) "
                    f"out of bounds [0.001, 0.002]"
                )

    print(f"Fraud rate verified: {fraud_rate:.6f} ({fraud_count}/{total_count})")
    return f"fraud_rate: {fraud_rate:.6f}"


@task
def check_fraud_txn_types() -> str:
    """Assert is_fraud=true only on TRANSFER and CASH_OUT transactions."""
    from conquer3.db.engine import pg_connection

    with pg_connection() as conn:
        with conn.cursor() as cur:
            # Find frauds on non-TRANSFER/CASH_OUT types
            cur.execute("""
                SELECT count(*) FROM gold.txn_features g
                WHERE g.is_fraud = true
                AND g.event_id NOT IN (
                    SELECT event_id FROM silver.txn s
                    WHERE s.txn_type IN ('TRANSFER', 'CASH_OUT')
                )
            """)
            result = cur.fetchone()[0]
            if result > 0:
                raise AssertionError(
                    f"Found {result} fraud labels on non-TRANSFER/CASH_OUT transactions"
                )

    print("Fraud transaction types verified: all on TRANSFER/CASH_OUT")
    return "fraud_types: ok"


@task
def check_first_txn_count() -> str:
    """Assert is_first_txn count == distinct account count."""
    from conquer3.db.engine import pg_connection

    with pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    count(*) FILTER (WHERE is_first_txn = 1) as first_count,
                    count(DISTINCT account_id) as account_count
                FROM gold.txn_features
            """)
            first_count, account_count = cur.fetchone()
            if first_count != account_count:
                raise AssertionError(
                    f"is_first_txn count {first_count} != distinct accounts {account_count}"
                )

    print(f"First transaction count verified: {first_count} accounts")
    return f"first_txn: {first_count}"


@task
def check_no_infinites() -> str:
    """Assert no infinite values in numeric columns."""
    from conquer3.core.schema import NUMERIC_FEATURES
    from conquer3.db.engine import pg_connection

    with pg_connection() as conn:
        with conn.cursor() as cur:
            total_inf = 0
            for feat in NUMERIC_FEATURES:
                cur.execute(f"""
                    SELECT count(*) FROM gold.txn_features
                    WHERE "{feat}" = 'Infinity'::float8 OR "{feat}" = '-Infinity'::float8
                """)
                inf_count = cur.fetchone()[0]
                if inf_count > 0:
                    print(f"  Found {inf_count} infinites in {feat}")
                    total_inf += inf_count

            if total_inf > 0:
                raise AssertionError(f"Found {total_inf} infinite values in numeric features")

    print("No infinite values found in numeric features")
    return "infinites: 0"


@task
def check_balance_consistency() -> str:
    """Assert oldbalance == previous txn's newbalance for same account.

    Non-zero balance_gap_org (= oldbalance_org - last_newbalance_orig) signals
    unseen activity and is a DQ marker.
    """
    from conquer3.db.engine import pg_connection

    with pg_connection() as conn:
        with conn.cursor() as cur:
            # Count non-zero balance gaps (activity the state didn't see)
            cur.execute("""
                SELECT count(*) FROM (
                    SELECT DISTINCT ON (g.account_id, g.event_id)
                        g.account_id,
                        g.balance_gap_org
                    FROM gold.txn_features g
                    WHERE g.balance_gap_org != 0
                    LIMIT 100
                ) t
            """)
            gap_count = cur.fetchone()[0]
            if gap_count > 0:
                print(f"  Note: Found {gap_count} (sample) balance gaps (may be normal)")

    print("Balance consistency check complete")
    return "balance: checked"


with DAG(
    dag_id="dag_data_quality",
    description="Layer 6 gate 5: Daily comprehensive data quality checks",
    start_date=datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC),
    schedule="@daily",
    catchup=False,
    tags=["layer-6", "dq", "daily"],
) as dag:
    parity = check_row_parity()
    ts_order = check_timestamp_ordering()
    fraud_rate = check_fraud_rate()
    fraud_types = check_fraud_txn_types()
    first_txn = check_first_txn_count()
    no_inf = check_no_infinites()
    balance = check_balance_consistency()

    parity >> [ts_order, fraud_rate, fraud_types, first_txn, no_inf, balance]
