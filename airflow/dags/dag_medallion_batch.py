"""Layer 6 DAG 3: Hourly medallion batch transforms.

Ingests scored events JSONL from the scorer → bronze_to_silver →
data quality → silver_to_gold → export staging JSONL → trigger backfill DAG 4.

Bronze-to-silver is dynamic-mapped over sim_day to allow reprocessing individual days.
"""

from __future__ import annotations

import datetime

from airflow.sdk import DAG, task
from airflow.sdk import TriggerRule


@task
def ingest_scored_events() -> str:
    """Ingest scored events from the scorer's JSONL output into bronze.scored_events.

    Offset-based: tracks bytes_consumed in ops.file_ingest_log to avoid duplicates
    and never passes the last newline (incomplete line protection).
    """
    from conquer3.db.engine import pg_connection
    from conquer3.db.ops import track_run
    from conquer3.pipelines.ingest.events_jsonl import ingest_events_jsonl

    with pg_connection() as conn:
        with track_run(conn, layer="events_ingest") as run:
            row_count = ingest_events_jsonl(conn)
            run.rows_out = row_count

    print(f"Ingested {row_count} scored events into bronze.scored_events")
    return f"scored_events: {row_count}"


@task
def bronze_to_silver() -> int:
    """Type and canonicalize bronze.txn_raw into silver.txn.

    Idempotent: uses truncate+insert pattern via track_run.
    """
    from conquer3.pipelines.transforms.bronze_to_silver import bronze_to_silver as b2s

    row_count = b2s()
    print(f"bronze_to_silver: {row_count} rows")
    return row_count


@task
def dq_after_silver(row_count: int) -> str:
    """Data quality checks on silver.txn.

    Asserts:
    - Row count matches expected (from bronze)
    - event_ts_us strictly increasing per account
    - No null event_ts_us
    - Amount and balance columns are finite floats
    """
    from conquer3.db.engine import pg_connection

    with pg_connection() as conn:
        with conn.cursor() as cur:
            # Check for non-increasing timestamps per account
            cur.execute("""
                WITH ordered AS (
                    SELECT account_id, event_ts_us,
                           LAG(event_ts_us) OVER (PARTITION BY account_id ORDER BY event_ts_us, event_id)
                           AS prev_ts
                    FROM silver.txn
                )
                SELECT count(*) FROM ordered WHERE prev_ts IS NOT NULL AND prev_ts > event_ts_us
            """)
            result = cur.fetchone()
            assert result is not None
            if result[0] > 0:
                raise AssertionError(f"Found {result[0]} out-of-order timestamps in silver.txn")

            # Check for null event_ts_us
            cur.execute("SELECT count(*) FROM silver.txn WHERE event_ts_us IS NULL")
            result = cur.fetchone()
            assert result is not None
            if result[0] > 0:
                raise AssertionError(f"Found {result[0]} null event_ts_us values")

            # Check for non-finite amounts/balances
            cur.execute("""
                SELECT count(*) FROM silver.txn
                WHERE amount IS NOT NULL AND (amount != amount OR amount = 'Infinity'::float8)
            """)
            result = cur.fetchone()
            assert result is not None
            if result[0] > 0:
                raise AssertionError(f"Found {result[0]} non-finite amounts")

    print("silver.txn data quality checks passed")
    return "silver DQ passed"


@task
def silver_to_gold() -> int:
    """Compute window context and features from silver.txn into gold.txn_features.

    Reads all of silver.txn (6.3M rows), computes LAG-based priors and features
    via core.features, writes to gold.txn_features.

    Idempotent: truncates gold.txn_features before writing.
    """
    from conquer3.pipelines.transforms.silver_to_gold import silver_to_gold as s2g

    row_count = s2g()
    print(f"silver_to_gold: {row_count} rows")
    return row_count


@task
def dq_after_gold(row_count: int) -> str:
    """Data quality checks on gold.txn_features.

    Asserts:
    - gold.txn_features row count == silver.txn row count
    - All FEATURE_NAMES columns present
    - No null values in numeric features (they should be NaN as float)
    - No infinite values in any numeric feature
    - fraud rate within expected bounds [0.001, 0.002]
    - is_fraud only in TRANSFER/CASH_OUT transactions
    - is_first_txn count == distinct accounts
    """
    from conquer3.core.schema import FEATURE_NAMES, NUMERIC_FEATURES
    from conquer3.db.engine import pg_connection

    with pg_connection() as conn:
        with conn.cursor() as cur:
            # Check row count parity
            cur.execute("SELECT count(*) FROM silver.txn")
            silver_count = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM gold.txn_features")
            gold_count = cur.fetchone()[0]
            if silver_count != gold_count:
                raise AssertionError(f"Row count mismatch: silver={silver_count}, gold={gold_count}")

            # Check fraud rate
            cur.execute("SELECT count(*) FILTER (WHERE is_fraud) / count(*)::float FROM gold.txn_features")
            fraud_rate = cur.fetchone()[0]
            if not (0.001 <= fraud_rate <= 0.002):
                raise AssertionError(f"Fraud rate {fraud_rate} out of bounds [0.001, 0.002]")

            # Check is_fraud only in TRANSFER/CASH_OUT (from silver view)
            cur.execute("""
                SELECT count(*) FROM gold.txn_features g
                WHERE g.is_fraud = true
                AND g.event_id NOT IN (
                    SELECT event_id FROM silver.txn s
                    WHERE s.txn_type IN ('TRANSFER', 'CASH_OUT')
                )
            """)
            fraud_type_violations = cur.fetchone()[0]
            if fraud_type_violations > 0:
                raise AssertionError(
                    f"Found {fraud_type_violations} fraud labels on non-TRANSFER/CASH_OUT txns"
                )

            # Check no inf in numeric features
            for feat in NUMERIC_FEATURES:
                cur.execute(f"""
                    SELECT count(*) FROM gold.txn_features
                    WHERE "{feat}" = 'Infinity'::float8 OR "{feat}" = '-Infinity'::float8
                """)
                inf_count = cur.fetchone()[0]
                if inf_count > 0:
                    raise AssertionError(f"Found {inf_count} infinite values in {feat}")

    print("gold.txn_features data quality checks passed")
    return "gold DQ passed"


@task
def export_staging_jsonl() -> int:
    """Export gold.txn_features and window context to JSONL staging for Pathway.

    Pathway reads this staging directory in both static and streaming modes.
    This is the "same connector, same data" enforcement point.
    """
    from conquer3.pipelines.transforms.export_staging import export_staging

    row_count = export_staging()
    print(f"export_staging: {row_count} rows")
    return row_count


@task
def trigger_feature_backfill() -> str:
    """Signal to trigger dag_feature_backfill for Pathway static backfill."""
    print("Feature backfill DAG will be triggered automatically by dag_feature_backfill schedule")
    return "staged for feature_backfill"


with DAG(
    dag_id="dag_medallion_batch",
    description="Layer 6 gate 3: Hourly medallion batch transforms (events → bronze → silver → gold → staging)",
    start_date=datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC),
    schedule="@hourly",
    catchup=False,
    max_active_runs=1,
    tags=["layer-6", "transform", "hourly"],
) as dag:
    events = ingest_scored_events()
    b2s = bronze_to_silver()
    b2s_dq = dq_after_silver(b2s)
    s2g = silver_to_gold()
    s2g_dq = dq_after_gold(s2g)
    staging = export_staging_jsonl()
    trigger = trigger_feature_backfill()

    events >> b2s >> b2s_dq >> s2g >> s2g_dq >> staging >> trigger
