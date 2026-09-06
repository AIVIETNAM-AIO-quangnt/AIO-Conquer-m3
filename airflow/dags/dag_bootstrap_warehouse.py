"""Layer 6 DAG 1: Bootstrap the medallion warehouse (manual, one-shot).

Creates the schema DDL in order, then applies indexes last. Run this once
before any ingestion.
"""

from __future__ import annotations

import datetime

from airflow.sdk import DAG, task


@task
def migrate_schema() -> str:
    """Apply all schema DDL idempotently."""
    from conquer3.db.bootstrap import apply_ddl
    from conquer3.db.engine import pg_connection

    with pg_connection() as conn:
        applied = apply_ddl(conn)
    result = "; ".join(applied)
    print(f"Applied DDL: {result}")
    return result


@task
def apply_indexes() -> str:
    """Create indexes last, after bulk load is complete.

    Gold layer only -- bronze/silver indexing is entirely owned by
    db/ddl/90_indexes.sql, applied idempotently via migrate_schema's apply_ddl()
    above. This task used to also create idx_silver_txn_acct_ts (a 314MB strict
    column-prefix duplicate of 90_indexes.sql's idx_silver_txn_account_ts) and
    idx_silver_txn_ts_brin -- both removed so bronze/silver indexing has exactly
    one owner instead of two DAGs racing to define overlapping indexes.
    """
    from conquer3.db.engine import pg_connection

    with pg_connection() as conn:
        with conn.cursor() as cur:
            # Gold layer indexes
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_gold_features_acct_ts
                ON gold.txn_features (account_id, event_ts_us)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_gold_features_fraud
                ON gold.txn_features (is_fraud)
                WHERE is_fraud = true
            """)
        conn.commit()
    return "indexes created"


with DAG(
    dag_id="dag_bootstrap_warehouse",
    description="Layer 6 gate 1: Create medallion schemas and indexes (manual, one-shot)",
    start_date=datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC),
    schedule=None,  # Manual trigger only
    catchup=False,
    tags=["layer-6", "warehouse", "manual"],
) as dag:
    migrate = migrate_schema()
    indexes = apply_indexes()

    migrate >> indexes
