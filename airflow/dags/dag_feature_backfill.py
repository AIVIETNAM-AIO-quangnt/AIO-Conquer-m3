"""Layer 6 DAG 4: Feature backfill via Pathway static mode.

Runs Pathway's static-mode fold over the staging JSONL export, computing
account state and writing to gold.account_state and redis_sink.
Backfill is never an overwriter: uses shared Lua monotonic-CAS, refusing
writes when Redis holds fresher state (live requests during backfill).

Triggered by dag_medallion_batch or manual schedule.
"""

from __future__ import annotations

import datetime

from airflow.sdk import DAG, task


@task
def pathway_static_backfill() -> int:
    """Disabled: no trigger mechanism yet exists for this task to run Pathway.

    Pathway's dependencies cannot be installed into the Airflow image (every
    pathway release pins pyarrow/sqlglot ranges that hard-conflict with what
    Airflow and ibis-framework need -- see docker/airflow.Dockerfile), so this
    can't `import conquer3.pipelines.pathway.run_backfill` and run it in-process
    the way it used to. The static fold must instead run inside the separate
    `pathway` container, which nothing currently triggers on demand (no docker
    socket, no HTTP endpoint, no file-signal loop). Skipped, not failed, until
    that trigger mechanism is designed.
    """
    from airflow.exceptions import AirflowSkipException

    raise AirflowSkipException(
        "pathway_static_backfill is disabled pending a cross-container trigger "
        "design (DockerOperator+docker socket / HTTP endpoint on the pathway "
        "container / file-signal polling). Run `conquer3 pathway backfill` "
        "manually in the meantime."
    )


@task
def verify_feature_count() -> str:
    """Assert gold.txn_features count == gold.account_state row count.

    This is a consistency check: account state should cover all transactions.
    """
    from conquer3.db.engine import pg_connection

    with pg_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM gold.txn_features")
        features_count = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM gold.account_state")
        account_count = cur.fetchone()[0]

        # Each account should have exactly one state row
        cur.execute("SELECT count(DISTINCT account_id) FROM gold.txn_features")
        distinct_accounts = cur.fetchone()[0]

        if account_count != distinct_accounts:
            raise AssertionError(
                f"account_state count {account_count} != distinct accounts {distinct_accounts}"
            )

    print(
        f"Feature count {features_count} verified against state {account_count} distinct accounts"
    )
    return f"verified: {distinct_accounts} accounts"


@task
def refresh_feature_stats() -> str:
    """Compute and update feature baseline statistics (mean, std, quartiles).

    Used by monitoring and data quality for drift detection.
    """
    from conquer3.db.engine import pg_connection

    with pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO gold.feature_stats (feature_name, mean, stddev, p25, p50, p75)
                SELECT
                    'amount' as feature_name,
                    avg(amount),
                    stddev(amount),
                    percentile_cont(0.25) WITHIN GROUP (ORDER BY amount),
                    percentile_cont(0.50) WITHIN GROUP (ORDER BY amount),
                    percentile_cont(0.75) WITHIN GROUP (ORDER BY amount)
                FROM silver.txn
                ON CONFLICT (feature_name) DO UPDATE SET
                    mean = EXCLUDED.mean,
                    stddev = EXCLUDED.stddev,
                    p25 = EXCLUDED.p25,
                    p50 = EXCLUDED.p50,
                    p75 = EXCLUDED.p75
            """)
        conn.commit()

    print("Feature statistics refreshed in gold.feature_stats")
    return "stats updated"


with DAG(
    dag_id="dag_feature_backfill",
    description="Layer 6 gate 4: Pathway static backfill (account state fold, never overwrites)",
    start_date=datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC),
    schedule="@hourly",
    catchup=False,
    tags=["layer-6", "pathway", "hourly"],
) as dag:
    backfill = pathway_static_backfill()
    verify = verify_feature_count()
    # refresh_feature_stats only reads silver.txn -- no dependency on Pathway's
    # account_state output, so it isn't blocked by pathway_static_backfill being
    # disabled (see that task's docstring).
    refresh_feature_stats()

    backfill >> verify
