"""Layer 6 DAG 2: Ingest PaySim1 CSV into bronze.txn_raw (one-shot).

Loads the raw PaySim1 CSV into bronze.txn_raw via psycopg COPY,
asserts exactly 6,362,620 rows, and records the run in ops.pipeline_runs.
"""

from __future__ import annotations

import datetime

from airflow.sdk import DAG, task


@task
def download_csv() -> str:
    """Download the PaySim1 CSV from Kaggle if not already present."""
    from conquer3.config.settings import get_settings
    from conquer3.pipelines.ingest.kaggle import download_paysim_csv

    path = download_paysim_csv(get_settings().kaggle.csv_path)
    print(f"CSV available at: {path}")
    return str(path)


@task
def ingest_bronze(csv_path: str) -> int:
    """Load CSV into bronze.txn_raw via COPY and assert row count."""
    from conquer3.db.engine import pg_connection
    from conquer3.db.ops import track_run
    from conquer3.pipelines.ingest.bronze import load_csv_to_bronze

    # Use track_run to record in ops.pipeline_runs
    with pg_connection() as conn:
        with track_run(conn, layer="bronze_ingest") as run:
            row_count = load_csv_to_bronze(csv_path)
            run.rows_out = row_count

            # Assert exactly 6,362,620 rows
            if row_count != 6_362_620:
                raise AssertionError(
                    f"Expected 6,362,620 rows in bronze.txn_raw, got {row_count}"
                )

    print(f"Ingested bronze.txn_raw: {row_count} rows")
    return row_count


with DAG(
    dag_id="dag_bronze_ingest_paysim",
    description="Layer 6 gate 2: Ingest PaySim1 CSV into bronze (one-shot)",
    start_date=datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC),
    schedule=None,  # Manual trigger only
    catchup=False,
    tags=["layer-6", "ingest", "manual"],
) as dag:
    csv = download_csv()
    ingest = ingest_bronze(csv)

    csv >> ingest
