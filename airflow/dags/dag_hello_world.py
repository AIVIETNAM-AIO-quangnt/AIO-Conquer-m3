"""Layer 1 gate: proves Airflow can parse a DAG, schedule it, and run a task that
imports conquer3 -- before any real pipeline code exists.

Superseded once dag_bootstrap_warehouse (Layer 6) exists and has run successfully;
safe to delete at that point.
"""

from __future__ import annotations

import datetime

from airflow.sdk import DAG, task


@task
def say_hello() -> str:
    import conquer3

    message = f"hello from conquer3 {conquer3.__version__}"
    print(message)
    return message


with DAG(
    dag_id="hello_world",
    description="Layer 1 infra gate: Airflow can parse and run a DAG.",
    start_date=datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC),
    schedule=None,
    catchup=False,
    tags=["smoke-test"],
):
    say_hello()
