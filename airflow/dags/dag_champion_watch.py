"""Layer 6 DAG 7: Hourly champion watch.

Observes the champion alias in MLflow and verifies that the serving supervisor
has picked it up (recorded in ops.model_deployments) within one poll interval.

No longer POSTs reload commands — reload is the supervisor's job (Layer 5).
This DAG is purely an observer and will alert if supervisor is out of sync.
"""

from __future__ import annotations

import datetime

from airflow.sdk import DAG, task


@task
def watch_champion_alias() -> str:
    """Resolve the champion alias and return version."""
    from conquer3.config.settings import get_settings
    from conquer3.contracts.model_registry import resolve_champion

    settings = get_settings()

    try:
        _model, ref = resolve_champion(settings.model.name)
        print(f"Champion alias resolved: version={ref.version}, run_id={ref.run_id}, degraded={ref.degraded}")
        return ref.version
    except Exception as e:
        raise RuntimeError(f"Failed to resolve champion alias: {e}")


@task
def verify_supervisor_synced(registry_version: str) -> str:
    """Verify ops.model_deployments records the same version.

    The supervisor polls resolve_champion() every C3_CHAMPION_POLL_S and updates
    ops.model_deployments on each promotion. This task verifies it's keeping up.
    """
    from conquer3.config.settings import get_settings
    from conquer3.db.engine import pg_connection

    settings = get_settings()

    with pg_connection() as conn:
        with conn.cursor() as cur:
            # Get the most recently recorded deployment
            cur.execute(
                "SELECT version, created_at FROM ops.model_deployments "
                "WHERE model_name = %s ORDER BY created_at DESC LIMIT 1",
                (settings.model.name,),
            )
            result = cur.fetchone()

            if result is None:
                raise AssertionError(
                    f"No deployment recorded yet for {settings.model.name} "
                    "(supervisor may not have started)"
                )

            recorded_version, recorded_at = result

            if recorded_version != registry_version:
                raise AssertionError(
                    f"Supervisor out of sync: registry says {registry_version}, "
                    f"ops.model_deployments says {recorded_version} (as of {recorded_at})"
                )

    print(f"Supervisor in sync: both track version {registry_version}")
    return f"synced: {registry_version}"


@task
def check_deployment_freshness(registry_version: str) -> str:
    """Verify the recorded deployment is recent (within one poll interval).

    Default C3_CHAMPION_POLL_S is typically 60s, so supervisor should update
    within ~2 minutes on a promotion.
    """
    import datetime as dt

    from conquer3.config.settings import get_settings
    from conquer3.db.engine import pg_connection

    settings = get_settings()
    poll_interval_s = settings.c3.champion_poll_s
    max_age_s = poll_interval_s * 2  # Allow 2x the poll interval before alerting

    with pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT created_at FROM ops.model_deployments "
                "WHERE model_name = %s AND version = %s ORDER BY created_at DESC LIMIT 1",
                (settings.model.name, registry_version),
            )
            result = cur.fetchone()

            if result is None:
                raise AssertionError(f"No deployment record found for version {registry_version}")

            recorded_at = result[0]
            now = dt.datetime.now(dt.timezone.utc)
            age_s = (now - recorded_at).total_seconds()

            if age_s > max_age_s:
                raise AssertionError(
                    f"Deployment record stale: recorded {age_s:.0f}s ago (>max {max_age_s}s). "
                    f"Supervisor may have crashed or is unresponsive."
                )

    print(f"Deployment record fresh: recorded {age_s:.1f}s ago (max {max_age_s}s)")
    return f"fresh: {age_s:.1f}s old"


@task
def record_watch_run() -> str:
    """Log this watch run in ops.pipeline_runs for audit trail."""
    from conquer3.db.engine import pg_connection
    from conquer3.db.ops import track_run

    with pg_connection() as conn:
        with track_run(conn, layer="champion_watch") as run:
            run.detail = "champion watch completed successfully"

    print("Watch run recorded in ops.pipeline_runs")
    return "recorded"


with DAG(
    dag_id="dag_champion_watch",
    description="Layer 6 gate 7: Hourly champion alias monitor (supervisor sync verification)",
    start_date=datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC),
    schedule="@hourly",
    catchup=False,
    tags=["layer-6", "champion", "hourly"],
) as dag:
    registry_version = watch_champion_alias()
    synced = verify_supervisor_synced(registry_version)
    fresh = check_deployment_freshness(registry_version)
    recorded = record_watch_run()

    registry_version >> synced >> fresh >> recorded
