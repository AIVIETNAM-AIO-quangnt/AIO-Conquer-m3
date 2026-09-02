"""Static-mode entry point: read the current staging JSONL snapshot once, fold it
into per-account state, write to Redis + gold.account_state, then exit. Invoked by
`conquer3 pathway backfill`, Layer 6's dag_feature_backfill, and this layer's gate.
"""

from __future__ import annotations


def main() -> int:
    import pathway as pw

    from conquer3.config.settings import get_settings
    from conquer3.pipelines.pathway.graph import build_account_state_table
    from conquer3.pipelines.pathway.sinks.postgres_sink import write_account_state
    from conquer3.pipelines.pathway.sinks.redis_sink import RedisStateObserver
    from conquer3.pipelines.pathway.sources import read_staging_events
    from conquer3.telemetry.otel import init_telemetry

    init_telemetry("conquer3-pathway-backfill")
    settings = get_settings()
    pw.set_license_key(settings.pathway.license_key or None)

    events = read_staging_events(
        event_settings=settings.event, pathway_settings=settings.pathway, mode_override="static"
    )
    result = build_account_state_table(events)

    pw.io.python.write(
        result, RedisStateObserver(redis_settings=settings.redis, state_settings=settings.state)
    )
    write_account_state(result, pathway_settings=settings.pathway, pg_settings=settings.pg)

    pw.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
