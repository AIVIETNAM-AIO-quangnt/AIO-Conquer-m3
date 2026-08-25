"""OpenTelemetry wiring, shared by every entrypoint (BentoML, Pathway, Airflow tasks,
the producer).

If ``OTEL_EXPORTER_OTLP_ENDPOINT`` is unset, :func:`init_telemetry` installs no-op
providers, so tests and Colab never need a collector running. App code only ever
talks to this module -- never directly to Prometheus/Tempo/Loki -- which is what
keeps the remote observability endpoints pure configuration.
"""

from __future__ import annotations

import logging

from conquer3.config.settings import get_settings

_initialized = False


def init_telemetry(service_name: str) -> None:
    """Idempotent. Call once at process startup, before any instrument is used."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    settings = get_settings().otel
    if not settings.enabled:
        logging.getLogger(__name__).info(
            "OTEL_EXPORTER_OTLP_ENDPOINT unset; telemetry disabled for %s", service_name
        )
        return

    # Deferred import: opentelemetry is only in the `otel` extra, pulled in by
    # `serving`/`pipeline`/`stream`, never by `core`.
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.exporter_otlp_endpoint))
    )
    trace.set_tracer_provider(provider)
