"""OpenTelemetry wiring, shared by every entrypoint (the scorer -- both the
supervisor process and each uvicorn worker process, Pathway, Airflow tasks, the
producer).

If ``OTEL_EXPORTER_OTLP_ENDPOINT`` is unset, :func:`init_telemetry` installs no-op
providers, so tests and Colab never need a collector running. App code only ever
talks to this module -- never directly to Prometheus/Tempo/Loki -- which is what
keeps the remote observability endpoints pure configuration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from conquer3.config.settings import get_settings

if TYPE_CHECKING:
    from opentelemetry.metrics import Meter
    from opentelemetry.trace import Tracer

_initialized = False


def init_telemetry(service_name: str) -> None:
    """Idempotent per process. Call once at process startup, before any span or
    instrument is used -- in particular, ``serving.pyfunc_model.FraudScorerModel.
    load_context`` calls this too, not just the ``conquer3 serve`` CLI entrypoint,
    because uvicorn's scoring-server workers are separate OS processes from the
    supervisor that resolved and launched them (plan §8.5); without a call inside
    ``load_context`` itself, the process that actually handles ``/invocations``
    would never install a real provider and every metric/span recorded there would
    be silently dropped.

    Wires all three OTel signals to the same collector endpoint: traces, metrics (a
    MeterProvider -- without this, ``get_meter(...).create_counter(...)`` calls
    elsewhere in the codebase are safe no-ops that record nothing), and Python's
    root ``logging`` output (via a LoggingHandler), so the collector's logs
    pipeline (-> Loki) has something to receive.
    """
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
    from opentelemetry import _logs as logs_api
    from opentelemetry import metrics, trace
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    span_exporter_cls, metric_exporter_cls, log_exporter_cls = _exporter_classes(
        settings.exporter_otlp_protocol
    )
    resource = Resource.create({SERVICE_NAME: service_name})

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter_cls()))
    trace.set_tracer_provider(tracer_provider)

    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[PeriodicExportingMetricReader(metric_exporter_cls())],
    )
    metrics.set_meter_provider(meter_provider)

    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter_cls()))
    logs_api.set_logger_provider(logger_provider)
    logging.getLogger().addHandler(LoggingHandler(logger_provider=logger_provider))


def _grpc_exporter_classes() -> tuple[type, type, type]:
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    return OTLPSpanExporter, OTLPMetricExporter, OTLPLogExporter


def _http_exporter_classes() -> tuple[type, type, type]:
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    return OTLPSpanExporter, OTLPMetricExporter, OTLPLogExporter


def _exporter_classes(protocol: str) -> tuple[type, type, type]:
    """(SpanExporter, MetricExporter, LogExporter) classes for the configured
    ``OTEL_EXPORTER_OTLP_PROTOCOL`` (``grpc`` or ``http/protobuf``).

    Every exporter returned is constructed with no ``endpoint`` argument -- each
    resolves its own endpoint from ``OTEL_EXPORTER_OTLP_ENDPOINT`` (guaranteed set;
    see ``settings.enabled`` in :func:`init_telemetry`), including the per-signal
    path suffix (``/v1/traces``, ``/v1/metrics``, ``/v1/logs``) the HTTP variant
    requires -- so this module never builds a collector URL itself, and a remote
    endpoint that only speaks HTTP OTLP (common for managed Grafana ingest) works
    by changing ``OTEL_EXPORTER_OTLP_PROTOCOL`` alone. Split into two single-branch
    helpers, not one function with both imports inline, because mypy treats two
    same-named conditional imports in one function scope as a redefinition error.
    """
    return _http_exporter_classes() if protocol.startswith("http") else _grpc_exporter_classes()


def get_meter(name: str) -> Meter:
    """Returns an OTel Meter. Safe to call regardless of whether init_telemetry has
    run: before init (or when telemetry is disabled) this is the OTel API's own
    no-op meter; every instrument created against it is transparently "upgraded" in
    place once init_telemetry installs the real MeterProvider, per the OTel
    proxy-provider contract -- callers never need to check settings.otel themselves
    or re-fetch the meter after init."""
    from opentelemetry import metrics

    return metrics.get_meter(name)


def get_tracer(name: str) -> Tracer:
    """Same proxy-provider contract as get_meter, for spans."""
    from opentelemetry import trace

    return trace.get_tracer(name)
