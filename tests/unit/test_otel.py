"""conquer3.telemetry.otel: the no-op-when-unset contract, per-process
idempotency, and OTEL_EXPORTER_OTLP_PROTOCOL exporter dispatch.

Every test that actually calls init_telemetry() runs in a *subprocess*:
init_telemetry mutates process-global state (its own module-level `_initialized`
flag, plus the OTel API's global tracer/meter/logger providers) that, once set to a
real (non-proxy) provider, cannot be reset from inside the same interpreter -- a
second `set_tracer_provider` call is a silent no-op. Calling it in-process would
permanently wedge `_initialized=True` for the rest of this pytest session, so every
*other* test that expects init_telemetry to actually initialize something (e.g. a
future serving/pathway entrypoint test) would silently no-op instead. See
tests/contract/test_core_is_dependency_light.py's docstring for the same concern.
"""

from __future__ import annotations

import os
import subprocess
import sys

from conquer3.telemetry.otel import _exporter_classes


def _run(script: str, *, otlp_endpoint: str | None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    # An OS env var (even "") always outranks a real .env FILE value in
    # pydantic-settings' precedence -- this repo's own .env has a real
    # OTEL_EXPORTER_OTLP_ENDPOINT set (see .env.example), so merely `del`-ing the
    # OS var would still resolve settings.otel.exporter_otlp_endpoint from that
    # file. An explicit "" is what actually reproduces "unset".
    env["OTEL_EXPORTER_OTLP_ENDPOINT"] = otlp_endpoint or ""
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False, env=env
    )


def test_exporter_classes_dispatch_on_protocol() -> None:
    span_cls, metric_cls, log_cls = _exporter_classes("grpc")
    assert all(".grpc." in cls.__module__ for cls in (span_cls, metric_cls, log_cls))

    span_cls, metric_cls, log_cls = _exporter_classes("http/protobuf")
    assert all(".http." in cls.__module__ for cls in (span_cls, metric_cls, log_cls))


def test_disabled_without_endpoint_leaves_default_noop_providers() -> None:
    script = (
        "from conquer3.telemetry import otel\n"
        "otel.init_telemetry('test-svc')\n"
        "from opentelemetry import trace\n"
        "print(type(trace.get_tracer_provider()).__name__)\n"
    )
    result = _run(script, otlp_endpoint=None)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "ProxyTracerProvider"


def test_disabled_get_meter_and_get_tracer_are_still_usable() -> None:
    """The no-op path must never make get_meter/get_tracer unsafe to call --
    every existing call site (state_store.py, model_registry.py, scorer.py)
    calls them unconditionally, with no settings.otel check of its own."""
    script = (
        "from conquer3.telemetry.otel import get_meter, get_tracer, init_telemetry\n"
        "init_telemetry('test-svc')\n"
        "get_meter('x').create_counter('c3_test').add(1)\n"
        "with get_tracer('x').start_as_current_span('s'):\n"
        "    pass\n"
        "print('ok')\n"
    )
    result = _run(script, otlp_endpoint=None)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "ok"


def test_enabled_installs_real_tracer_meter_and_logger_providers() -> None:
    script = (
        "from conquer3.telemetry import otel\n"
        "otel.init_telemetry('test-svc')\n"
        "from opentelemetry import metrics, trace\n"
        "from opentelemetry import _logs as logs_api\n"
        "print(type(trace.get_tracer_provider()).__name__)\n"
        "print(type(metrics.get_meter_provider()).__name__)\n"
        "print(type(logs_api.get_logger_provider()).__name__)\n"
        "print(trace.get_tracer_provider().resource.attributes['service.name'])\n"
    )
    result = _run(script, otlp_endpoint="http://127.0.0.1:4317")
    assert result.returncode == 0, result.stderr
    tracer_cls, meter_cls, logger_cls, service_name = result.stdout.strip().splitlines()[-4:]
    assert tracer_cls == "TracerProvider"
    assert meter_cls == "MeterProvider"
    assert logger_cls == "LoggerProvider"
    assert service_name == "test-svc"


def test_init_telemetry_is_idempotent_and_logging_forwards_to_otlp() -> None:
    """A second call must not install a second logging handler (or crash), and a
    plain logging.getLogger(...).info(...) call -- no OTel-specific code at the
    call site -- must reach it, which is what gives the collector's already-
    configured `logs` pipeline (-> Loki) something to receive."""
    script = (
        "import logging\n"
        "from conquer3.telemetry import otel\n"
        "otel.init_telemetry('test-svc')\n"
        "otel.init_telemetry('test-svc')\n"
        "logging.getLogger('conquer3.somewhere').info('hello')\n"
        "handlers = [\n"
        "    h for h in logging.getLogger().handlers if type(h).__name__ == 'LoggingHandler'\n"
        "]\n"
        "print(len(handlers))\n"
    )
    result = _run(script, otlp_endpoint="http://127.0.0.1:4317")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "1"
