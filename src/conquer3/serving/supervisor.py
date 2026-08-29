"""``conquer3 serve``'s process supervisor.

Launches MLflow's own scoring server as a **direct child**, using
``mlflow.pyfunc.scoring_server.get_cmd`` -- the same internal seam
``mlflow models serve`` itself calls one process deeper (verified by reading
``mlflow.pyfunc.backend.PyFuncBackend.serve``, ``mlflow==3.15.1``) -- so this
process owns the uvicorn master PID directly instead of hunting for a grandchild.
That reused seam is also where the ``bash -c "exec ..."`` launch pattern below
comes from: without ``exec``, the child would be a shell wrapping uvicorn, and
this process's PID would belong to bash, not to the process that actually
understands ``SIGHUP``.

Every ``C3_CHAMPION_POLL_S`` seconds, a background thread re-runs
:func:`build_and_activate_champion`. On a version change it ``SIGHUP``s the
child; uvicorn's own multiprocess supervisor turns that into a graceful
``restart_all()`` that **aborts the restart and keeps the old workers** if the
replacement fails to boot -- strictly safer than a hand-rolled reload endpoint,
and the reason no ``/admin/reload`` route exists anywhere in this layer.

``restart_all()`` is asynchronous, best-effort, and has no callback: uvicorn
never tells this process whether a given ``SIGHUP`` actually landed. Confirmed
empirically (aborted restarts under host load during this layer's own test
suite -- ``ERROR: New child process was not ready in time; keeping worker and
aborting the restart`` in uvicorn's own log): an aborted restart is silent from
here. So the poll loop does not trust "I sent SIGHUP" as "the new version is
live" -- it probes the running server for the version it is actually serving
and only advances its own tracked version (and only calls ``on_deployment``)
once that is confirmed. An unconfirmed promotion is retried on the *next* poll
tick rather than being forgotten -- which is what makes ``ops.model_deployments``
(Layer 6's audit trail, written from ``on_deployment``) trustworthy: a row there
means the switch was observed to actually happen, not merely attempted.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import signal
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable
from contextlib import suppress

from conquer3.config.settings import Settings, get_settings
from conquer3.contracts.model_registry import ModelRef
from conquer3.core.types import TransactionEvent
from conquer3.serving.build import build_and_activate_champion
from conquer3.serving.signature import TXN_FIELD_NAMES

__all__ = ["serve"]

_logger = logging.getLogger(__name__)


def serve(
    settings: Settings | None = None,
    *,
    on_deployment: Callable[[ModelRef], None] | None = None,
    stop_event: threading.Event | None = None,
) -> int:
    """Builds + activates the champion, launches the scoring server as a child,
    starts the champion-poll thread, and blocks until the child exits. Returns
    the child's exit code. ``on_deployment`` is called (from this thread, at
    boot, and from the poll thread on every version change) with the resolved
    ``ModelRef`` -- callers outside ``conquer3.serving`` use it to record an
    audit-trail row, since this package may never import ``conquer3.db``
    (import-linter forbids it; see ``db/ops.py``'s ``record_model_deployment``).
    """
    settings = settings or get_settings()
    serving = settings.serving

    # uvicorn only installs its SIGHUP-handling Multiprocess supervisor when
    # workers > 1 (confirmed by reading uvicorn.main, uvicorn==0.52.4): with
    # exactly one worker it runs the bare Server with no signal handling of its
    # own, so SIGHUP falls through to the OS default (terminate) instead of
    # reloading. Confirmed empirically -- C3_SCORER_WORKERS=1 does not "skip
    # reloads", it kills the whole scorer on the first champion promotion.
    if serving.scorer_workers < 2:
        raise ValueError(
            f"C3_SCORER_WORKERS={serving.scorer_workers} but must be >= 2: with "
            "exactly one worker, uvicorn installs no SIGHUP handler at all, so "
            "the champion-poll thread's reload signal terminates the whole "
            "scorer instead of reloading it."
        )

    ref = build_and_activate_champion(settings)
    _notify(on_deployment, ref)

    import mlflow.pyfunc.scoring_server as scoring_server

    command, env = scoring_server.get_cmd(
        model_uri=serving.current_model_symlink,
        port=serving.scorer_port,
        # get_cmd's own type hint says `host: int | None`, but its body does
        # `shlex.quote(host)` -- confirmed by reading the source (mlflow==3.15.1)
        # that this is an upstream stub bug, not a real `int` expectation.
        host=serving.scorer_host,  # type: ignore[arg-type]
        timeout=serving.scorer_timeout_s,
        nworkers=serving.scorer_workers,
    )
    child = subprocess.Popen(["bash", "-c", "exec " + command], env=env)

    stop_event = stop_event if stop_event is not None else threading.Event()
    _install_signal_forwarding(child, stop_event)

    poller = threading.Thread(
        target=_poll_champion,
        args=(settings, child, stop_event, on_deployment, ref.version),
        daemon=True,
        name="champion-poll",
    )
    poller.start()

    try:
        return child.wait()
    finally:
        stop_event.set()


def _poll_champion(
    settings: Settings,
    child: subprocess.Popen[bytes],
    stop_event: threading.Event,
    on_deployment: Callable[[ModelRef], None] | None,
    initial_version: str,
) -> None:
    current_version = initial_version
    interval = settings.serving.champion_poll_s
    while not stop_event.wait(interval):
        if child.poll() is not None:
            return  # child already exited; nothing left to reload

        try:
            ref = build_and_activate_champion(settings)
        except Exception:
            # A dead/unreachable remote MLflow here delays only the *next*
            # promotion. It must never take down an already-running scorer --
            # that property is what the Layer 5 gate's "not-a-proxy" test checks.
            _logger.warning(
                "champion poll failed; keeping version %s", current_version, exc_info=True
            )
            continue

        if ref.version == current_version:
            continue
        try:
            child.send_signal(signal.SIGHUP)
        except ProcessLookupError:
            return

        if _wait_for_served_version(
            settings, ref.version, timeout_s=settings.serving.scorer_timeout_s
        ):
            current_version = ref.version
            _notify(on_deployment, ref)
        else:
            _logger.warning(
                "SIGHUP sent for version %s but the server never confirmed "
                "serving it (uvicorn likely aborted the restart under load, "
                "keeping the old workers -- see this module's docstring); "
                "will retry on the next poll tick",
                ref.version,
            )


# A syntactically valid (semantically ignored) row for the op=model_info probe
# below -- MLflow enforces the input schema before predict() runs regardless of
# op, so even a pure metadata query needs one (see serving/signature.py).
_FIELD_TYPES: dict[str, str] = {
    f.name: f.type  # type: ignore[misc]
    for f in dataclasses.fields(TransactionEvent)
}
_PLACEHOLDER_BY_TYPE: dict[str, object] = {"str": "", "float": 0.0, "int": 0}
_PROBE_ROW = {name: _PLACEHOLDER_BY_TYPE[_FIELD_TYPES[name]] for name in TXN_FIELD_NAMES}


def _wait_for_served_version(settings: Settings, version: str, *, timeout_s: float) -> bool:
    """Polls the actually-running scoring server (not this process's own state)
    for the model_version it reports, via a real op=model_info request -- the
    only way to observe whether restart_all() actually switched over, since
    uvicorn gives this process no other signal. Best-effort: any error (server
    mid-restart, connection refused) is treated as "not yet", never raised."""
    url = f"http://127.0.0.1:{settings.serving.scorer_port}/invocations"
    body = json.dumps({"dataframe_records": [_PROBE_ROW], "params": {"op": "model_info"}}).encode(
        "utf-8"
    )
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            request = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(request, timeout=2) as resp:
                payload = json.loads(resp.read())
            if payload["predictions"][0]["version"] == version:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _install_signal_forwarding(child: subprocess.Popen[bytes], stop_event: threading.Event) -> None:
    def _forward(signum: int, _frame: object) -> None:
        stop_event.set()
        with suppress(ProcessLookupError):
            child.send_signal(signum)

    signal.signal(signal.SIGTERM, _forward)
    signal.signal(signal.SIGINT, _forward)


def _notify(on_deployment: Callable[[ModelRef], None] | None, ref: ModelRef) -> None:
    if on_deployment is None:
        return
    try:
        on_deployment(ref)
    except Exception:
        _logger.warning("on_deployment callback failed for version %s", ref.version, exc_info=True)
