"""``conquer3 serve``'s process supervisor.

Owns the two things a BentoML worker must not: contact with remote MLflow, and
the deployment audit trail.

At boot it resolves the champion (:func:`activate_champion`), records it as the
active version, then launches ``bentoml serve`` as a direct child. Every
``C3_CHAMPION_POLL_S`` seconds a background thread re-resolves. On a version
change it restarts the child so the new workers pick up the new pointer.

**The restart is a real cutover window.** ``SIGTERM`` lets BentoML drain -- requests
already accepted complete normally -- but new connections are refused for as long
as the replacement takes to boot and load the model (roughly 1-3s). This is the
one behavioural regression versus the previous MLflow implementation, whose
``SIGHUP``/``restart_all()`` path overlapped old and new workers. It buys the
removal of the entire pyfunc-wrapper rebuild (~2-4s of *every* poll tick, changed
or not), the symlink swap, and the ``C3_SCORER_WORKERS >= 2`` boot constraint that
existed only because uvicorn installs no SIGHUP handler for a single worker.

What has **not** changed is the thing Layer 6 depends on: ``on_deployment`` fires
only once the running server is *observed* serving the new version, never on
"I restarted it". An unconfirmed promotion is retried on the next poll tick
rather than being forgotten -- which is what makes ``ops.model_deployments``
(Layer 6's audit trail, read by ``dag_champion_watch``) trustworthy: a row there
means the switch was observed to actually happen, not merely attempted.
"""

from __future__ import annotations

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
from conquer3.serving.champion import activate_champion

__all__ = ["serve"]

_logger = logging.getLogger(__name__)

_SERVICE_IMPORT_PATH = "conquer3.serving.service:FraudScorerService"
# How long to wait for a SIGTERM'd child to drain before escalating to SIGKILL.
_DRAIN_TIMEOUT_S = 30.0


def serve(
    settings: Settings | None = None,
    *,
    on_deployment: Callable[[ModelRef], None] | None = None,
    stop_event: threading.Event | None = None,
) -> int:
    """Activates the champion, launches ``bentoml serve`` as a child, starts the
    champion-poll thread, and blocks until the child exits. Returns the child's
    exit code. ``on_deployment`` is called (from this thread, at boot, and from
    the poll thread on every confirmed version change) with the resolved
    ``ModelRef`` -- callers outside ``conquer3.serving`` use it to record an
    audit-trail row, since this package may never import ``conquer3.db``
    (import-linter forbids it; see ``db/ops.py``'s ``record_model_deployment``).
    """
    settings = settings or get_settings()

    ref = activate_champion(settings)
    _notify(on_deployment, ref)

    child = _spawn(settings)

    stop_event = stop_event if stop_event is not None else threading.Event()
    holder = _ChildHolder(child)
    _install_signal_forwarding(holder, stop_event)

    poller = threading.Thread(
        target=_poll_champion,
        args=(settings, holder, stop_event, on_deployment, ref.version),
        daemon=True,
        name="champion-poll",
    )
    poller.start()

    try:
        while True:
            child = holder.current
            code = child.wait()
            if stop_event.is_set():
                return code
            # A restart terminates the child deliberately. Taking the lock here
            # blocks until any in-flight restart has finished swapping in the
            # replacement, so "the child exited" can be told apart from "the
            # child was replaced" without racing the poll thread.
            with holder.lock:
                if holder.current is not child:
                    continue
            _logger.error("scoring server exited unexpectedly with code %s", code)
            return code
    finally:
        stop_event.set()
        _terminate(holder.current)


class _ChildHolder:
    """The live child, swapped under a lock on restart.

    Signal forwarding and the poll thread both need "whichever child is current
    right now"; without this they would race against a restart and signal a
    process that has already been replaced. Reentrant because the SIGTERM handler
    runs on the main thread, which may already hold the lock.
    """

    def __init__(self, child: subprocess.Popen[bytes]) -> None:
        self._child = child
        self.lock = threading.RLock()

    @property
    def current(self) -> subprocess.Popen[bytes]:
        with self.lock:
            return self._child

    def replace(self, child: subprocess.Popen[bytes]) -> None:
        with self.lock:
            self._child = child


def _spawn(settings: Settings) -> subprocess.Popen[bytes]:
    serving = settings.serving
    return subprocess.Popen(
        [
            "bentoml",
            "serve",
            _SERVICE_IMPORT_PATH,
            "--host",
            serving.scorer_host,
            "--port",
            str(serving.scorer_port),
        ]
    )


def _poll_champion(
    settings: Settings,
    holder: _ChildHolder,
    stop_event: threading.Event,
    on_deployment: Callable[[ModelRef], None] | None,
    initial_version: str,
) -> None:
    current_version = initial_version
    interval = settings.serving.champion_poll_s
    while not stop_event.wait(interval):
        if holder.current.poll() is not None:
            return  # child already exited; nothing left to reload

        try:
            ref = activate_champion(settings)
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

        _logger.info("champion %s -> %s; restarting the server", current_version, ref.version)
        if not _restart(holder, settings, stop_event):
            return

        if _wait_for_served_version(
            settings, ref.version, timeout_s=settings.serving.scorer_timeout_s
        ):
            current_version = ref.version
            _notify(on_deployment, ref)
        else:
            _logger.warning(
                "restarted for version %s but the server never confirmed serving "
                "it; will retry on the next poll tick",
                ref.version,
            )


def _restart(holder: _ChildHolder, settings: Settings, stop_event: threading.Event) -> bool:
    """Drain the current child and spawn its replacement. Returns False if the
    supervisor is shutting down and no replacement should be started.

    Holds the lock across the whole swap so the main loop cannot observe the
    intermediate state where the old child has exited and the new one does not
    exist yet.
    """
    with holder.lock:
        _terminate(holder.current)
        if stop_event.is_set():
            return False
        holder.replace(_spawn(settings))
        return True


def _terminate(child: subprocess.Popen[bytes]) -> None:
    """SIGTERM, then SIGKILL if it will not drain. BentoML shuts down gracefully
    on SIGTERM, so requests already accepted complete before the process exits."""
    if child.poll() is not None:
        return
    with suppress(ProcessLookupError):
        child.terminate()
    try:
        child.wait(timeout=_DRAIN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        _logger.warning("child did not drain within %ss; killing it", _DRAIN_TIMEOUT_S)
        with suppress(ProcessLookupError):
            child.kill()
        with suppress(subprocess.TimeoutExpired):
            child.wait(timeout=10)


def _wait_for_served_version(settings: Settings, version: str, *, timeout_s: float) -> bool:
    """Polls the actually-running server (not this process's own state) for the
    model version it reports, via a real ``/model_info`` request -- the only way
    to observe that the replacement booted and loaded the new champion, rather
    than crash-looping. Best-effort: any error (server still booting, connection
    refused) is treated as "not yet", never raised."""
    url = f"http://127.0.0.1:{settings.serving.scorer_port}/model_info"
    body = b"{}"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            request = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(request, timeout=2) as resp:
                payload = json.loads(resp.read())
            if payload["version"] == version:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _install_signal_forwarding(holder: _ChildHolder, stop_event: threading.Event) -> None:
    def _forward(signum: int, _frame: object) -> None:
        stop_event.set()
        with suppress(ProcessLookupError):
            holder.current.send_signal(signum)

    signal.signal(signal.SIGTERM, _forward)
    signal.signal(signal.SIGINT, _forward)


def _notify(on_deployment: Callable[[ModelRef], None] | None, ref: ModelRef) -> None:
    if on_deployment is None:
        return
    try:
        on_deployment(ref)
    except Exception:
        _logger.warning("on_deployment callback failed for version %s", ref.version, exc_info=True)
