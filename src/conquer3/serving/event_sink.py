"""Per-worker-process JSONL event sink.

Reuses ``contracts.events`` wholesale (``ScoredEvent``, the hour-partitioned path
layout, the ``_SUCCESS`` marker convention) rather than re-deriving any of it --
that module is stdlib-only specifically so both sides (this sink, and Layer 6's
ingest DAG) can agree on the layout without sharing a heavyweight dependency.

Uvicorn workers are OS processes, so ``pid`` alone disambiguates one worker's file
from another's -- the file layout's ``worker_id`` token is always 0 here (a BentoML
artifact from when workers were something else; see plan §8.6). Within one worker,
concurrent request threads share a single fd: appends write via one ``os.write()``
of the complete line, which Linux guarantees is atomic for an ``O_APPEND`` fd even
under concurrent writers, so thread interleaving can never truncate a line. Fsync
runs on a timer, not per event; rotating to a new hour closes the old fd, drops
``_SUCCESS`` in the directory that just closed, and opens the new one.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from pathlib import Path

from conquer3.config.settings import EventSettings
from conquer3.contracts.events import SUCCESS_MARKER, ScoredEvent, event_file_relpath

__all__ = ["JsonlEventSink"]


class JsonlEventSink:
    def __init__(self, *, event_settings: EventSettings) -> None:
        self._root = Path(event_settings.dir)
        self._fsync_interval_s = event_settings.fsync_interval_ms / 1000
        self._hostname = socket.gethostname()
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._fd: int | None = None
        self._current_relpath: str | None = None
        self._last_event_ts_us: int | None = None
        self._last_fsync = 0.0

    def append(self, event: ScoredEvent) -> None:
        relpath = event_file_relpath(
            event.scored_at_us, hostname=self._hostname, pid=self._pid, worker_id=0
        )
        line = event.to_json_line().encode("utf-8")
        with self._lock:
            if relpath != self._current_relpath:
                self._rotate(relpath)
            assert self._fd is not None
            os.write(self._fd, line)
            self._last_event_ts_us = event.scored_at_us
            now = time.monotonic()
            if now - self._last_fsync >= self._fsync_interval_s:
                os.fsync(self._fd)
                self._last_fsync = now

    def _rotate(self, new_relpath: str) -> None:
        if self._fd is not None:
            os.fsync(self._fd)
            os.close(self._fd)
            assert self._current_relpath is not None
            closing_dir = (self._root / self._current_relpath).parent
            (closing_dir / SUCCESS_MARKER).touch(exist_ok=True)

        path = self._root / new_relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        self._current_relpath = new_relpath
        self._last_fsync = time.monotonic()

    def close(self) -> None:
        """Called on worker shutdown -- fsyncs whatever the timer hasn't yet, but
        deliberately does NOT drop a `_SUCCESS` marker: the hour this worker was
        writing to may still be open on other workers."""
        with self._lock:
            if self._fd is not None:
                os.fsync(self._fd)
                os.close(self._fd)
                self._fd = None
                self._current_relpath = None
