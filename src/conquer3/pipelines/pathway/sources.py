"""Wraps pw.io.fs.read for the staging JSONL directory -- the one connector both
static backfill and streaming state repair use (architecture plan section 6, "same
connector, both modes"). `mode_override` lets each entry point pin its mode
independent of C3_PATHWAY_MODE, so `conquer3 pathway backfill` always runs static
regardless of what the env var happens to be set to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import pathway as pw

from conquer3.config.settings import EventSettings, PathwaySettings
from conquer3.pipelines.pathway.schemas import TransactionEventSchema

__all__ = ["read_staging_events"]

STAGING_SUBDIR = "ctx"


def read_staging_events(
    *,
    event_settings: EventSettings,
    pathway_settings: PathwaySettings,
    mode_override: Literal["static", "streaming"] | None = None,
) -> pw.Table[Any]:
    path = Path(event_settings.staging_dir) / STAGING_SUBDIR
    mode = mode_override if mode_override is not None else pathway_settings.mode
    return cast(
        "pw.Table[Any]",
        pw.io.fs.read(
            path,
            format="json",
            schema=TransactionEventSchema,
            mode=mode,
            autocommit_duration_ms=pathway_settings.autocommit_ms,
        ),
    )
