"""silver.txn -> JSONL staging for Pathway. Both static backfill and streaming state
repair read this same directory (pipelines/pathway/sources.py) -- "same connector,
both modes" (architecture plan, section 6).

Full refresh, atomic swap: writes a brand-new ctx/ directory alongside the old one,
then renames it into place, so a concurrently-running streaming Pathway process
(watching the same path) never observes a half-written file. Matches the
TRUNCATE-then-reload convention silver_to_gold.py established, applied to a
filesystem directory instead of a Postgres table.

Row order within the export does NOT matter for correctness -- unlike
silver_to_gold.py's Python loop, Pathway's accumulator merge is associative, so it
never needs pre-sorted input. ``ORDER BY bronze_row_num`` is kept only for
deterministic, reproducible part-file assignment across runs, not correctness.

Known limitation (accepted for this layer, revisit at Layer 6): the swap briefly
removes ctx/ before the new directory lands in its place (two renames, not one) --
a streaming reader polling at exactly that instant could see a missing directory
for one poll cycle. Pathway's fs connector is expected to tolerate a transient
missing path (it already has to tolerate the directory not existing before the
first backfill ever runs); if this proves not robust in practice, switch to an
add-then-prune scheme (write new part files under an incrementing generation
prefix, delete the previous generation's files only after the new ones are
confirmed on disk) instead of remove-then-add.

Never includes is_fraud/is_flagged_fraud or any other label -- these rows feed
directly into conquer3.core.types.TransactionEvent's constructor, and core must
never see labels (core/schema.py's FORBIDDEN_FEATURE_SOURCES).
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa

from conquer3.core.types import TransactionEvent
from conquer3.db import ops
from conquer3.db.engine import get_ibis_connection, pg_connection

__all__ = ["STAGING_SUBDIR", "export_staging"]

STAGING_SUBDIR = "ctx"
_READ_CHUNK_ROWS = 50_000
_ROWS_PER_FILE = 500_000
_FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in dataclasses.fields(TransactionEvent))

_READ_SQL = """
SELECT event_id, account_id, dest_id, txn_type, amount,
       oldbalance_org, newbalance_orig, oldbalance_dest, newbalance_dest,
       step, event_ts_us
FROM pg.silver.txn
ORDER BY bronze_row_num
"""


def export_staging(*, staging_dir: str | None = None) -> int:
    """Rewrites ``{staging_dir}/ctx/`` from ``silver.txn``. Returns rows written."""
    from conquer3.config.settings import get_settings

    base_dir = Path(staging_dir if staging_dir is not None else get_settings().event.staging_dir)

    with pg_connection() as conn, ops.track_run(conn, layer="export_staging") as run:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM silver.txn")
            row = cur.fetchone()
            assert row is not None
            run.rows_in = row[0]

        duck = get_ibis_connection()
        try:
            rows_out = _write_staging(duck, base_dir)
        finally:
            duck.disconnect()

        run.rows_out = rows_out
        run.detail = f"wrote {base_dir / STAGING_SUBDIR}"

    return rows_out


def _write_staging(duck: Any, base_dir: Path) -> int:
    new_dir = base_dir / f"{STAGING_SUBDIR}.new-{uuid.uuid4().hex}"
    new_dir.mkdir(parents=True, exist_ok=True)

    reader = duck.sql(_READ_SQL).to_pyarrow_batches(chunk_size=_READ_CHUNK_ROWS)
    total = 0
    file_index = 0
    buffer: list[dict[str, Any]] = []
    for row in _flatten(reader):
        buffer.append(row)
        if len(buffer) >= _ROWS_PER_FILE:
            _write_part_file(new_dir, file_index, buffer)
            total += len(buffer)
            file_index += 1
            buffer = []
    if buffer:
        _write_part_file(new_dir, file_index, buffer)
        total += len(buffer)

    _swap_in(base_dir, new_dir)
    return total


def _flatten(reader: pa.RecordBatchReader) -> Iterator[dict[str, Any]]:
    for batch in reader:
        yield from batch.to_pylist()


def _write_part_file(new_dir: Path, index: int, rows: list[dict[str, Any]]) -> None:
    path = new_dir / f"part-{index:05d}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({name: row[name] for name in _FIELD_NAMES}, separators=(",", ":")))
            fh.write("\n")


def _swap_in(base_dir: Path, new_dir: Path) -> None:
    target = base_dir / STAGING_SUBDIR
    if target.exists():
        old_dir = base_dir / f"{STAGING_SUBDIR}.old-{uuid.uuid4().hex}"
        target.replace(old_dir)
        new_dir.replace(target)
        shutil.rmtree(old_dir)
    else:
        new_dir.replace(target)
