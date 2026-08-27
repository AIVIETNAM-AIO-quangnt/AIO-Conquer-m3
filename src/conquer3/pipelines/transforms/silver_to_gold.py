"""Silver -> gold: the actual feature computation.

This is the **only** place that calls ``conquer3.core.features`` outside serving --
per that module's docstring, feature values must never be computed anywhere else
(not in SQL, not here as a reimplementation) or training/serving skew comes back.
So unlike ``bronze_to_silver``, this transform reads ``silver.txn`` as a streamed,
account-ordered Ibis/DuckDB batch reader, rebuilds each row as a
``TransactionEvent``, and folds it through ``core.features.compute_sequence`` in
Python -- exactly the batch/training path that module already documents.

``is_fraud``/``is_flagged_fraud`` are carried alongside from the same silver row and
attached to the output row directly -- never passed into ``TransactionEvent`` or
``core.features``, which must never see labels.

Streams rather than materializing all of ``silver.txn`` at once: at PaySim1's full
6.3M rows, holding every row as a Python object simultaneously is several GB of
avoidable overhead. Flushes to ``gold.txn_features`` every ``flush_every`` rows.

Full refresh: TRUNCATEs ``gold.txn_features`` before reloading.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import pyarrow as pa

from conquer3.core.features import compute_sequence
from conquer3.core.schema import CATEGORICAL_FEATURES, FEATURE_NAMES, NUMERIC_FEATURES
from conquer3.core.serde import features_to_row
from conquer3.core.types import TransactionEvent
from conquer3.db import ops
from conquer3.db.engine import get_ibis_connection, pg_connection

__all__ = ["silver_to_gold"]

_READ_SQL = """
SELECT event_id, account_id, dest_id, txn_type, amount,
       oldbalance_org, newbalance_orig, oldbalance_dest, newbalance_dest,
       step, event_ts_us, is_fraud, is_flagged_fraud
FROM pg.silver.txn
ORDER BY account_id, event_ts_us, event_id
"""

_READ_CHUNK_ROWS = 50_000
_DEFAULT_FLUSH_EVERY = 200_000

_GOLD_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string()),
        pa.field("account_id", pa.string()),
        pa.field("event_ts_us", pa.int64()),
        pa.field("feature_schema_version", pa.int32()),
        *(
            pa.field(name, pa.float64() if name in NUMERIC_FEATURES else pa.string())
            for name in FEATURE_NAMES
        ),
        pa.field("is_fraud", pa.bool_()),
        pa.field("is_flagged_fraud", pa.bool_()),
    ]
)

assert set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES) == set(FEATURE_NAMES)


def silver_to_gold(*, flush_every: int = _DEFAULT_FLUSH_EVERY) -> int:
    """TRUNCATEs ``gold.txn_features``, then computes and writes every feature row.

    Returns the row count written.
    """
    with pg_connection() as conn, ops.track_run(conn, layer="silver_to_gold") as run:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM silver.txn")
            row = cur.fetchone()
            assert row is not None
            run.rows_in = row[0]
            cur.execute("TRUNCATE gold.txn_features")

        duck = get_ibis_connection()
        try:
            rows_out = _transform(duck, flush_every=flush_every)
        finally:
            duck.disconnect()

        run.rows_out = rows_out

    return rows_out


def _transform(duck: Any, *, flush_every: int) -> int:
    reader = duck.sql(_READ_SQL).to_pyarrow_batches(chunk_size=_READ_CHUNK_ROWS)
    rows = _flatten(reader)

    buffer: list[dict[str, Any]] = []
    total = 0
    for group in _account_groups(rows):
        buffer.extend(_compute_gold_rows(group))
        if len(buffer) >= flush_every:
            total += _flush(duck, buffer)
            buffer = []
    if buffer:
        total += _flush(duck, buffer)
    return total


def _flatten(reader: pa.RecordBatchReader) -> Iterator[dict[str, Any]]:
    for batch in reader:
        yield from batch.to_pylist()


def _account_groups(rows: Iterable[dict[str, Any]]) -> Iterator[list[dict[str, Any]]]:
    """Groups a stream already ordered by account_id into contiguous per-account lists.

    Ordinary ``itertools.groupby`` would silently split a group across two calls if
    used per-batch; this operates on the flattened row stream instead, so an
    account's rows stay together even when they straddle a batch boundary.
    """
    current_account: str | None = None
    current: list[dict[str, Any]] = []
    for row in rows:
        if row["account_id"] != current_account:
            if current:
                yield current
            current = []
            current_account = row["account_id"]
        current.append(row)
    if current:
        yield current


def _compute_gold_rows(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    txns = [
        TransactionEvent(
            event_id=r["event_id"],
            account_id=r["account_id"],
            dest_id=r["dest_id"],
            txn_type=r["txn_type"],
            amount=r["amount"],
            oldbalance_org=r["oldbalance_org"],
            newbalance_orig=r["newbalance_orig"],
            oldbalance_dest=r["oldbalance_dest"],
            newbalance_dest=r["newbalance_dest"],
            event_ts_us=r["event_ts_us"],
            step=r["step"],
        )
        for r in group
    ]
    labels = {r["event_id"]: (r["is_fraud"], r["is_flagged_fraud"]) for r in group}

    out = []
    for features, _state in compute_sequence(txns):
        gold_row = features_to_row(features)
        is_fraud, is_flagged_fraud = labels[features.event_id]
        gold_row["is_fraud"] = is_fraud
        gold_row["is_flagged_fraud"] = is_flagged_fraud
        out.append(gold_row)
    return out


def _flush(duck: Any, rows: list[dict[str, Any]]) -> int:
    table = pa.Table.from_pylist(rows, schema=_GOLD_SCHEMA)
    raw = duck.con
    raw.register("_stage_gold", table)
    try:
        raw.execute("INSERT INTO pg.gold.txn_features SELECT * FROM _stage_gold")
    finally:
        raw.unregister("_stage_gold")
    return len(rows)
