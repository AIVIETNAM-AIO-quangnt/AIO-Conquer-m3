"""Regression test for the DuckDB pitfall that caused
``pipelines/transforms/silver_to_gold.py``'s gold row count to silently
truncate to a single flush's worth of rows instead of the full dataset
(confirmed empirically: every full-pipeline run wrote exactly one flush's
worth of ``gold.txn_features`` rows, no error, no warning).

A DuckDB connection supports only one active result at a time. Issuing a
second statement on the same connection while a ``fetch_record_batch``/
``to_pyarrow_batches`` reader from an earlier query is still being iterated
silently ends that reader early -- it just stops yielding batches, with no
exception raised. ``_transform``'s fix is to run the write-side flush through
its own cursor (``duck.con.cursor()``), which shares the same in-process
database without competing for the connection's single result slot.

Uses a purely local, in-memory table -- no Postgres/Docker needed -- because
the hazard lives entirely in DuckDB's own connection/cursor semantics, not in
anything Postgres-specific.
"""

from __future__ import annotations

from collections.abc import Iterator

import duckdb
import pytest

_ROW_COUNT = 500_000
_CHUNK_ROWS = 50_000


def _stream_total(con: duckdb.DuckDBPyConnection, *, writer: duckdb.DuckDBPyConnection) -> int:
    """Streams ``t`` in batches, running one no-op statement on ``writer`` per
    chunk -- the same read/flush shape as silver_to_gold.py's _transform."""
    reader = con.execute("SELECT i FROM t ORDER BY i").fetch_record_batch(
        rows_per_batch=_CHUNK_ROWS
    )
    total = 0
    milestones_hit = 0
    for batch in reader:
        total += batch.num_rows
        if total // _CHUNK_ROWS > milestones_hit:
            milestones_hit += 1
            writer.execute("SELECT 1").fetchall()
    return total


@pytest.fixture
def local_table() -> Iterator[duckdb.DuckDBPyConnection]:
    con = duckdb.connect(":memory:")
    con.execute(f"CREATE TABLE t AS SELECT range AS i FROM range({_ROW_COUNT})")
    try:
        yield con
    finally:
        con.close()


def test_writing_on_the_same_connection_truncates_an_open_stream(
    local_table: duckdb.DuckDBPyConnection,
) -> None:
    """Documents the hazard _transform works around: without a dedicated
    cursor, an in-progress read silently stops early, well short of the full
    table, with no error."""
    total = _stream_total(local_table, writer=local_table)
    assert total < _ROW_COUNT


def test_writing_on_a_separate_cursor_does_not_truncate_the_stream(
    local_table: duckdb.DuckDBPyConnection,
) -> None:
    """The fix _transform actually applies: a cursor sharing the same database
    can run write statements without disturbing another still-open streaming
    read on the connection."""
    total = _stream_total(local_table, writer=local_table.cursor())
    assert total == _ROW_COUNT
