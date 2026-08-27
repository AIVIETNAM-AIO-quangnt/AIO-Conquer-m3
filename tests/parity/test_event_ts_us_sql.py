"""The DuckDB SQL formula in pipelines/transforms/bronze_to_silver.py must compute
``event_ts_us`` bit-for-bit identically to conquer3.core.timeref.derive_event_ts_us --
see that module's docstring for why. No Postgres needed: this only needs the plain
integer-arithmetic expression, evaluated in an in-memory DuckDB connection.
"""

from __future__ import annotations

import random

import duckdb
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from conquer3.core import timeref


def _expr(step_col: str, seq_col: str, card_col: str) -> str:
    """The DuckDB arithmetic under test, parameterised by column/placeholder text.

    Single source for both the single-row spot checks (called with ``"?"``
    placeholders below) and the 100k-row sweep (called with real column names).
    """
    return f"""
CAST({timeref.SIM_EPOCH_US} AS BIGINT)
    + CAST({step_col} - 1 AS BIGINT) * CAST({timeref.US_PER_HOUR} AS BIGINT)
    + (CAST({seq_col} - 1 AS BIGINT) * CAST({timeref.US_PER_HOUR} AS BIGINT))
      // CAST(GREATEST({card_col}, 1) AS BIGINT)
"""


_EXPR = _expr("?", "?", "?")


@pytest.fixture(scope="module")
def con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(":memory:")


def _sql_derive(con: duckdb.DuckDBPyConnection, step: int, seq: int, card: int) -> int:
    row = con.execute(f"SELECT {_EXPR}", [step, seq, card]).fetchone()
    assert row is not None
    return int(row[0])


@given(
    step=st.integers(min_value=1, max_value=timeref.MAX_STEP),
    card=st.integers(min_value=1, max_value=10_000),
    data=st.data(),
)
@settings(max_examples=200, deadline=None)
def test_sql_matches_python_derivation(
    con: duckdb.DuckDBPyConnection, step: int, card: int, data: st.DataObject
) -> None:
    seq = data.draw(st.integers(min_value=1, max_value=card))
    assert _sql_derive(con, step, seq, card) == timeref.derive_event_ts_us(step, seq, card)


def test_sql_matches_python_at_step_boundaries(con: duckdb.DuckDBPyConnection) -> None:
    cases = [(1, 1, 1), (1, 1, 8500), (1, 8500, 8500), (744, 1, 1), (500, 4250, 8500)]
    for step, seq, card in cases:
        assert _sql_derive(con, step, seq, card) == timeref.derive_event_ts_us(step, seq, card)


@pytest.mark.slow
def test_sql_matches_python_exactly_across_100k_rows(con: duckdb.DuckDBPyConnection) -> None:
    """The Layer 3 gate (plan section 12) calls for an *exact-equality* sweep over
    100k rows, not a sampled property test -- Hypothesis's 200 examples above check
    the property holds, this checks it holds at the gate's specified scale, with
    zero tolerance for mismatch. One batched DuckDB query, not 100k round trips.
    """
    rng = random.Random(20260827)
    n = 100_000
    # step_cardinality up to 20_000 covers PaySim's ~8,500-row average steps with
    # headroom; seq is always drawn in-range for its own cardinality.
    cards = [rng.randint(1, 20_000) for _ in range(n)]
    steps = [rng.randint(1, timeref.MAX_STEP) for _ in range(n)]
    seqs = [rng.randint(1, card) for card in cards]

    sql = f"""
        WITH t AS (
            SELECT unnest(?) AS step, unnest(?) AS seq, unnest(?) AS card
        )
        SELECT step, seq, card, {_expr("step", "seq", "card")} AS ts_us
        FROM t
    """
    rows = con.execute(sql, [steps, seqs, cards]).fetchall()
    assert len(rows) == n

    mismatches = [
        (step, seq, card, sql_ts_us)
        for step, seq, card, sql_ts_us in rows
        if sql_ts_us != timeref.derive_event_ts_us(step, seq, card)
    ]
    assert not mismatches, f"{len(mismatches)} SQL/Python mismatches, e.g. {mismatches[:5]}"
