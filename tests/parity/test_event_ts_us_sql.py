"""The DuckDB SQL formula in pipelines/transforms/bronze_to_silver.py must compute
``event_ts_us`` bit-for-bit identically to conquer3.core.timeref.derive_event_ts_us --
see that module's docstring for why. No Postgres needed: this only needs the plain
integer-arithmetic expression, evaluated in an in-memory DuckDB connection.
"""

from __future__ import annotations

import duckdb
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from conquer3.core import timeref

_EXPR = f"""
CAST({timeref.SIM_EPOCH_US} AS BIGINT)
    + CAST(? - 1 AS BIGINT) * CAST({timeref.US_PER_HOUR} AS BIGINT)
    + (CAST(? - 1 AS BIGINT) * CAST({timeref.US_PER_HOUR} AS BIGINT))
      // CAST(GREATEST(?, 1) AS BIGINT)
"""


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
