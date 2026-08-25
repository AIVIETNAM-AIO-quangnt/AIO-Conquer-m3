"""step -> timestamp determinism.

This formula is reproduced in DuckDB SQL and in pandas. Any divergence reorders tied
transactions and silently changes what "the previous transaction" is, so the
properties here are checked exactly, never approximately.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from conquer3.core import timeref

steps = st.integers(min_value=1, max_value=timeref.MAX_STEP)


def test_first_row_of_first_step_is_the_epoch() -> None:
    assert timeref.derive_event_ts_us(1, 1, 1) == timeref.SIM_EPOCH_US


def test_steps_are_exactly_one_hour_apart() -> None:
    a = timeref.derive_event_ts_us(5, 1, 1)
    b = timeref.derive_event_ts_us(6, 1, 1)
    assert b - a == timeref.US_PER_HOUR


def test_rows_within_a_step_are_spread_across_its_hour() -> None:
    stamps = [timeref.derive_event_ts_us(3, i, 4) for i in range(1, 5)]
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == 4, "ties would make window ordering non-deterministic"
    assert stamps[-1] - stamps[0] < timeref.US_PER_HOUR


@given(step=steps, cardinality=st.integers(min_value=1, max_value=50_000))
def test_a_step_never_bleeds_into_the_next(step: int, cardinality: int) -> None:
    lower = timeref.derive_event_ts_us(step, 1, cardinality)
    upper = timeref.derive_event_ts_us(step, cardinality, cardinality)
    assert lower <= upper
    assert upper < lower + timeref.US_PER_HOUR


@given(step=steps, seq=st.integers(min_value=1, max_value=1000))
def test_step_roundtrips_through_the_timestamp(step: int, seq: int) -> None:
    cardinality = max(seq, 1)
    ts = timeref.derive_event_ts_us(step, seq, cardinality)
    assert timeref.derive_step_from_ts_us(ts) == step


@given(step=steps, seq=st.integers(min_value=1, max_value=5000))
def test_result_is_an_exact_integer(step: int, seq: int) -> None:
    """Float arithmetic here would diverge from DuckDB in the last ulp."""
    value = timeref.derive_event_ts_us(step, seq, 5000)
    assert isinstance(value, int)


@pytest.mark.parametrize(
    ("step", "day", "hour", "dow"),
    [(1, 0, 0, 0), (24, 0, 23, 0), (25, 1, 0, 1), (744, 30, 23, 2)],
)
def test_calendar_decomposition(step: int, day: int, hour: int, dow: int) -> None:
    assert timeref.sim_day(step) == day
    assert timeref.hour_of_day(step) == hour
    assert timeref.day_of_week(step) == dow


@pytest.mark.parametrize(("step", "seq"), [(0, 1), (-1, 1), (1, 0)])
def test_one_based_inputs_are_enforced(step: int, seq: int) -> None:
    with pytest.raises(ValueError, match="1-based"):
        timeref.derive_event_ts_us(step, seq, 1)
