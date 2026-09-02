"""The vectorized pandas formula in producer/replay.py must compute
``event_ts_us`` bit-for-bit identically to conquer3.core.timeref.derive_event_ts_us --
see that module's docstring for why, and tests/parity/test_event_ts_us_sql.py for
the same check against the DuckDB SQL reproduction.

Calls ``producer.replay._derive_event_ts_us_vectorized`` directly -- the actual
function ``to_transactions_frame`` uses, not a copy -- so a formula edit there
that breaks parity fails here, rather than silently drifting.
"""

from __future__ import annotations

import random

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from conquer3.core import timeref
from conquer3.producer.replay import _derive_event_ts_us_vectorized as _pandas_derive


@given(
    step=st.integers(min_value=1, max_value=timeref.MAX_STEP),
    card=st.integers(min_value=1, max_value=10_000),
    data=st.data(),
)
@settings(max_examples=200, deadline=None)
def test_pandas_matches_python_derivation(step: int, card: int, data: st.DataObject) -> None:
    seq = data.draw(st.integers(min_value=1, max_value=card))
    result = _pandas_derive(pd.Series([step]), pd.Series([seq]), pd.Series([card])).iloc[0]
    assert int(result) == timeref.derive_event_ts_us(step, seq, card)


def test_pandas_matches_python_at_step_boundaries() -> None:
    cases = [(1, 1, 1), (1, 1, 8500), (1, 8500, 8500), (744, 1, 1), (500, 4250, 8500)]
    for step, seq, card in cases:
        result = _pandas_derive(pd.Series([step]), pd.Series([seq]), pd.Series([card])).iloc[0]
        assert int(result) == timeref.derive_event_ts_us(step, seq, card)


@pytest.mark.slow
def test_pandas_matches_python_exactly_across_100k_rows() -> None:
    """Same exact-equality sweep test_event_ts_us_sql.py runs against DuckDB,
    against the pandas/numpy reproduction instead -- one vectorized call, not
    100k round trips."""
    rng = random.Random(20260827)
    n = 100_000
    cards = [rng.randint(1, 20_000) for _ in range(n)]
    steps = [rng.randint(1, timeref.MAX_STEP) for _ in range(n)]
    seqs = [rng.randint(1, card) for card in cards]

    result = _pandas_derive(pd.Series(steps), pd.Series(seqs), pd.Series(cards))
    expected = [
        timeref.derive_event_ts_us(s, q, c) for s, q, c in zip(steps, seqs, cards, strict=True)
    ]

    mismatches = [
        (steps[i], seqs[i], cards[i], int(result.iloc[i]))
        for i in range(n)
        if int(result.iloc[i]) != expected[i]
    ]
    assert not mismatches, f"{len(mismatches)} pandas/Python mismatches, e.g. {mismatches[:5]}"
