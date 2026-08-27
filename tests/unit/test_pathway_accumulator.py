"""build_account_state_table's output must agree with conquer3.core.features'
pure-Python fold, for the same reason core.features.merge_states must agree with
compute_sequence (tests/unit/test_features.py's Hypothesis property) -- this file
proves that property holds through the actual Pathway reducer plumbing, not just
the pure-Python delegation it wraps.

Uses pw.debug (table_from_rows/table_to_dicts), not pw.run() -- confirmed safe to
build and tear down many independent debug graphs in one process, unlike pw.run()
which the Pathway engine only supports calling once per process.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import pathway as pw
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from conquer3.core import timeref
from conquer3.core.features import compute_sequence
from conquer3.core.serde import state_from_json
from conquer3.core.types import AccountState, TransactionEvent
from conquer3.pipelines.pathway.graph import build_account_state_table
from conquer3.pipelines.pathway.schemas import TransactionEventSchema

logging.getLogger("pathway").setLevel(logging.WARNING)


def _txn(
    seq: int, *, account_id: str, amount: float, step: int, txn_type: str = "TRANSFER"
) -> TransactionEvent:
    return TransactionEvent(
        event_id=f"e{seq:05d}",
        account_id=account_id,
        dest_id="C900",
        txn_type=txn_type,
        amount=amount,
        oldbalance_org=10_000.0,
        newbalance_orig=10_000.0 - amount,
        oldbalance_dest=0.0,
        newbalance_dest=amount,
        event_ts_us=timeref.derive_event_ts_us(step, 1, 1),
        step=step,
    )


def _to_row(txn: TransactionEvent) -> tuple:
    return (
        txn.event_id,
        txn.account_id,
        txn.dest_id,
        txn.txn_type,
        txn.amount,
        txn.oldbalance_org,
        txn.newbalance_orig,
        txn.oldbalance_dest,
        txn.newbalance_dest,
        txn.event_ts_us,
        txn.step,
    )


def _expected_final_states(events: list[TransactionEvent]) -> dict[str, AccountState]:
    """The independent Python oracle: sort each account's events, fold sequentially."""
    by_account: dict[str, list[TransactionEvent]] = defaultdict(list)
    for txn in events:
        by_account[txn.account_id].append(txn)

    expected: dict[str, AccountState] = {}
    for account_id, txns in by_account.items():
        ordered = sorted(txns, key=lambda t: (t.event_ts_us, t.event_id))
        final_state: AccountState | None = None
        for _features, state in compute_sequence(ordered):
            final_state = state
        assert final_state is not None
        expected[account_id] = final_state
    return expected


def _run_pathway(events: list[TransactionEvent]) -> dict[str, AccountState]:
    rows = [_to_row(txn) for txn in events]
    table = pw.debug.table_from_rows(schema=TransactionEventSchema, rows=rows)
    result = build_account_state_table(table)
    ids, cols = pw.debug.table_to_dicts(result)

    actual: dict[str, AccountState] = {}
    for key in ids:
        state = state_from_json(cols["state_json"][key])
        assert state is not None
        actual[cols["account_id"][key]] = state
    return actual


def _assert_states_equal(actual: AccountState, expected: AccountState) -> None:
    assert actual.last_event_id == expected.last_event_id
    assert actual.last_event_ts_us == expected.last_event_ts_us
    assert actual.last_amount == pytest.approx(expected.last_amount)
    assert actual.txn_count == expected.txn_count
    assert actual.amount_sum == pytest.approx(expected.amount_sum, rel=1e-9)
    assert actual.amount_sqsum == pytest.approx(expected.amount_sqsum, rel=1e-9)
    assert actual.max_amount == pytest.approx(expected.max_amount)
    assert actual.first_event_ts_us == expected.first_event_ts_us


def test_pathway_matches_core_features_for_a_fixed_out_of_order_example() -> None:
    events = [
        _txn(0, account_id="A", amount=100.0, step=3),  # later ts, arrives 1st
        _txn(1, account_id="A", amount=50.0, step=1),  # earlier ts, arrives 2nd
        _txn(2, account_id="A", amount=900.0, step=2, txn_type="CASH_OUT"),
        _txn(3, account_id="B", amount=200.0, step=1),
    ]

    expected = _expected_final_states(events)
    actual = _run_pathway(events)

    assert set(actual) == set(expected)
    for account_id in expected:
        _assert_states_equal(actual[account_id], expected[account_id])


_accounts = st.sampled_from(["A", "B", "C"])
_amounts = st.floats(min_value=0.0, max_value=1e7, allow_nan=False, allow_infinity=False)
_types = st.sampled_from(["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])


@st.composite
def _multi_account_event_list(draw: st.DrawFn) -> list[TransactionEvent]:
    n = draw(st.integers(min_value=1, max_value=12))
    accounts = draw(st.lists(_accounts, min_size=n, max_size=n))
    steps = draw(st.lists(st.integers(min_value=1, max_value=744), min_size=n, max_size=n))
    amounts = draw(st.lists(_amounts, min_size=n, max_size=n))
    types = draw(st.lists(_types, min_size=n, max_size=n))
    return [
        _txn(i, account_id=acc, amount=a, step=s, txn_type=t)
        for i, (acc, a, s, t) in enumerate(zip(accounts, amounts, steps, types, strict=True))
    ]


@settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(events=_multi_account_event_list())
def test_pathway_reducer_matches_core_features_fold(events: list[TransactionEvent]) -> None:
    """The property that guarantees Pathway's streaming/batch state == core's fold,
    proven through the actual reducer/groupby/udf plumbing rather than only through
    core.features.merge_states directly (tests/unit/test_features.py already proves
    that narrower property)."""
    expected = _expected_final_states(events)
    actual = _run_pathway(events)

    assert set(actual) == set(expected)
    for account_id in expected:
        _assert_states_equal(actual[account_id], expected[account_id])
