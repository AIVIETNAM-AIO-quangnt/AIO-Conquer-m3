"""Feature-core correctness: golden vectors, cold-start policy, and merge associativity."""

from __future__ import annotations

import functools
import itertools
import math
import random

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from conquer3.core import timeref
from conquer3.core.features import (
    advance_state,
    compute_features,
    compute_sequence,
    merge_states,
)
from conquer3.core.schema import (
    CATEGORICAL_FEATURES,
    COLD_START_NULL_FEATURES,
    FEATURE_NAMES,
    NO_PREV_CATEGORY,
    NUMERIC_FEATURES,
)
from conquer3.core.schema import validate as validate_schema
from conquer3.core.types import AccountState, TransactionEvent

ACCOUNT = "C1000"


def txn(
    seq: int,
    *,
    amount: float,
    step: int,
    txn_type: str = "TRANSFER",
    dest: str = "C900",
    oldbal: float = 10_000.0,
    newbal: float | None = None,
    olddest: float = 0.0,
    newdest: float | None = None,
) -> TransactionEvent:
    return TransactionEvent(
        event_id=f"e{seq:04d}",
        account_id=ACCOUNT,
        dest_id=dest,
        txn_type=txn_type,
        amount=amount,
        oldbalance_org=oldbal,
        newbalance_orig=oldbal - amount if newbal is None else newbal,
        oldbalance_dest=olddest,
        newbalance_dest=olddest + amount if newdest is None else newdest,
        event_ts_us=timeref.derive_event_ts_us(step, 1, 1),
        step=step,
    )


def test_schema_invariants_hold() -> None:
    validate_schema()


def test_feature_vector_covers_exactly_the_declared_schema() -> None:
    fv = compute_features(txn(1, amount=100.0, step=1), None)
    assert set(fv.values) == set(FEATURE_NAMES)
    assert list(fv.model_inputs()) == list(FEATURE_NAMES)


class TestColdStart:
    """The first transaction of an account has no window; that must be explicit."""

    def test_undefined_window_features_are_null_never_sentinels(self) -> None:
        fv = compute_features(txn(1, amount=100.0, step=1), None)
        for name in COLD_START_NULL_FEATURES:
            value = fv.values[name]
            if name in CATEGORICAL_FEATURES:
                # type_pair is composed from prev_txn_type, so it starts with the marker.
                assert isinstance(value, str) and value.startswith(NO_PREV_CATEGORY), name
            else:
                assert value is None, f"{name} should be None on cold start, got {value!r}"

    def test_counts_and_markers_are_defined_on_cold_start(self) -> None:
        fv = compute_features(txn(1, amount=100.0, step=1), None)
        assert fv.values["is_first_txn"] == 1
        assert fv.values["txn_count_prior"] == 0
        # Transaction-intrinsic features never depend on history.
        assert fv.values["amount"] == 100.0
        assert fv.values["txn_type"] == "TRANSFER"

    def test_second_transaction_populates_the_window(self) -> None:
        first = txn(1, amount=100.0, step=1)
        state = advance_state(first, None)
        fv = compute_features(txn(2, amount=300.0, step=3), state)

        assert fv.values["is_first_txn"] == 0
        assert fv.values["seconds_since_last_txn"] == pytest.approx(2 * 3600.0)
        assert fv.values["steps_since_last_txn"] == 2
        assert fv.values["amount_delta_vs_last"] == pytest.approx(200.0)
        assert fv.values["amount_ratio_vs_last"] == pytest.approx(3.0)
        assert fv.values["txn_count_prior"] == 1


class TestGoldenVectors:
    """Hand-computed values for a three-transaction account."""

    def test_z_score_uses_priors_excluding_the_current_row(self) -> None:
        # amounts 100 then 200 -> mean 150, population std 50.
        seq = [
            txn(1, amount=100.0, step=1),
            txn(2, amount=200.0, step=2, txn_type="CASH_OUT"),
            txn(3, amount=900.0, step=4),
        ]
        results = list(compute_sequence(seq))
        third = results[2][0].values
        assert third["amount_ratio_vs_prior_mean"] == pytest.approx(900.0 / 150.0)
        assert third["amount_ratio_vs_prior_max"] == pytest.approx(900.0 / 200.0)
        assert third["amount_z_vs_prior"] == pytest.approx((900.0 - 150.0) / 50.0)

    def test_z_score_needs_two_priors(self) -> None:
        """With one prior the population std is 0; a z-score would be meaningless."""
        state = advance_state(txn(1, amount=100.0, step=1), None)
        fv = compute_features(txn(2, amount=500.0, step=2), state)
        assert fv.values["amount_z_vs_prior"] is None

    def test_type_transition_features(self) -> None:
        state = advance_state(txn(1, amount=100.0, step=1, txn_type="PAYMENT"), None)
        fv = compute_features(txn(2, amount=100.0, step=2, txn_type="CASH_OUT"), state)
        assert fv.values["prev_txn_type"] == "PAYMENT"
        assert fv.values["type_changed"] == 1
        assert fv.values["type_pair"] == "PAYMENT->CASH_OUT"
        assert fv.values["is_fraud_capable_type"] == 1

    def test_balance_gap_detects_unobserved_activity(self) -> None:
        """The account's balance moved between our two observations."""
        first = txn(1, amount=100.0, step=1, oldbal=1000.0, newbal=900.0)
        state = advance_state(first, None)
        # Next transaction starts from 950, not the 900 we last saw.
        fv = compute_features(txn(2, amount=50.0, step=2, oldbal=950.0), state)
        assert fv.values["balance_gap_org"] == pytest.approx(50.0)
        assert fv.values["balance_gap_flag"] == 1

    def test_balance_gap_is_zero_when_state_is_consistent(self) -> None:
        first = txn(1, amount=100.0, step=1, oldbal=1000.0, newbal=900.0)
        state = advance_state(first, None)
        fv = compute_features(txn(2, amount=50.0, step=2, oldbal=900.0), state)
        assert fv.values["balance_gap_org"] == pytest.approx(0.0)
        assert fv.values["balance_gap_flag"] == 0

    def test_drain_and_merchant_flags(self) -> None:
        fv = compute_features(
            txn(1, amount=1000.0, step=1, dest="M123", oldbal=1000.0, newbal=0.0), None
        )
        assert fv.values["drains_account"] == 1
        assert fv.values["dest_is_merchant"] == 1
        assert fv.values["error_balance_orig"] == pytest.approx(0.0)


class TestNoInfinities:
    """A single inf survives imputation and poisons StandardScaler downstream."""

    def test_zero_denominators_yield_none_not_inf(self) -> None:
        first = txn(1, amount=0.0, step=1, oldbal=0.0, newbal=0.0)
        state = advance_state(first, None)
        fv = compute_features(txn(2, amount=500.0, step=2, oldbal=0.0, newbal=0.0), state)
        assert fv.values["amount_ratio_vs_last"] is None
        assert fv.values["amount_ratio_vs_prior_mean"] is None
        assert fv.values["amount_to_balance_ratio"] is None

    def test_no_feature_is_ever_inf_or_nan(self) -> None:
        seq = [
            txn(1, amount=0.0, step=1, oldbal=0.0, newbal=0.0),
            txn(2, amount=1e12, step=1, oldbal=0.0, newbal=0.0),
            txn(3, amount=0.0, step=744, oldbal=1e12, newbal=1e12),
        ]
        for fv, _ in compute_sequence(seq):
            for name in NUMERIC_FEATURES:
                value = fv.values[name]
                if value is None:
                    continue
                assert math.isfinite(float(value)), f"{name} = {value}"


class TestMonotonicSafety:
    """A late event must not move the account backwards in time."""

    def test_out_of_order_event_keeps_the_newer_anchor(self) -> None:
        newer = txn(2, amount=200.0, step=10)
        state = advance_state(newer, None)
        late = txn(1, amount=50.0, step=3)
        after = advance_state(late, state)

        assert after.last_event_ts_us == newer.event_ts_us
        assert after.last_amount == 200.0
        # ...but the running aggregates still absorb it.
        assert after.txn_count == 2
        assert after.amount_sum == pytest.approx(250.0)
        assert after.first_event_ts_us == late.event_ts_us


# ── the property that guarantees streaming state == batch state ──────────────

_amounts = st.floats(min_value=0.0, max_value=1e7, allow_nan=False, allow_infinity=False)


@st.composite
def _event_list(draw: st.DrawFn) -> list[TransactionEvent]:
    n = draw(st.integers(min_value=1, max_value=12))
    steps = draw(st.lists(st.integers(min_value=1, max_value=744), min_size=n, max_size=n))
    amounts = draw(st.lists(_amounts, min_size=n, max_size=n))
    types = draw(
        st.lists(
            st.sampled_from(["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]),
            min_size=n,
            max_size=n,
        )
    )
    return [
        txn(i, amount=a, step=s, txn_type=t)
        for i, (a, s, t) in enumerate(zip(amounts, steps, types, strict=True))
    ]


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(events=_event_list(), seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_merge_is_order_independent_and_matches_the_sequential_fold(
    events: list[TransactionEvent], seed: int
) -> None:
    """reduce(merge, shuffled singletons) == the sequential fold over sorted events.

    If this ever fails, Pathway's streaming state silently disagrees with the batch
    state and every online feature is suspect. This is the single most valuable test
    in the repo.
    """
    ordered = sorted(events, key=lambda e: (e.event_ts_us, e.event_id))

    sequential: AccountState | None = None
    for event in ordered:
        sequential = advance_state(event, sequential)
    assert sequential is not None

    singletons = [advance_state(e, None) for e in events]
    random.Random(seed).shuffle(singletons)
    merged = functools.reduce(merge_states, singletons)

    assert merged.last_event_id == sequential.last_event_id
    assert merged.last_event_ts_us == sequential.last_event_ts_us
    assert merged.last_amount == pytest.approx(sequential.last_amount)
    assert merged.txn_count == sequential.txn_count
    assert merged.amount_sum == pytest.approx(sequential.amount_sum, rel=1e-9)
    assert merged.amount_sqsum == pytest.approx(sequential.amount_sqsum, rel=1e-9)
    assert merged.max_amount == pytest.approx(sequential.max_amount)
    assert merged.first_event_ts_us == sequential.first_event_ts_us


@given(events=_event_list())
def test_merge_is_commutative(events: list[TransactionEvent]) -> None:
    states = [advance_state(e, None) for e in events]
    for left, right in itertools.pairwise(states):
        assert merge_states(left, right) == merge_states(right, left)
