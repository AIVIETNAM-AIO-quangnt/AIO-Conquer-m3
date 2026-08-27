"""The Pathway custom reducer that folds TransactionEvents into AccountState.

Delegates every bit of merge logic to conquer3.core.features -- this file only
translates between Pathway's per-reducer-call row list and TransactionEvent/
AccountState. update()'s associativity/commutativity is exactly the property
tests/unit/test_features.py::test_merge_is_order_independent_and_matches_the_
sequential_fold already proves for core.features.merge_states -- delegating means
that Hypothesis test is this accumulator's correctness proof too, with zero
duplicated logic.

`retract` is intentionally NOT implemented: pw.BaseCustomAccumulator's default
(absent) simply means the engine falls back to a full recompute of that account's
group on a retraction instead of an incremental undo -- a cost, never a correctness
gap. Our source (export_staging.py's JSONL files) is written once and never mutated
in place, so retraction is not expected to occur in the backfill/streaming paths
this layer builds.
"""

from __future__ import annotations

from typing import Any

import pathway as pw

from conquer3.core.features import advance_state, merge_states
from conquer3.core.serde import state_to_json
from conquer3.core.types import AccountState, TransactionEvent

__all__ = ["AccountStateAccumulator", "account_state_reducer"]


class AccountStateAccumulator(pw.BaseCustomAccumulator):
    def __init__(self, state: AccountState) -> None:
        self.state = state

    @classmethod
    def from_row(cls, row: list[Any]) -> AccountStateAccumulator:
        (
            event_id,
            account_id,
            dest_id,
            txn_type,
            amount,
            oldbalance_org,
            newbalance_orig,
            oldbalance_dest,
            newbalance_dest,
            event_ts_us,
            step,
        ) = row
        txn = TransactionEvent(
            event_id=event_id,
            account_id=account_id,
            dest_id=dest_id,
            txn_type=txn_type,
            amount=amount,
            oldbalance_org=oldbalance_org,
            newbalance_orig=newbalance_orig,
            oldbalance_dest=oldbalance_dest,
            newbalance_dest=newbalance_dest,
            event_ts_us=event_ts_us,
            step=step,
        )
        return cls(advance_state(txn, None))

    def update(self, other: AccountStateAccumulator) -> None:
        self.state = merge_states(self.state, other.state)

    def compute_result(self) -> str:
        return state_to_json(self.state)


account_state_reducer = pw.reducers.udf_reducer(AccountStateAccumulator)
