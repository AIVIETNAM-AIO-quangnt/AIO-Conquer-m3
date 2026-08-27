"""The reusable Pathway graph: staged transactions -> one row per account_id
(account_id, state_json, state_version, updated_at_us). Static backfill and
streaming state repair both build this exact graph over a source table that
differs only in `mode` (see sources.py) -- "same connector, both modes."
"""

from __future__ import annotations

from typing import Any, cast

import pathway as pw

from conquer3.core.serde import state_from_json
from conquer3.pipelines.pathway.accumulators import account_state_reducer

__all__ = ["build_account_state_table"]


@pw.udf
def _extract_state_version(state_json: str) -> int:
    state = state_from_json(state_json)
    assert state is not None, "accumulator always emits a valid state document"
    return state.state_version


@pw.udf
def _extract_updated_at_us(state_json: str) -> int:
    state = state_from_json(state_json)
    assert state is not None, "accumulator always emits a valid state document"
    return state.updated_at_us


def build_account_state_table(events: pw.Table[Any]) -> pw.Table[Any]:
    """``events`` must be TransactionEventSchema-shaped (see schemas.py)."""
    grouped = events.groupby(events.account_id).reduce(
        account_id=events.account_id,
        state_json=account_state_reducer(
            events.event_id,
            events.account_id,
            events.dest_id,
            events.txn_type,
            events.amount,
            events.oldbalance_org,
            events.newbalance_orig,
            events.oldbalance_dest,
            events.newbalance_dest,
            events.event_ts_us,
            events.step,
        ),
    )
    return cast(
        "pw.Table[Any]",
        grouped.select(
            pw.this.account_id,
            pw.this.state_json,
            state_version=_extract_state_version(pw.this.state_json),
            updated_at_us=_extract_updated_at_us(pw.this.state_json),
        ),
    )
