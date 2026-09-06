"""ui.inference's pure helper -- transaction row assembly -- exercised without a
Streamlit runtime, same pattern as test_ui_history.py.
"""

from __future__ import annotations

from conquer3.core.timeref import derive_step_from_ts_us
from conquer3.db.accounts import TransferResult
from conquer3.ui.inference import _build_transaction_row


def test_build_transaction_row_carries_the_executed_transfer_s_balances() -> None:
    transfer = TransferResult(
        oldbalance_org=181.0, newbalance_orig=0.0, oldbalance_dest=0.0, newbalance_dest=181.0
    )
    event_ts_us = 1_700_000_000_000_000
    row = _build_transaction_row(
        event_id="ui-abc123",
        event_ts_us=event_ts_us,
        account_id="C1",
        dest_id="M900",
        txn_type="PAYMENT",
        amount=181.0,
        transfer=transfer,
    )
    assert row == {
        "event_id": "ui-abc123",
        "event_ts_us": event_ts_us,
        "step": derive_step_from_ts_us(event_ts_us),
        "account_id": "C1",
        "dest_id": "M900",
        "txn_type": "PAYMENT",
        "amount": 181.0,
        "oldbalance_org": 181.0,
        "newbalance_orig": 0.0,
        "oldbalance_dest": 0.0,
        "newbalance_dest": 181.0,
    }
