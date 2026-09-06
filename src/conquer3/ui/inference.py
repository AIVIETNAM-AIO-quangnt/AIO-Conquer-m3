"""Tab 1 -- score a hand-entered transaction or an uploaded CSV.

Model-version selection/promotion lives in the sidebar (ui/app.py), not here.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pandas as pd
import streamlit as st

from conquer3.config.settings import Settings
from conquer3.core.timeref import derive_step_from_ts_us
from conquer3.core.types import TransactionEvent, TxnType
from conquer3.db.accounts import InsufficientBalanceError, TransferResult, execute_transfer
from conquer3.db.engine import pg_connection
from conquer3.producer.replay import load_raw_paysim, to_transactions_frame
from conquer3.ui.scorer_client import ScorerError, score_transactions

__all__ = ["render"]

_TXN_FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in dataclasses.fields(TransactionEvent))
# Sample values only (not part of the request contract) -- the same numbers the
# README's own curl example uses, so a fresh user sees a realistic, known-good row.
_FIELD_SAMPLE_VALUES: dict[str, float | str] = {
    "account_id": "C1",
    "dest_id": "M900",
    "amount": 181.0,
}
# CASH_IN excluded: it models a merchant/agent crediting a person, which doesn't fit
# this form's model (name_orig is always the sender being debited).
_SELECTABLE_TXN_TYPES: tuple[TxnType, ...] = (
    TxnType.PAYMENT,
    TxnType.TRANSFER,
    TxnType.DEBIT,
    TxnType.CASH_OUT,
)


def _new_event_id() -> str:
    # "ui-" keeps hand-entered rows out of the replay driver's "ps-{row:010d}"
    # namespace. Auto-assigned and read-only: event_id is the primary key of
    # ops.prediction_labels and the join key back to silver.txn, so a hand-typed
    # duplicate would silently overwrite an existing label.
    return f"ui-{uuid4().hex[:12]}"


def _build_transaction_row(
    *,
    event_id: str,
    event_ts_us: int,
    account_id: str,
    dest_id: str,
    txn_type: str,
    amount: float,
    transfer: TransferResult,
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "event_ts_us": event_ts_us,
        "step": derive_step_from_ts_us(event_ts_us),
        "account_id": account_id,
        "dest_id": dest_id,
        "txn_type": txn_type,
        "amount": amount,
        "oldbalance_org": transfer.oldbalance_org,
        "newbalance_orig": transfer.newbalance_orig,
        "oldbalance_dest": transfer.oldbalance_dest,
        "newbalance_dest": transfer.newbalance_dest,
    }


def _single_transaction_form() -> list[dict[str, Any]] | None:
    if "ui_event_id" not in st.session_state:
        st.session_state.ui_event_id = _new_event_id()

    with st.form("single_txn"):
        st.text_input(
            "event_id",
            value=st.session_state.ui_event_id,
            disabled=True,
            help="Auto-assigned and unique; the primary key of ops.prediction_labels.",
        )
        account_id = st.text_input(
            "name_orig",
            value=str(_FIELD_SAMPLE_VALUES["account_id"]),
            help="Sending account. Auto-provisioned with a starting balance on first use.",
        )
        dest_id = st.text_input(
            "name_dest",
            value=str(_FIELD_SAMPLE_VALUES["dest_id"]),
            help="Receiving account. Auto-provisioned at a zero balance on first use.",
        )
        txn_type = st.selectbox("txn_type", [t.value for t in _SELECTABLE_TXN_TYPES], index=1)
        amount = float(
            st.number_input("amount", value=float(_FIELD_SAMPLE_VALUES["amount"]), min_value=0.01)
        )
        submitted = st.form_submit_button("Transfer and score")

    if not submitted:
        return None

    event_ts_us = int(datetime.now(tz=UTC).timestamp() * 1_000_000)
    with pg_connection() as conn:
        try:
            transfer = execute_transfer(
                conn, name_orig=account_id, name_dest=dest_id, amount=amount
            )
        except (InsufficientBalanceError, ValueError) as exc:
            st.error(str(exc))
            return None

    st.caption(
        f"Transferred {amount:.2f}: {account_id} {transfer.oldbalance_org:.2f} -> "
        f"{transfer.newbalance_orig:.2f}; {dest_id} {transfer.oldbalance_dest:.2f} -> "
        f"{transfer.newbalance_dest:.2f}"
    )

    row = _build_transaction_row(
        event_id=st.session_state.ui_event_id,
        event_ts_us=event_ts_us,
        account_id=account_id,
        dest_id=dest_id,
        txn_type=txn_type,
        amount=amount,
        transfer=transfer,
    )
    st.session_state.ui_event_id = _new_event_id()
    return [row]


def _csv_upload_rows() -> list[dict[str, Any]] | None:
    uploaded = st.file_uploader("Upload a raw PaySim1 CSV", type="csv")
    if uploaded is None:
        return None
    try:
        raw = load_raw_paysim(uploaded)  # type: ignore[arg-type]
    except ValueError as exc:
        st.error(str(exc))
        return None

    frame = to_transactions_frame(raw)
    st.caption(f"{len(frame)} rows validated; derived event_id shown below")
    st.dataframe(frame[["event_id", *_TXN_FIELD_NAMES[1:]]].head(10), hide_index=True)
    if not st.button(f"Score all {len(frame)} rows"):
        return None
    rows: list[dict[str, Any]] = json.loads(frame[list(_TXN_FIELD_NAMES)].to_json(orient="records"))
    return rows


def _render_score_results(results: list[dict[str, Any]], *, threshold: float) -> None:
    if not results:
        st.warning("scorer returned no results")
        return
    df = pd.DataFrame(results)
    # fraud_score is a probability, so re-deriving the decision at the sidebar's
    # threshold is a pure display computation -- no new request needed, and it's
    # the same slider the Inspection tab's precision/recall reads.
    df[f"decision @ {threshold:.2f}"] = (
        df["fraud_score"].ge(threshold).map({True: "FRAUD", False: "LEGIT"})
    )
    display_df = df[['event_id', 'fraud_score', f"decision @ {threshold:.2f}"]].copy()
    st.dataframe(display_df, hide_index=True)
    server_threshold = results[0].get("decision")
    st.caption(
        f"Server threshold decision is `{server_threshold}` "
        f"(the scorer's own C3_DECISION_THRESHOLD); the column above re-derives "
        f"it locally at the slider's value."
    )


def render(settings: Settings, knobs: dict[str, Any]) -> None:
    st.subheader("Score a transaction")
    mode = st.radio("Input", ["Single transaction", "Upload CSV"], horizontal=True)
    rows = _single_transaction_form() if mode == "Single transaction" else _csv_upload_rows()

    if rows:
        try:
            results = score_transactions(
                rows,
                base_url=settings.ui.scorer_url,
                dry_run=knobs["dry_run"],
                batch_size=knobs["batch_size"],
                timeout_s=knobs["timeout_s"],
            )
        except ScorerError as exc:
            st.error(str(exc))
        else:
            _render_score_results(results, threshold=knobs["threshold"])
