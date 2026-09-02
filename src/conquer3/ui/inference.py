"""Tab 1 -- score a hand-entered transaction or an uploaded CSV, and manage which
MLflow version the scorer is aliased to serve.
"""

from __future__ import annotations

import dataclasses
import json
import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pandas as pd
import streamlit as st

from conquer3.config.settings import Settings
from conquer3.contracts.model_registry import (
    IncompatibleModelError,
    list_model_versions,
    promote_champion,
)
from conquer3.core.timeref import derive_step_from_ts_us
from conquer3.core.types import TransactionEvent, TxnType
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
    "oldbalance_org": 181.0,
    "newbalance_orig": 0.0,
    "oldbalance_dest": 0.0,
    "newbalance_dest": 181.0,
}


def _new_event_id() -> str:
    # "ui-" keeps hand-entered rows out of the replay driver's "ps-{row:010d}"
    # namespace. Auto-assigned and read-only: event_id is the primary key of
    # ops.prediction_labels and the join key back to silver.txn, so a hand-typed
    # duplicate would silently overwrite an existing label.
    return f"ui-{uuid4().hex[:12]}"


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
        values: dict[str, Any] = {}
        cols = st.columns(2)
        slot = 0
        for f in dataclasses.fields(TransactionEvent):
            if f.name in ("event_id", "event_ts_us", "step"):
                continue
            col = cols[slot % 2]
            slot += 1
            if f.name == "txn_type":
                values[f.name] = col.selectbox("txn_type", [t.value for t in TxnType])
                continue
            # TransactionEvent uses `from __future__ import annotations`, so
            # `f.type` is the string annotation, not the real type object --
            # same assumption api_models.py's `_build_transaction_in` makes.
            annotation = f.type
            assert isinstance(annotation, str), (
                "core/types.py must keep `from __future__ import annotations` for "
                "this string-annotation assumption to hold"
            )
            if annotation == "float":
                default_f = _FIELD_SAMPLE_VALUES.get(f.name, 0.0)
                values[f.name] = float(col.number_input(f.name, value=float(default_f)))
            else:
                default_s = _FIELD_SAMPLE_VALUES.get(f.name, "")
                values[f.name] = str(col.text_input(f.name, value=str(default_s)))

        now_us = int(datetime.now(tz=UTC).timestamp() * 1_000_000)
        event_ts_us = int(st.number_input("event_ts_us", value=now_us, step=1))
        submitted = st.form_submit_button("Score")

    if not submitted:
        return None

    row = {
        "event_id": st.session_state.ui_event_id,
        "event_ts_us": event_ts_us,
        "step": derive_step_from_ts_us(event_ts_us),
        **values,
    }
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
    st.dataframe(df, hide_index=True)
    server_threshold = results[0].get("decision")
    st.caption(
        f"Server threshold decision is `{server_threshold}` "
        f"(the scorer's own C3_DECISION_THRESHOLD); the column above re-derives "
        f"it locally at the slider's value."
    )


def _render_model_registry(settings: Settings) -> None:
    st.markdown("#### Registry — MLflow model versions")
    try:
        versions = list_model_versions(settings=settings)
    except Exception as exc:
        st.info(f"Could not list model versions: {exc}")
        return
    if not versions:
        st.caption("No registered versions found.")
        return

    rows = [
        {
            "version": v.version,
            "created": datetime.fromtimestamp(v.created_at_ms / 1000, tz=UTC).strftime(
                "%Y-%m-%d %H:%M"
            ),
            "feature_schema_version": v.tags.get("feature_schema_version", "?"),
            "compatible": v.compatible,
            "aliases": ", ".join(v.aliases) or "-",
            "run_id": v.run_id,
        }
        for v in versions
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    compatible_versions = [v.version for v in versions if v.compatible]
    target = st.selectbox("Version to promote", compatible_versions)
    if st.button("Promote to champion", disabled=not compatible_versions):
        try:
            promote_champion(target, settings=settings)
        except IncompatibleModelError as exc:
            st.error(f"Refused: {exc}")
        else:
            st.session_state.ui_pending_promotion = {"version": target, "requested_at": time.time()}

    pending = st.session_state.get("ui_pending_promotion")
    if pending:
        try:
            from conquer3.ui.scorer_client import get_model_info

            served_version = get_model_info(base_url=settings.ui.scorer_url).get("version")
        except ScorerError:
            served_version = None
        if served_version == pending["version"]:
            st.success(f"scorer now serving version {pending['version']}")
            del st.session_state["ui_pending_promotion"]
        else:
            elapsed = int(time.time() - pending["requested_at"])
            st.info(
                f"Promoted v{pending['version']} in MLflow {elapsed}s ago -- waiting for the "
                f"scorer's own poll (≤ {settings.serving.champion_poll_s}s) plus a brief "
                "restart. Interact with the page (or reload) to re-check."
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

    st.divider()
    _render_model_registry(settings)
