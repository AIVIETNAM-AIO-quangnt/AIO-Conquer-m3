"""Tab 3 -- search scored-event history by name_orig/name_dest/txn_type, then
compare every model's prediction for one transaction.

"Same transaction" is matched by natural key (account_id + dest_id + txn_type +
amount), not event_id: the Inference tab's single-transaction form mints a fresh
event_id on every submit, so a value resubmitted after switching the active model
via the sidebar's Instant-swap control would otherwise never link back up.

Read-only and historical only -- this never calls the scorer. It only ever reads
whatever ``ScoredEvent`` rows already exist in the JSONL history (see history.py).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from conquer3.config.settings import Settings
from conquer3.ui import history

__all__ = ["render"]

_NATURAL_KEY = ("account_id", "dest_id", "txn_type", "amount")


def _with_natural_key_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Adds flat ``dest_id``/``txn_type``/``amount`` columns pulled out of the
    nested ``transaction`` dict, so every filter/group/sort below is plain column
    access. ``account_id`` is already flat on ``df`` and always mirrors
    ``transaction["account_id"]`` -- see serving/scorer.py's ``_to_scored_event``,
    which builds ``transaction`` via ``dataclasses.asdict(txn)`` and sets
    ``account_id=txn.account_id`` from that same object."""
    if df.empty:
        return df.assign(dest_id=[], txn_type=[], amount=[])
    return df.assign(
        dest_id=df["transaction"].map(lambda t: t.get("dest_id")),
        txn_type=df["transaction"].map(lambda t: t.get("txn_type")),
        amount=df["transaction"].map(lambda t: t.get("amount")),
    )


def _has_any_filter(*, name_orig: str, name_dest: str, txn_type: str) -> bool:
    return bool(name_orig) or bool(name_dest) or (txn_type != "All")


def _apply_search_filters(
    df: pd.DataFrame, *, name_orig: str, name_dest: str, txn_type: str
) -> pd.DataFrame:
    """ANDs together whichever of the three filters are actually set."""
    if name_orig:
        df = df[df["account_id"] == name_orig]
    if name_dest:
        df = df[df["dest_id"] == name_dest]
    if txn_type != "All":
        df = df[df["txn_type"] == txn_type]
    return df


def _aggregate_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Count of matching predictions per ``model_name``/``model_version``, most
    predictions first."""
    counts = (
        df.groupby(["model_name", "model_version"], as_index=False)
        .size()
        .rename(columns={"size": "predictions"})
        .sort_values("predictions", ascending=False, ignore_index=True)
    )
    counts["model"] = counts["model_name"] + "@" + counts["model_version"]
    return counts[["model", "predictions"]]


def _distinct_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct natural key among the matches, with how many
    predictions exist for it and when it was last scored -- newest first."""
    return (
        df.groupby(list(_NATURAL_KEY), as_index=False)
        .agg(predictions=("event_id", "size"), last_scored_at_us=("scored_at_us", "max"))
        .sort_values("last_scored_at_us", ascending=False, ignore_index=True)
    )


def _natural_key_label(row: pd.Series) -> str:
    return (
        f"{row['account_id']} -> {row['dest_id']} "
        f"({row['txn_type']}, amount={row['amount']:.2f})"
    )


def _compare_rows(df: pd.DataFrame, natural_key: dict[str, Any]) -> pd.DataFrame:
    """Every prediction across the FULL loaded history (not just the current
    search results) that shares ``natural_key`` exactly, oldest first."""
    mask = pd.Series(True, index=df.index)
    for field, value in natural_key.items():
        mask &= df[field] == value
    matches = df[mask].sort_values("scored_at_us", ignore_index=True)
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(matches["scored_at_us"], unit="us", utc=True),
            "model@version": matches["model_name"] + "@" + matches["model_version"],
            "fraud_score": matches["fraud_score"],
            "threshold": matches["threshold"],
            "decision": matches["decision"],
        }
    )


def render(settings: Settings, knobs: dict[str, Any]) -> None:
    df = history.load_events_frame(
        settings.event.dir, settings.ui.history_max_files, settings.ui.history_max_rows
    )
    if df.empty:
        st.info("No scored events found yet under the events volume.")
        return
    df = _with_natural_key_fields(df)

    cols = st.columns(3)
    name_orig = cols[0].text_input("name_orig (account_id)")
    name_dest = cols[1].text_input("name_dest (dest_id)")
    txn_type = cols[2].selectbox("txn_type", ["All", *sorted(df["txn_type"].dropna().unique())])

    if not _has_any_filter(name_orig=name_orig, name_dest=name_dest, txn_type=txn_type):
        st.info("Enter at least one of name_orig, name_dest, or txn_type to search.")
        return

    matched = _apply_search_filters(df, name_orig=name_orig, name_dest=name_dest, txn_type=txn_type)
    if matched.empty:
        st.info("No predictions match that search.")
        return

    st.markdown("#### Predictions by model")
    st.dataframe(_aggregate_by_model(matched), hide_index=True)

    st.markdown("#### Matching transactions")
    distinct = _distinct_transactions(matched)
    display = distinct.assign(
        last_scored_at=pd.to_datetime(distinct["last_scored_at_us"], unit="us", utc=True)
    ).drop(columns="last_scored_at_us")
    st.dataframe(display, hide_index=True)

    labels = [_natural_key_label(row) for _, row in distinct.iterrows()]
    pick_index = st.selectbox(
        "Pick a transaction to compare", range(len(distinct)), format_func=lambda i: labels[i]
    )
    if st.button("Compare"):
        picked = distinct.iloc[pick_index]
        st.session_state.ui_benchmark_natural_key = {f: picked[f] for f in _NATURAL_KEY}

    selected_key = st.session_state.get("ui_benchmark_natural_key")
    if selected_key is not None:
        st.divider()
        st.markdown("#### Compare across models")
        st.dataframe(_compare_rows(df, selected_key), hide_index=True)
