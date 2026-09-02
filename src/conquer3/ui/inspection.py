"""Tab 2 -- browse scored-event history, assign ground truth, measure P-metrics,
and plot predictions over two chosen dimensions.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import Any, cast

import pandas as pd
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from conquer3.config.settings import Settings
from conquer3.core.schema import COLD_START_NULL_FEATURES, FEATURE_NAMES
from conquer3.core.types import TransactionEvent
from conquer3.ui import history, labels

__all__ = ["render"]

_GROUND_TRUTH_OPTIONS = ("unlabeled", "fraud", "legit")
_RAW_TXN_FIELDS = tuple(
    f.name for f in dataclasses.fields(TransactionEvent) if f.name != "event_id"
)
# Deduplicated, order-preserving: "txn_type" appears in both FEATURE_NAMES and the
# raw transaction fields, and a selectbox shouldn't offer the same axis twice.
_PLOT_AXES = tuple(dict.fromkeys((*FEATURE_NAMES, *_RAW_TXN_FIELDS)))


def _label_to_ground_truth(is_fraud: bool | None) -> str:
    if is_fraud is None:
        return "unlabeled"
    return "fraud" if is_fraud else "legit"


def _parse_bool_column(series: pd.Series) -> pd.Series:
    """A CSV's is_fraud column may be "True"/"False", "1"/"0", or a native bool
    dtype -- plain `bool(x)` on a string like "False" is truthy, so this parses
    the text explicitly instead."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "fraud"})


def _correctness_color(predicted: str, ground_truth: str) -> str:
    actual = {"fraud": "FRAUD", "legit": "LEGIT"}.get(ground_truth)
    if actual is None:
        return "unlabeled"
    return "correct" if predicted == actual else "wrong"


def _axis_value(row: pd.Series, axis: str) -> float | str | None:
    source = row["features"] if axis in FEATURE_NAMES else row["transaction"]
    return cast("float | str | None", source.get(axis))


@st.cache_data(ttl=15, show_spinner=False)
def _load_frame(events_dir: str, max_files: int, max_rows: int) -> pd.DataFrame:
    events = history.load_recent_events(events_dir, max_files=max_files, max_rows=max_rows)
    return history.events_to_frame(events)


def _render_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    lo, hi = st.slider("Confidence (fraud_score)", 0.0, 1.0, (0.0, 1.0))
    df = df[df["fraud_score"].between(lo, hi)]

    decision = st.selectbox("Decision", ["All", "FRAUD", "LEGIT"])
    if decision != "All":
        df = df[df["decision"] == decision]

    versions = ["All", *sorted(df["model_version"].unique(), reverse=True)]
    version = st.selectbox("Model version", versions)
    if version != "All":
        df = df[df["model_version"] == version]

    scored_after = st.text_input(
        "Scored after (ISO, blank for no filter)", value="", placeholder="2026-09-01T00:00:00"
    )
    if scored_after:
        try:
            cutoff_us = int(datetime.fromisoformat(scored_after).timestamp() * 1_000_000)
            df = df[df["scored_at_us"] >= cutoff_us]
        except ValueError:
            st.warning(f"Could not parse {scored_after!r} as an ISO timestamp; ignoring.")

    return df


def _render_metrics(df: pd.DataFrame, ground_truth: pd.Series, *, threshold: float) -> None:
    labeled = df.assign(ground_truth=ground_truth)
    labeled = labeled[labeled["ground_truth"] != "unlabeled"]

    st.caption(
        f"{len(labeled)} of {len(df)} predictions labeled "
        f"({(len(labeled) / len(df) * 100) if len(df) else 0:.1f}%) -- "
        "metrics are over the labeled subset only."
    )
    if labeled.empty:
        st.info("Label at least one prediction to see precision/recall/F-beta.")
        return

    y_true = (labeled["ground_truth"] == "fraud").to_numpy()
    y_pred = (labeled["fraud_score"] >= threshold).to_numpy()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    cols = st.columns(4)
    cols[0].metric("Predictions", len(df))
    cols[1].metric("Labeled", len(labeled))
    cols[2].metric(f"Precision @ {threshold:.2f}", f"{precision:.3f}")
    cols[3].metric(f"Recall @ {threshold:.2f}", f"{recall:.3f}")

    beta = st.number_input("beta (β)", min_value=0.01, value=1.0, step=0.1)
    fbeta = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
    st.metric(f"F{beta:g} score", f"{fbeta:.3f}")
    st.caption(
        "β = 1.0 is exactly F1. β > 1 weights recall -- the usual direction for fraud, "
        "where a missed fraud costs more than a false alarm."
    )

    if y_true.any() and (~y_true).any():
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        st.dataframe(
            pd.DataFrame(
                [[tn, fp], [fn, tp]],
                index=["actual legit", "actual fraud"],
                columns=["predicted legit", "predicted fraud"],
            )
        )
        precisions, recalls, _ = precision_recall_curve(y_true, labeled["fraud_score"])
        fpr, tpr, _ = roc_curve(y_true, labeled["fraud_score"])
        pr_col, roc_col = st.columns(2)
        pr_col.caption("Precision-recall curve")
        pr_col.line_chart(pd.DataFrame({"precision": precisions}, index=recalls))
        roc_col.caption("ROC curve")
        roc_col.line_chart(pd.DataFrame({"tpr": tpr}, index=fpr))
    else:
        st.caption("Need at least one labeled fraud and one labeled legit row for PR/ROC curves.")


def _render_history_and_labels(df: pd.DataFrame) -> pd.Series:
    event_ids = df["event_id"].tolist()
    current_labels = labels.get_labels(event_ids)
    ground_truth = df["event_id"].map(lambda eid: _label_to_ground_truth(current_labels.get(eid)))

    display_cols = [
        "scored_at_us",
        "event_id",
        "account_id",
        "fraud_score",
        "decision",
        "model_version",
        "ground_truth",
    ]
    display = df.assign(ground_truth=ground_truth)[display_cols]
    display["scored_at"] = pd.to_datetime(display["scored_at_us"], unit="us", utc=True)
    display = display.drop(columns="scored_at_us").set_index("event_id")

    edited = st.data_editor(
        display,
        column_config={
            "ground_truth": st.column_config.SelectboxColumn(
                "ground truth", options=list(_GROUND_TRUTH_OPTIONS), required=True
            )
        },
        disabled=[c for c in display.columns if c != "ground_truth"],
        hide_index=False,
        use_container_width=True,
    )

    col_save, col_import = st.columns(2)
    if col_save.button("Save labels"):
        edits: dict[str, str | None] = {
            str(event_id): (None if value == "unlabeled" else value)
            for event_id, value in edited["ground_truth"].items()
        }
        labels.apply_label_edits(edits, source="ui")
        st.cache_data.clear()
        st.success(f"Saved {len(edits)} label(s).")

    uploaded = col_import.file_uploader("Import labels CSV (event_id,is_fraud)", type="csv")
    if uploaded is not None:
        try:
            imported = pd.read_csv(uploaded)
            unknown = set(imported["event_id"]) - set(event_ids)
            is_fraud = _parse_bool_column(imported["is_fraud"])
            csv_edits: dict[str, str | None] = {
                str(event_id): ("fraud" if fraud else "legit")
                for event_id, fraud in zip(imported["event_id"], is_fraud, strict=True)
            }
            labels.apply_label_edits(csv_edits, source="csv")
            st.cache_data.clear()
            st.success(f"Imported {len(csv_edits)} label(s).")
            if unknown:
                st.warning(
                    f"{len(unknown)} event_id(s) from the CSV are not in the currently "
                    f"loaded history window: {sorted(unknown)[:10]}"
                )
        except (KeyError, ValueError) as exc:
            st.error(f"Could not import labels CSV: {exc}")

    if edited.empty:
        return pd.Series(dtype=str)
    return edited["ground_truth"].reset_index(drop=True)


def _render_plot(df: pd.DataFrame, ground_truth: pd.Series, *, threshold: float) -> None:
    st.markdown("#### 2D plot")
    cols = st.columns(3)
    x_axis = cols[0].selectbox("X axis", _PLOT_AXES, index=_PLOT_AXES.index("log1p_amount"))
    y_axis = cols[1].selectbox("Y axis", _PLOT_AXES, index=_PLOT_AXES.index("amount_z_vs_prior"))
    color_by = cols[2].selectbox("Color by", ["decision", "ground truth", "correctness"])

    gt_by_row = ground_truth.reindex(range(len(df)), fill_value="unlabeled")
    plot_rows = []
    excluded = 0
    for (_, row), gt in zip(df.iterrows(), gt_by_row, strict=True):
        x = _axis_value(row, x_axis)
        y = _axis_value(row, y_axis)
        if x is None or y is None:
            excluded += 1
            continue
        if color_by == "decision":
            color = row["decision"]
        elif color_by == "ground truth":
            color = gt
        else:
            predicted = "FRAUD" if row["fraud_score"] >= threshold else "LEGIT"
            color = _correctness_color(predicted, gt)
        plot_rows.append({"x": x, "y": y, "color": color})

    if not plot_rows:
        st.info("No rows have both axes defined.")
        return

    st.scatter_chart(pd.DataFrame(plot_rows), x="x", y="y", color="color")
    note = f"{excluded} row(s) excluded: one of the chosen axes was null for them."
    if x_axis in COLD_START_NULL_FEATURES or y_axis in COLD_START_NULL_FEATURES:
        note += " That's expected on an account's first transaction (cold start)."
    st.caption(note)


def render(settings: Settings, knobs: dict[str, Any]) -> None:
    df = _load_frame(
        settings.event.dir, settings.ui.history_max_files, settings.ui.history_max_rows
    )
    if df.empty:
        st.info("No scored events found yet under the events volume.")
        return

    df = _render_filters(df)
    if df.empty:
        st.info("No predictions match the current filters.")
        return
    # Reset to a plain 0..n-1 index so `ground_truth` (built in the same row
    # order by _render_history_and_labels) aligns with `df` positionally --
    # `DataFrame.assign` below aligns by index label, not position, and the
    # post-filter index is otherwise a non-contiguous leftover from `df`.
    df = df.reset_index(drop=True)

    ground_truth = _render_history_and_labels(df)
    _render_metrics(df, ground_truth, threshold=knobs["threshold"])
    st.divider()
    _render_plot(df, ground_truth, threshold=knobs["threshold"])
