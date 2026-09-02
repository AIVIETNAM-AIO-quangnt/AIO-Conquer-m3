"""Streamlit entrypoint (``conquer3 ui``) -- Layer 9.

The UI is a client of the scorer, never a second scorer: this package holds no
model, computes no feature, and never imports ``conquer3.serving`` (see the "ui
talks to serving over HTTP, never by import" import-linter contract). Every
score shown came from a real ``POST /predict``.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from conquer3.config.settings import Settings, get_settings
from conquer3.ui import inference, inspection
from conquer3.ui.scorer_client import ScorerError, get_model_info, is_scorer_healthy


def _sidebar(settings: Settings) -> dict[str, Any]:
    st.sidebar.markdown("### conquer3")

    healthy = is_scorer_healthy(base_url=settings.ui.scorer_url)
    status = "healthy" if healthy else "unreachable"
    st.sidebar.markdown(f"{'🟢' if healthy else '🔴'} scorer {status}")

    champion_version = None
    if healthy:
        try:
            info = get_model_info(base_url=settings.ui.scorer_url)
        except ScorerError as exc:
            st.sidebar.caption(f"model_info unavailable: {exc}")
        else:
            champion_version = info.get("version")
            fsv = info.get("tags", {}).get("feature_schema_version", "?")
            st.sidebar.caption(f"champion {champion_version} · fsv {fsv}")
            if info.get("degraded"):
                st.sidebar.warning("degraded: serving from the cached champion")

    st.sidebar.divider()
    threshold = st.sidebar.slider(
        "Decision threshold",
        min_value=0.0,
        max_value=1.0,
        value=settings.serving.decision_threshold,
        step=0.01,
        help="Re-derives FRAUD/LEGIT from fraud_score locally; the server's own "
        "threshold (C3_DECISION_THRESHOLD) is unaffected.",
    )
    dry_run = st.sidebar.toggle("dry run (skip Redis/event writes)", value=False)
    batch_size = int(
        st.sidebar.number_input(
            "Batch size", min_value=1, max_value=5000, value=settings.ui.request_batch_size
        )
    )
    timeout_s = float(
        st.sidebar.number_input(
            "Request timeout (s)", min_value=1, max_value=600, value=settings.ui.request_timeout_s
        )
    )

    return {
        "threshold": threshold,
        "dry_run": dry_run,
        "batch_size": batch_size,
        "timeout_s": timeout_s,
        "champion_version": champion_version,
    }


def main() -> None:
    st.set_page_config(page_title="conquer3", layout="wide")
    settings = get_settings()
    knobs = _sidebar(settings)

    tab_inference, tab_inspection = st.tabs(["Inference", "Inspection"])
    with tab_inference:
        inference.render(settings, knobs)
    with tab_inspection:
        inspection.render(settings, knobs)


if __name__ == "__main__":
    main()
