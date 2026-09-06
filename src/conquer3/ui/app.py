"""Streamlit entrypoint (``conquer3 ui``) -- Layer 9.

The UI is a client of the scorer, never a second scorer: this package holds no
model, computes no feature, and never imports ``conquer3.serving`` (see the "ui
talks to serving over HTTP, never by import" import-linter contract). Every
score shown came from a real ``POST /predict``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd
import streamlit as st

from conquer3.config.settings import Settings, get_settings
from conquer3.contracts.model_registry import list_model_versions, list_registered_models
from conquer3.ui import benchmark, inference, inspection
from conquer3.ui.scorer_client import (
    ScorerError,
    get_model_info,
    is_scorer_healthy,
    list_loaded_models,
    switch_model,
)


def _render_model_registry(settings: Settings) -> None:
    st.sidebar.markdown("#### Registry — MLflow models")
    try:
        model_names = list_registered_models(settings=settings)
    except Exception as exc:
        st.sidebar.info(f"Could not list registered models: {exc}")
        return
    if not model_names:
        st.sidebar.caption("No registered models found.")
        return

    default_name = settings.model.name
    selected_name = st.sidebar.selectbox(
        "Model",
        model_names,
        index=model_names.index(default_name) if default_name in model_names else 0,
        help="Every model name registered on the MLflow registry, not just "
        f"the configured default ({default_name!r}).",
    )

    try:
        versions = list_model_versions(model_name=selected_name, settings=settings)
    except Exception as exc:
        st.sidebar.info(f"Could not list versions of {selected_name!r}: {exc}")
        return
    if not versions:
        st.sidebar.caption(f"No registered versions of {selected_name!r} found.")
        return

    rows = [
        {
            "version": v.version,
            "created": datetime.fromtimestamp(v.created_at_ms / 1000, tz=UTC).strftime(
                "%Y-%m-%d"
            )
        }
        for v in versions
    ]
    st.sidebar.dataframe(pd.DataFrame(rows), hide_index=True)

    # Promotion only ever affects settings.model.name: that's the one model
    # name this worker's automatic champion-hot-reload loop watches (see
    # serving/service.py's _champion_reload_loop). Promoting an alias on any
    # other model name would silently do nothing observable, so the control is
    # only offered while browsing that one model.
    # if selected_name == default_name:
    #     compatible_versions = [v.version for v in versions if v.compatible]
    #     target = st.sidebar.selectbox("Version to promote", compatible_versions)
    #     if st.sidebar.button("Promote to champion", disabled=not compatible_versions):
    #         try:
    #             promote_champion(target, model_name=selected_name, settings=settings)
    #         except IncompatibleModelError as exc:
    #             st.sidebar.error(f"Refused: {exc}")
    #         else:
    #             st.session_state.ui_pending_promotion = {
    #                 "version": target,
    #                 "requested_at": time.time(),
    #             }

    #     pending = st.session_state.get("ui_pending_promotion")
    #     if pending:
    #         try:
    #             served_version = get_model_info(base_url=settings.ui.scorer_url).get("version")
    #         except ScorerError:
    #             served_version = None
    #         if served_version == pending["version"]:
    #             st.sidebar.success(f"scorer now serving version {pending['version']}")
    #             del st.session_state["ui_pending_promotion"]
    #         else:
    #             elapsed = int(time.time() - pending["requested_at"])
    #             st.sidebar.info(
    #                 f"Promoted v{pending['version']} in MLflow {elapsed}s ago -- waiting for the "
    #                 f"scorer's own poll (≤ {settings.serving.champion_poll_s}s) to pick it up. "
    #                 "Interact with the page (or reload) to re-check."
    #             )
    # else:
    #     st.sidebar.caption(
    #         f"Promotion only applies to the configured champion model ({default_name!r}) -- "
    #         "select it above to promote one of its versions."
    #     )

    st.sidebar.divider()
    st.sidebar.markdown("### Instant swap (this worker only)")
    try:
        loaded = list_loaded_models(base_url=settings.ui.scorer_url)
    except ScorerError as exc:
        st.sidebar.info(f"Could not list pre-loaded models: {exc}")
        return
    if not loaded:
        st.sidebar.caption("No pre-loaded models reported.")
        return

    # Spans every registered model now, not just the configured default --
    # labeled "name vversion" since two different models can share a version
    # number (e.g. paysim-fraud-lightgbm v3 and paysim_fraud_clf v3 both
    # exist on the real registry).
    labels = [f"{m['name']} v{m['version']}" for m in loaded]
    active_index = next((i for i, m in enumerate(loaded) if m["active"]), 0)
    swap_index = st.sidebar.selectbox(
        "Model to switch to now",
        range(len(loaded)),
        format_func=lambda i: labels[i],
        index=active_index,
        help="Only models/versions this specific worker pre-loaded at startup "
        "(POST /models) -- switching is immediate, no MLflow round-trip, but "
        "only affects whichever worker answers this call (with "
        "C3_SCORER_WORKERS > 1, other workers are unaffected until switched "
        "too), and pauses this worker's automatic champion-hot-reload polling "
        "until it restarts.",
    )
    if st.sidebar.button("Switch now", disabled=swap_index == active_index):
        swap_choice = loaded[swap_index]
        try:
            result = switch_model(
                base_url=settings.ui.scorer_url,
                name=swap_choice["name"],
                version=swap_choice["version"],
            )
        except ScorerError as exc:
            st.sidebar.error(f"Switch failed: {exc}")
        else:
            st.sidebar.success(f"worker now serving {result['name']} v{result['version']}")
            st.rerun()


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
    _render_model_registry(settings)

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
            "Request timeout (s)", min_value=1, max_value=1200, value=settings.ui.request_timeout_s
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

    tab_inference, tab_inspection, tab_benchmark = st.tabs(["Inference", "Inspection", "Benchmark"])
    with tab_inference:
        inference.render(settings, knobs)
    with tab_inspection:
        inspection.render(settings, knobs)
    with tab_benchmark:
        benchmark.render(settings, knobs)


if __name__ == "__main__":
    main()
