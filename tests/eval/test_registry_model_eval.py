"""Model-registry evaluation suite: replays real PaySim1 transactions against a
*live* scorer's ``/predict``, once per (model name, version) that scorer has
pre-loaded -- spanning every registered model family, not just one -- and
records per-request predicted label, fraud score, and latency.

This is a benchmarking/comparison tool, not a correctness gate -- it exercises
whatever the running scorer already pre-loaded at startup (see
``serving/service.py``'s ``_preload_all_versions``, exposed as
``POST /models``), against the real MLflow registry and real PaySim1 CSV data.
A fresh ephemeral mlflow+redis stack (the way ``tests/integration/
test_serving_e2e.py`` builds one) would defeat the purpose here: there would be
nothing registered yet to evaluate. Point ``C3_EVAL_ENDPOINT`` at a scorer
that's already up against the registry/deployment you actually want numbers
for.

Talks to the scorer over HTTP only, via the same client module the Streamlit UI
uses (``conquer3.ui.scorer_client``) -- no serving-side import, matching the
"ui talks to serving over HTTP, never by import" contract even though this is a
test, not app code, so the client path this suite exercises is the same one a
real caller uses. Also imports ``conquer3.contracts.model_registry`` directly,
purely to enumerate the *full* registry for the "what got excluded" reporting
below -- allowed the same way ``ui/app.py`` already imports it directly:
``contracts`` sits below ``serving`` in the layering, so this isn't the
serving-side import the contract above forbids.

Requires a single-worker scorer (``configs/default.yaml``'s
``serving.scorer_workers: 1``, not ``.env`` -- see
``test_serving_e2e.py::test_switch_model_activates_a_preloaded_version_and_pauses_auto_reload``'s
docstring for why): ``POST /switch_model`` is per-worker, so with more than one
worker, BentoML's own load balancing could route ``/switch_model`` and the
``/predict`` calls that follow it to different processes, silently mixing
versions into one version's reported numbers. This suite calls
``GET /model_info`` right after every switch and asserts it reflects the
switch, so a multi-worker misconfiguration fails loudly (a version mismatch
assertion) instead of silently producing a mislabeled report.

Skips cleanly (does not fail) when no live scorer is reachable, or the CSV
isn't present -- this suite is opt-in, run by hand against a real deployment,
not part of the default ``pytest tests/`` gate.

Input contract -- all via environment variables, all optional:

  C3_EVAL_ENDPOINT      Scorer base URL.
                        Default: http://localhost:<C3_SCORER_HOST_PORT or 3000>.
  C3_EVAL_CSV_PATH      Raw PaySim1 CSV (Kaggle's own column names).
                        Default: data/raw/paysim1.csv.
  C3_EVAL_SAMPLE_SIZE   Number of transactions to replay per model -- the
                        *first* N rows of the CSV (not a random sample): this
                        keeps event_id/event_ts_us derivation identical to a
                        real ingest (see to_transactions_frame), and every
                        model is evaluated against the exact same,
                        reproducible sample. PaySim1's first rows are step=1
                        and fraud-sparse -- raise this if you need more
                        positive (is_fraud=True) examples in the sample.
                        Default: 500.
  C3_EVAL_TARGETS       Comma-separated "name:version" pairs to evaluate, e.g.
                        "paysim-fraud-lightgbm:2,paysim-fraud-lightgbm:3".
                        Default: every (name, version) GET /models reports as
                        pre-loaded on the target scorer -- i.e. every
                        registered model this scorer can actually serve.
  C3_EVAL_OUT_DIR       Directory per-model artifacts are written to.
                        Default: data/eval.
  C3_EVAL_TIMEOUT_S     Per-request HTTP timeout, seconds. Default: 30.

Output contract -- written to C3_EVAL_OUT_DIR, nested by model name and
version so a multi-model, multi-version report reads as a directory tree
rather than a flat pile of similarly-named files:

  <name>/v<version>/predictions.csv   One row per scored transaction, columns:
      event_id, is_fraud, is_flagged_fraud, fraud_score, decision,
      had_prev_state, seconds_since_last_txn, latency_ms
    -- fraud_score/decision/had_prev_state/seconds_since_last_txn are exactly
    ScoreResult's fields (serving/api_models.py); is_fraud/is_flagged_fraud are
    PaySim1's own ground truth, carried through for the caller's own
    precision/recall/ROC-AUC computation (not computed here -- see
    producer/replay.py's docstring on the same division of labor);
    latency_ms is this suite's own per-request wall-clock measurement, timed
    only after one discarded warm-up call right after the /switch_model that
    activated this version (see _evaluate_one_version) -- so this is
    steady-state serving latency, not inflated by a version's own one-off
    cold-start cost on its first post-switch request.

  <name>/v<version>/summary.json      One object, keys:
      model_name, model_version, run_id, alias, tags (registry identity, from
        the switched-to ModelInfoResponse -- confirms which artifact actually
        served these requests, not just which (name, version) was requested),
      n_requests, n_errors,
      p50_latency_ms, p90_latency_ms, p95_latency_ms, p99_latency_ms,
      mean_latency_ms, min_latency_ms, max_latency_ms

  <name>/summary.json          Per-model-family rollup, one per registered
                                model name (evaluated or not): keys
                                model_name, evaluated (that name's own
                                per-version summary objects above, one per
                                loaded version), excluded_versions (that name's
                                own entries from the exclusion list below).
                                Lets a caller compare versions within one
                                model family without reading the root summary.

  all_versions_summary.json    The *only* summary file directly under
                                C3_EVAL_OUT_DIR: every evaluated model's
                                summary object above across every name, for a
                                side-by-side comparison in one file, plus an
                                `excluded_versions` key: every (name, version)
                                registered in MLflow but *not* in the scorer's
                                pre-loaded pool -- so a model this scorer can't
                                feed a well-defined row (e.g. one with no
                                MLflow signature at all) is accounted for
                                explicitly, not silently absent from the
                                report.

Every request runs with ``dry_run=True``: no evaluation run writes Redis state
or the scored-event log, so every model reads the *same* pre-existing
account-state trajectory -- back-to-back runs across models never contaminate
each other's inputs, which is what makes their outputs comparable at all. It
also makes the whole suite safe to re-run repeatedly against a shared
deployment.

Each transaction is sent as its own ``/predict`` call (batch size 1), never
batched, so the measured latency is per-request -- what a live caller actually
experiences -- not amortized over a large batch the way a bulk replay would
blur it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from conquer3.contracts.model_registry import list_model_versions, list_registered_models
from conquer3.core.types import TransactionEvent
from conquer3.producer.replay import load_raw_paysim, to_transactions_frame
from conquer3.ui.scorer_client import (
    ScorerError,
    get_model_info,
    is_scorer_healthy,
    list_loaded_models,
    score_transactions,
    switch_model,
)

pytestmark = pytest.mark.eval

_DEFAULT_ENDPOINT = f"http://localhost:{os.environ.get('C3_SCORER_HOST_PORT', '3000')}"

# The exact key set/order /predict's TransactionIn expects per transaction --
# generated from TransactionEvent's own fields, the same rule
# producer/replay.py, serving/api_models.py, and pipelines/pathway/schemas.py
# each follow independently rather than hand-duplicating a parallel list.
_TXN_FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in dataclasses.fields(TransactionEvent))

# The exact per-row and per-version output shapes -- asserted against below, so
# a caller diffing reports across model versions (or across a code change to
# this suite) can rely on the column/key set never silently drifting.
PREDICTION_COLUMNS: tuple[str, ...] = (
    "event_id",
    "is_fraud",
    "is_flagged_fraud",
    "fraud_score",
    "decision",
    "had_prev_state",
    "seconds_since_last_txn",
    "latency_ms",
)
SUMMARY_KEYS: tuple[str, ...] = (
    "model_name",
    "model_version",
    "run_id",
    "alias",
    "tags",
    "n_requests",
    "n_errors",
    "p50_latency_ms",
    "p90_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "mean_latency_ms",
    "min_latency_ms",
    "max_latency_ms",
)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@pytest.fixture(scope="module")
def eval_config() -> dict[str, Any]:
    return {
        "endpoint": _env("C3_EVAL_ENDPOINT", _DEFAULT_ENDPOINT).rstrip("/"),
        "csv_path": Path(_env("C3_EVAL_CSV_PATH", "data/raw/paysim1.csv")),
        "sample_size": int(_env("C3_EVAL_SAMPLE_SIZE", "500")),
        "out_dir": Path(_env("C3_EVAL_OUT_DIR", "data/eval")),
        "timeout_s": float(_env("C3_EVAL_TIMEOUT_S", "30")),
    }


@pytest.fixture(scope="module")
def live_scorer(eval_config: dict[str, Any]) -> str:
    """The endpoint, confirmed reachable and ready -- skips the whole module
    otherwise, rather than failing every test with a connection error."""
    endpoint: str = eval_config["endpoint"]
    if not is_scorer_healthy(base_url=endpoint, timeout_s=eval_config["timeout_s"]):
        pytest.skip(f"no live, ready scorer at {endpoint} (GET /readyz) -- set C3_EVAL_ENDPOINT")
    return endpoint


@pytest.fixture(scope="module")
def sample_transactions(eval_config: dict[str, Any]) -> pd.DataFrame:
    """The first ``sample_size`` rows of the CSV, mapped onto TransactionEvent
    field names + ground truth -- see to_transactions_frame. The *first* N
    rows, not a random sample: to_transactions_frame derives event_id/
    event_ts_us from each row's position and its step's row count, so a
    contiguous head-slice keeps those identical to a real ingest, while a
    random sample would recompute them over a different, sample-only step
    distribution.
    """
    csv_path = eval_config["csv_path"]
    if not csv_path.is_file():
        pytest.skip(f"{csv_path} not found -- run `conquer3 ingest download` first")
    raw = load_raw_paysim(csv_path, limit=eval_config["sample_size"])
    return to_transactions_frame(raw)


@pytest.fixture(scope="module")
def targets_to_eval(eval_config: dict[str, Any], live_scorer: str) -> list[tuple[str, str]]:
    env_targets = os.environ.get("C3_EVAL_TARGETS")
    if env_targets:
        pairs = []
        for raw in env_targets.split(","):
            raw = raw.strip()
            if not raw:
                continue
            name, _, version = raw.partition(":")
            if not version:
                raise ValueError(f"C3_EVAL_TARGETS entry {raw!r} must be 'name:version'")
            pairs.append((name, version))
        return pairs
    loaded = list_loaded_models(base_url=live_scorer, timeout_s=eval_config["timeout_s"])
    if not loaded:
        pytest.skip("scorer reports no pre-loaded models (POST /models empty)")
    return [(str(info["name"]), str(info["version"])) for info in loaded]


def _excluded_targets(loaded_targets: list[tuple[str, str]]) -> list[dict[str, str]]:
    """Every (name, version) registered in MLflow but not pre-loaded by the
    live scorer -- e.g. a model with no usable MLflow signature at all (real
    example: paysim-fraud-lightgbm v1), which serving/service.py's pre-load
    step deliberately excludes rather than leave it to 500 on every
    /predict. Reported here so "every real model/version" is honestly
    accounted for instead of silently missing from the eval report.
    """
    loaded_set = set(loaded_targets)
    excluded: list[dict[str, str]] = []
    for name in list_registered_models():
        for info in list_model_versions(name):
            if (name, info.version) not in loaded_set:
                excluded.append({"name": name, "version": info.version})
    return excluded


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile -- no numpy dependency needed for this."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (pct / 100)
    lo, hi = int(rank), min(int(rank) + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def _evaluate_one_version(
    *, endpoint: str, name: str, version: str, transactions: pd.DataFrame, timeout_s: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Switches the live scorer to ``(name, version)``, confirms the switch
    actually took (guards the single-worker assumption -- see module
    docstring), discards one untimed warm-up call, then replays every row in
    ``transactions`` as its own timed /predict call. Returns (per-row records,
    summary dict) -- exactly PREDICTION_COLUMNS/SUMMARY_KEYS shaped.
    """
    switch_model(base_url=endpoint, name=name, version=version, timeout_s=timeout_s)
    active = get_model_info(base_url=endpoint, timeout_s=timeout_s)
    assert (active["name"], active["version"]) == (name, version), (
        f"requested {name!r} v{version!r} but /model_info reports "
        f"{active['name']!r} v{active['version']!r} active -- likely a multi-worker scorer "
        "(see module docstring: this suite needs serving.scorer_workers: 1)"
    )

    # One untimed warm-up call, discarded before timing starts: a just-switched
    # version can pay a one-off cold-start cost on its very first request
    # (confirmed empirically -- one version's first-call latency ran >10x its
    # own steady-state mean) that would otherwise land in every latency
    # percentile below, including max, for no reason a real caller would ever
    # see twice. Reuses the sample's own first row rather than a synthetic
    # one, so it still exercises the model's real feature path; that row is
    # still scored again, timed, in the loop below -- this warm-up doesn't
    # shrink the reported sample.
    warmup_row = transactions.iloc[0].to_dict()
    warmup_payload = {field: warmup_row[field] for field in _TXN_FIELD_NAMES}
    with contextlib.suppress(ScorerError, ValueError):
        # a genuinely broken version still fails for real on the timed request below
        score_transactions(
            [warmup_payload], base_url=endpoint, dry_run=True, batch_size=1, timeout_s=timeout_s
        )

    rows: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    errors = 0
    for record in transactions.to_dict(orient="records"):
        payload = {name: record[name] for name in _TXN_FIELD_NAMES}
        start = time.perf_counter()
        try:
            (result,) = score_transactions(
                [payload], base_url=endpoint, dry_run=True, batch_size=1, timeout_s=timeout_s
            )
        except (ScorerError, ValueError):
            errors += 1
            continue
        latency_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(latency_ms)
        rows.append(
            {
                "event_id": record["event_id"],
                "is_fraud": bool(record["is_fraud"]),
                "is_flagged_fraud": bool(record["is_flagged_fraud"]),
                "fraud_score": result["fraud_score"],
                "decision": result["decision"],
                "had_prev_state": result["had_prev_state"],
                "seconds_since_last_txn": result["seconds_since_last_txn"],
                "latency_ms": latency_ms,
            }
        )

    summary = {
        "model_name": active["name"],
        "model_version": active["version"],
        "run_id": active["run_id"],
        "alias": active["alias"],
        "tags": active.get("tags", {}),
        "n_requests": len(rows),
        "n_errors": errors,
        "p50_latency_ms": _percentile(latencies_ms, 50),
        "p90_latency_ms": _percentile(latencies_ms, 90),
        "p95_latency_ms": _percentile(latencies_ms, 95),
        "p99_latency_ms": _percentile(latencies_ms, 99),
        "mean_latency_ms": statistics.fmean(latencies_ms) if latencies_ms else float("nan"),
        "min_latency_ms": min(latencies_ms) if latencies_ms else float("nan"),
        "max_latency_ms": max(latencies_ms) if latencies_ms else float("nan"),
    }
    return rows, summary


def test_evaluate_every_registered_version(
    eval_config: dict[str, Any],
    live_scorer: str,
    sample_transactions: pd.DataFrame,
    targets_to_eval: list[tuple[str, str]],
) -> None:
    out_dir = eval_config["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    all_summaries: list[dict[str, Any]] = []
    summaries_by_name: dict[str, list[dict[str, Any]]] = {}

    for name, version in targets_to_eval:
        rows, summary = _evaluate_one_version(
            endpoint=live_scorer,
            name=name,
            version=version,
            transactions=sample_transactions,
            timeout_s=eval_config["timeout_s"],
        )

        # -- well-defined output, part 1: per-row predictions, nested under
        # <name>/v<version>/ so two registered models sharing a version number
        # (confirmed on the real registry: paysim-fraud-lightgbm v3 and
        # paysim_fraud_clf v3 both exist) never collide on disk -----------
        version_dir = out_dir / name / f"v{version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        predictions = pd.DataFrame(rows, columns=list(PREDICTION_COLUMNS))
        predictions.to_csv(version_dir / "predictions.csv", index=False)

        # -- well-defined output, part 2: latency/identity summary -----------
        (version_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        all_summaries.append(summary)
        summaries_by_name.setdefault(name, []).append(summary)

        # -- sanity assertions (not accuracy checks -- see module docstring) -
        assert set(summary) == set(SUMMARY_KEYS)
        assert summary["n_requests"] == len(sample_transactions) - summary["n_errors"]
        assert summary["n_requests"] > 0, f"{name} v{version}: every request errored"
        assert list(predictions.columns) == list(PREDICTION_COLUMNS)
        assert predictions["decision"].isin(["FRAUD", "LEGIT"]).all()
        assert predictions["fraud_score"].between(0.0, 1.0).all()
        assert (predictions["latency_ms"] > 0).all()

    # -- well-defined output, part 3: every registered (name, version) this
    # scorer excluded from the pool, grouped by name alongside the evaluated
    # ones below -- a model excluded on every version (e.g.
    # paysim-fraud-xgb-baseline) still gets its own rollup file. --
    excluded = _excluded_targets(targets_to_eval)
    excluded_by_name: dict[str, list[dict[str, str]]] = {}
    for entry in excluded:
        excluded_by_name.setdefault(entry["name"], []).append(entry)

    # -- well-defined output, part 4: one rollup per model family, directly
    # under <name>/ (a sibling of that name's v<version>/ directories, never
    # inside one) -- lets a caller compare versions within one model without
    # reading the all-model root summary --------------------------------
    for name in set(summaries_by_name) | set(excluded_by_name):
        family_dir = out_dir / name
        family_dir.mkdir(parents=True, exist_ok=True)
        (family_dir / "summary.json").write_text(
            json.dumps(
                {
                    "model_name": name,
                    "evaluated": summaries_by_name.get(name, []),
                    "excluded_versions": excluded_by_name.get(name, []),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # -- well-defined output, part 5: the *only* summary file directly under
    # out_dir -- one file to compare every model, plus every registered
    # (name, version) this scorer excluded from the pool --
    (out_dir / "all_versions_summary.json").write_text(
        json.dumps({"evaluated": all_summaries, "excluded_versions": excluded}, indent=2),
        encoding="utf-8",
    )
