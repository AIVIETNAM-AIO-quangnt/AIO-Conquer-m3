"""Transaction replay driver used to exercise the serving path.

Reads a raw PaySim1 CSV (Kaggle's own column names -- ``step``, ``type``,
``amount``, ``nameOrig``, ``oldbalanceOrg``, ``newbalanceOrig``, ``nameDest``,
``oldbalanceDest``, ``newbalanceDest``, ``isFraud``, ``isFlaggedFraud``),
replays it in file order as a client of the scorer's ``POST /predict``, and
writes one row per transaction -- ground truth plus the score -- to an output
CSV for offline evaluation (precision/recall/ROC-AUC, computed later, not here).

``event_id`` and ``event_ts_us`` are derived exactly the way
``pipelines.transforms.bronze_to_silver`` derives them for ``silver.txn``
(``'ps-' + row_num`` zero-padded to 10 digits; the same integer formula as
``core.timeref.derive_event_ts_us``, vectorized here for 6.3M-row throughput --
see that module's docstring, which already documents this formula as
reproduced in multiple places, and ``tests/parity/test_event_ts_us_pandas.py``
for the check that this reproduction is exact). Replaying a CSV here and
ingesting the same CSV through the warehouse pipeline therefore produce
identical ids and timestamps -- results are joinable against
``silver.txn``/``gold.txn_features`` without a second identity scheme.

Talks to the scorer over HTTP only. ``conquer3.producer`` sits in the same
architectural layer as ``conquer3.serving`` (see the "layered architecture"
import-linter contract): it may not import serving code directly, only call
it the way any other client would.

**Not dry-run by default.** A faithful replay needs each account's
transactions to see the state the *previous* transaction in this same replay
committed, and that continuity has to survive request-batch boundaries across
a 6M-row file -- Redis is what carries it, so this writes real state and real
scored-event JSONL lines. Point it at a throwaway/dev deployment, not a shared
one. ``--dry-run`` skips those writes; a batch spanning more than one
account's total history will then score later rows in that batch against
stale (pre-replay) priors for that account -- fine for a quick smoke check,
not for a run meant to match ``gold.txn_features``.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pandas as pd

from conquer3.core.timeref import SIM_EPOCH_US, US_PER_HOUR
from conquer3.core.types import TransactionEvent

__all__ = ["load_raw_paysim", "replay", "run_replay", "to_transactions_frame"]

_RAW_COLUMNS = (
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
)

# Generated from TransactionEvent's own fields, not hand-duplicated -- the same
# rule serving/api_models.py and pipelines/pathway/schemas.py follow, and the
# exact key set/order the scorer's /predict route expects per transaction.
_TXN_FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in dataclasses.fields(TransactionEvent))

_LABEL_COLUMNS = ("is_fraud", "is_flagged_fraud")

# The score/decision fields /predict returns, in the order they're written to
# the output CSV.
_RESULT_COLUMNS = (
    "fraud_score",
    "decision",
    "had_prev_state",
    "seconds_since_last_txn",
    "model_version",
    "feature_schema_version",
    "degraded",
)

_OUTPUT_COLUMNS = ("event_id", *_LABEL_COLUMNS, *_RESULT_COLUMNS)


def load_raw_paysim(csv_path: Path | str, *, limit: int | None = None) -> pd.DataFrame:
    """Reads a raw PaySim1 CSV, verifying it has the Kaggle column set.

    Reads whatever columns are present first, rather than passing
    ``usecols=_RAW_COLUMNS`` to ``read_csv`` directly -- pandas' own error for
    a missing ``usecols`` entry names only the mismatch, not which of our
    required columns it is, so the check below reports it clearly instead.
    """
    df = pd.read_csv(csv_path, nrows=limit)
    missing = set(_RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing raw PaySim1 columns: {sorted(missing)}")
    return df[list(_RAW_COLUMNS)]


def _derive_event_ts_us_vectorized(
    step: pd.Series, intra_step_seq: pd.Series, step_cardinality: pd.Series
) -> pd.Series:
    """Vectorized reproduction of ``core.timeref.derive_event_ts_us`` -- see
    that module's docstring on why the formula is duplicated rather than
    called per row (a 6.3M-row Python-level ``apply`` doesn't scale).
    Constants are imported, never retyped, so only the arithmetic shape is
    duplicated; ``tests/parity/test_event_ts_us_pandas.py`` calls this exact
    function against a 100k-row sweep to check the duplication stays exact.
    """
    offset = ((intra_step_seq - 1) * US_PER_HOUR) // step_cardinality.clip(lower=1)
    return SIM_EPOCH_US + (step - 1) * US_PER_HOUR + offset


def to_transactions_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Maps raw Kaggle rows onto ``TransactionEvent`` field names plus labels.

    ``event_id``/``event_ts_us`` reproduce ``bronze_to_silver``'s derivation
    exactly, assuming (as that transform also assumes) the input is already in
    file/row order -- true of the official PaySim1 CSV, whose ``step`` column
    is non-decreasing throughout the file. If it weren't, the derived
    timestamps would still be internally consistent per step, just not
    globally chronological with row order -- ``replay()`` submits batches in
    frame order, so a caller replaying a reordered CSV should re-sort by
    (step, row order) first.
    """
    # Indexed like `raw` throughout, so every Series below aligns positionally
    # with it when the DataFrame is constructed -- no accidental reindexing.
    row_num = pd.Series(range(1, len(raw) + 1), index=raw.index)
    step = raw["step"].astype("int64")
    step_cardinality = step.groupby(step).transform("count")
    intra_step_seq = step.groupby(step).cumcount() + 1
    event_ts_us = _derive_event_ts_us_vectorized(step, intra_step_seq, step_cardinality)

    out = pd.DataFrame(
        {
            "event_id": "ps-" + row_num.astype(str).str.zfill(10),
            "account_id": raw["nameOrig"],
            "dest_id": raw["nameDest"],
            "txn_type": raw["type"],
            "amount": raw["amount"].astype("float64"),
            "oldbalance_org": raw["oldbalanceOrg"].astype("float64"),
            "newbalance_orig": raw["newbalanceOrig"].astype("float64"),
            "oldbalance_dest": raw["oldbalanceDest"].astype("float64"),
            "newbalance_dest": raw["newbalanceDest"].astype("float64"),
            "event_ts_us": event_ts_us,
            "step": step,
            "is_fraud": raw["isFraud"].astype(bool),
            "is_flagged_fraud": raw["isFlaggedFraud"].astype(bool),
        }
    )
    missing = set(_TXN_FIELD_NAMES) - set(out.columns)
    assert not missing, f"to_transactions_frame is missing TransactionEvent fields: {missing}"
    return out


def _batches(frame: pd.DataFrame, batch_size: int) -> Iterator[pd.DataFrame]:
    for start in range(0, len(frame), batch_size):
        yield frame.iloc[start : start + batch_size]


def replay(
    csv_path: Path | str,
    *,
    endpoint: str,
    batch_size: int = 200,
    dry_run: bool = False,
    timeout_s: float = 30.0,
    limit: int | None = None,
    client: httpx.Client | None = None,
) -> Iterator[pd.DataFrame]:
    """Replays ``csv_path`` against ``{endpoint}/predict`` in file order.

    Yields one :class:`pandas.DataFrame` per batch, with ``_OUTPUT_COLUMNS``
    -- the caller (``main`` below) is what actually writes a CSV; this stays a
    generator so a full 6.3M-row replay never holds every scored row in memory
    at once.
    """
    raw = load_raw_paysim(csv_path, limit=limit)
    frame = to_transactions_frame(raw)

    owns_client = client is None
    client = client or httpx.Client(timeout=timeout_s)
    url = endpoint.rstrip("/") + "/predict"
    try:
        for batch in _batches(frame, batch_size):
            # `.to_dict(orient="records")` would leave numpy int64/float64/bool_
            # scalars in each row, which the stdlib `json` module httpx serializes
            # with cannot encode. Routing through pandas' own `to_json` (a
            # vectorized C implementation, not a per-row Python cast loop) yields
            # plain int/float/str/bool -- safe to serialize, and the columns
            # selected are exactly `_TXN_FIELD_NAMES`, so the request body matches
            # what `/predict`'s TransactionIn model expects, field for field.
            transactions = json.loads(batch[list(_TXN_FIELD_NAMES)].to_json(orient="records"))
            resp = client.post(url, json={"transactions": transactions, "dry_run": dry_run})
            if resp.status_code != 200:
                first_id = batch["event_id"].iloc[0]
                raise RuntimeError(
                    f"POST {url} failed with {resp.status_code} for batch starting at "
                    f"{first_id!r}: {resp.text[:500]}"
                )
            results = pd.DataFrame(resp.json())
            merged = batch[["event_id", *_LABEL_COLUMNS]].merge(results, on="event_id", how="left")
            yield merged[list(_OUTPUT_COLUMNS)]
    finally:
        if owns_client:
            client.close()


def run_replay(
    csv_path: Path | str,
    out_path: Path | str,
    *,
    endpoint: str,
    batch_size: int = 200,
    dry_run: bool = False,
    timeout_s: float = 30.0,
    limit: int | None = None,
    progress_every: int = 50,
) -> int:
    """Replays ``csv_path`` against ``{endpoint}/predict`` and writes ground
    truth + prediction, one row per transaction, to ``out_path``.

    Returns the row count written. The CLI wrapper (``conquer3 replay``, in
    ``cli.py``) owns argument parsing and defaults, matching every other
    subcommand in this project -- this stays a plain function so it's callable
    directly from a test or a notebook without going through argv.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"replaying {csv_path} -> {endpoint}/predict (dry_run={dry_run})", file=sys.stderr)
    start = time.monotonic()
    total = 0
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_OUTPUT_COLUMNS)
        for i, batch in enumerate(
            replay(
                csv_path,
                endpoint=endpoint,
                batch_size=batch_size,
                dry_run=dry_run,
                timeout_s=timeout_s,
                limit=limit,
            ),
            start=1,
        ):
            writer.writerows(batch.itertuples(index=False, name=None))
            total += len(batch)
            if progress_every and i % progress_every == 0:
                elapsed = time.monotonic() - start
                print(
                    f"  batch {i}: {total} rows scored, {total / elapsed:.0f} rows/s",
                    file=sys.stderr,
                )

    elapsed = time.monotonic() - start
    print(f"done: {total} rows -> {out} in {elapsed:.1f}s", file=sys.stderr)
    return total
