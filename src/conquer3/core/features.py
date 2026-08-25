"""The feature computation. This module is the product; everything else is plumbing.

**Purity contract** -- enforced by tests, not just convention:

* no I/O, no network, no filesystem;
* no ``time.time()`` / ``datetime.now()`` -- every time input arrives as a parameter;
* no randomness, no global mutation, no logging;
* stdlib only.

That is what makes the three call sites provably equivalent: a live BentoML request,
a Pathway UDF, and a Colab notebook over the raw Kaggle CSV all get identical output
for identical input. The moment a feature is computed anywhere else -- in SQL, in a
notebook cell, in the service -- training/serving skew is back.

**Cold-start policy.** On an account's first transaction the window features are
*undefined*, and undefined is represented as ``None`` (numeric) or ``"__NONE__"``
(categorical). Never a numeric sentinel: ``0`` is a legitimate value for
``amount_delta_vs_last``, and ``-1`` for ``seconds_since_last_txn`` would be learned
as "very recent". Imputation is the sklearn pipeline's job, downstream, where it is
fitted and logged.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator

from conquer3.core import timeref
from conquer3.core.schema import (
    BALANCE_EPSILON,
    EPSILON,
    FEATURE_NAMES,
    NO_PREV_CATEGORY,
    STATE_SCHEMA_VERSION,
)
from conquer3.core.types import (
    FRAUD_CAPABLE_TYPES,
    AccountState,
    FeatureValue,
    FeatureVector,
    TransactionEvent,
)

__all__ = [
    "advance_state",
    "compute",
    "compute_features",
    "compute_sequence",
    "merge_last_block",
    "merge_states",
]


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    """Ratio, or ``None`` when the denominator is effectively zero.

    Never returns ``inf`` or ``nan``: an infinity would survive imputation, poison
    StandardScaler, and silently produce all-NaN predictions downstream.
    """
    if abs(denominator) < EPSILON:
        return None
    return numerator / denominator


def _log1p_or_none(value: float | None) -> float | None:
    if value is None or value < 0:
        return None
    return math.log1p(value)


def merge_last_block(a: AccountState, b: AccountState) -> AccountState:
    """Pick whichever state holds the genuinely later transaction.

    Shared by :func:`advance_state` and the Pathway accumulator so the two cannot
    drift. The comparison uses the total order ``(last_event_ts_us, last_event_id)``
    -- event_id breaks timestamp ties, so the result never depends on argument order.
    """
    return a if a.order_key >= b.order_key else b


def merge_states(a: AccountState, b: AccountState) -> AccountState:
    """Combine two partial states for the same account.

    **This must be associative and commutative.** Pathway's ``udf_reducer`` merges
    accumulators in an arbitrary, partition-dependent order, so a merge that assumed
    ``b`` was "the later one" would produce state that disagrees with the sequential
    fold intermittently -- the worst kind of bug, because it only shows up under
    certain partitionings.

    It holds because every prior is an associative aggregate (sum / count / max /
    min) and the ``last_*`` block is chosen by a *total order* on
    ``(last_event_ts_us, last_event_id)``, which has no ties.

    Lives here rather than in the Pathway layer so it is testable without pathway
    installed, and so the streaming and batch paths cannot drift apart.
    """
    if a.account_id != b.account_id:
        raise ValueError(f"cannot merge states for {a.account_id!r} and {b.account_id!r}")

    winner = merge_last_block(a, b)
    return AccountState(
        account_id=a.account_id,
        state_version=STATE_SCHEMA_VERSION,
        last_event_id=winner.last_event_id,
        last_event_ts_us=winner.last_event_ts_us,
        last_step=winner.last_step,
        last_txn_type=winner.last_txn_type,
        last_amount=winner.last_amount,
        last_oldbalance_org=winner.last_oldbalance_org,
        last_newbalance_orig=winner.last_newbalance_orig,
        last_dest_id=winner.last_dest_id,
        last_dest_is_merchant=winner.last_dest_is_merchant,
        last_fraud_score=winner.last_fraud_score,
        first_event_ts_us=min(a.first_event_ts_us, b.first_event_ts_us),
        txn_count=a.txn_count + b.txn_count,
        amount_sum=a.amount_sum + b.amount_sum,
        amount_sqsum=a.amount_sqsum + b.amount_sqsum,
        max_amount=max(a.max_amount, b.max_amount),
        updated_at_us=max(a.updated_at_us, b.updated_at_us),
    )


def compute_features(txn: TransactionEvent, prev: AccountState | None) -> FeatureVector:
    """Compute the feature vector for ``txn`` given the account's prior state.

    Args:
        txn: the transaction being scored.
        prev: the account's state *before* this transaction, or ``None`` if this is
            the account's first-ever transaction.
    """
    is_first = prev is None

    # -- recency ---------------------------------------------------------
    seconds_since_last: float | None = None
    steps_since_last: int | None = None
    if prev is not None:
        seconds_since_last = (txn.event_ts_us - prev.last_event_ts_us) / timeref.US_PER_SECOND
        steps_since_last = txn.step - prev.last_step

    # -- amount vs last ---------------------------------------------------
    amount_delta: float | None = None
    amount_ratio_last: float | None = None
    amount_velocity: float | None = None
    if prev is not None:
        amount_delta = txn.amount - prev.last_amount
        amount_ratio_last = _safe_ratio(txn.amount, prev.last_amount)
        # Clamp the interval to 1s so a burst of same-second transactions produces a
        # large-but-finite velocity rather than an explosion.
        elapsed = max(seconds_since_last or 0.0, 1.0)
        amount_velocity = txn.amount / elapsed

    # -- amount vs running priors -----------------------------------------
    ratio_prior_mean: float | None = None
    ratio_prior_max: float | None = None
    amount_z: float | None = None
    if prev is not None and prev.txn_count > 0:
        mean = prev.amount_sum / prev.txn_count
        ratio_prior_mean = _safe_ratio(txn.amount, mean)
        ratio_prior_max = _safe_ratio(txn.amount, prev.max_amount)
        if prev.txn_count >= 2:
            # Population variance from sum-of-squares. Clamp at 0: floating-point
            # cancellation can make this a small negative for near-constant amounts.
            variance = max(prev.amount_sqsum / prev.txn_count - mean * mean, 0.0)
            amount_z = _safe_ratio(txn.amount - mean, math.sqrt(variance))

    # -- tenure ------------------------------------------------------------
    account_age_hours: float | None = None
    txn_rate_per_hour: float | None = None
    if prev is not None:
        account_age_hours = (txn.event_ts_us - prev.first_event_ts_us) / timeref.US_PER_HOUR
        txn_rate_per_hour = _safe_ratio(float(prev.txn_count), max(account_age_hours, EPSILON))

    # -- type transition ---------------------------------------------------
    prev_type = NO_PREV_CATEGORY if prev is None else prev.last_txn_type
    type_changed: int | None = None if prev is None else int(txn.txn_type != prev.last_txn_type)

    # -- balance consistency ----------------------------------------------
    # Nonzero exactly when the account moved between our observations -- e.g. it was
    # the *destination* of someone else's transfer, or we dropped an event. Doubles
    # as a data-quality signal.
    balance_gap: float | None = None
    balance_gap_flag: int | None = None
    if prev is not None:
        balance_gap = txn.oldbalance_org - prev.last_newbalance_orig
        balance_gap_flag = int(abs(balance_gap) > BALANCE_EPSILON)

    values: dict[str, FeatureValue] = {
        "is_first_txn": int(is_first),
        "seconds_since_last_txn": seconds_since_last,
        "log1p_seconds_since_last": _log1p_or_none(seconds_since_last),
        "steps_since_last_txn": steps_since_last,
        "amount": txn.amount,
        "log1p_amount": _log1p_or_none(txn.amount),
        "amount_delta_vs_last": amount_delta,
        "amount_ratio_vs_last": amount_ratio_last,
        "amount_velocity": amount_velocity,
        "amount_ratio_vs_prior_mean": ratio_prior_mean,
        "amount_ratio_vs_prior_max": ratio_prior_max,
        "amount_z_vs_prior": amount_z,
        "txn_count_prior": 0 if prev is None else prev.txn_count,
        "account_age_hours": account_age_hours,
        "txn_rate_per_hour": txn_rate_per_hour,
        "type_changed": type_changed,
        "is_fraud_capable_type": int(txn.txn_type in FRAUD_CAPABLE_TYPES),
        "balance_gap_org": balance_gap,
        "balance_gap_flag": balance_gap_flag,
        "error_balance_orig": txn.oldbalance_org - txn.amount - txn.newbalance_orig,
        "error_balance_dest": txn.oldbalance_dest + txn.amount - txn.newbalance_dest,
        "balance_delta_org": txn.newbalance_orig - txn.oldbalance_org,
        "amount_to_balance_ratio": _safe_ratio(txn.amount, txn.oldbalance_org),
        "drains_account": int(
            txn.newbalance_orig < BALANCE_EPSILON and txn.oldbalance_org > BALANCE_EPSILON
        ),
        "orig_balance_was_zero": int(txn.oldbalance_org < BALANCE_EPSILON),
        "dest_is_merchant": int(txn.dest_is_merchant),
        "dest_is_new": None if prev is None else int(txn.dest_id != prev.last_dest_id),
        "dest_balance_was_zero": int(txn.oldbalance_dest < BALANCE_EPSILON),
        "hour_of_day": timeref.hour_of_day(txn.step),
        "sim_day_of_week": timeref.day_of_week(txn.step),
        "prev_fraud_score": None if prev is None else prev.last_fraud_score,
        "txn_type": txn.txn_type,
        "prev_txn_type": prev_type,
        "type_pair": f"{prev_type}->{txn.txn_type}",
    }

    # Cheap structural guard: catches a feature added to schema.py but not here (or
    # vice versa) at the first call rather than at DDL-generation time.
    if values.keys() != set(FEATURE_NAMES):
        missing = sorted(set(FEATURE_NAMES) - values.keys())
        extra = sorted(values.keys() - set(FEATURE_NAMES))
        raise RuntimeError(f"feature set mismatch (missing={missing}, extra={extra})")

    return FeatureVector(
        event_id=txn.event_id,
        account_id=txn.account_id,
        event_ts_us=txn.event_ts_us,
        values=values,
    )


def advance_state(
    txn: TransactionEvent,
    prev: AccountState | None,
    *,
    fraud_score: float | None = None,
    now_us: int | None = None,
) -> AccountState:
    """Fold ``txn`` into the account's state.

    Monotonic-safe: a late or out-of-order event still updates the running
    aggregates, but does **not** overwrite the ``last_*`` anchor. Without this, a
    replayed or delayed event would move the account "back in time" and every
    subsequent recency feature would be wrong.
    """
    updated_at = now_us if now_us is not None else txn.event_ts_us

    if prev is None:
        return AccountState(
            account_id=txn.account_id,
            state_version=STATE_SCHEMA_VERSION,
            last_event_id=txn.event_id,
            last_event_ts_us=txn.event_ts_us,
            last_step=txn.step,
            last_txn_type=txn.txn_type,
            last_amount=txn.amount,
            last_oldbalance_org=txn.oldbalance_org,
            last_newbalance_orig=txn.newbalance_orig,
            last_dest_id=txn.dest_id,
            last_dest_is_merchant=txn.dest_is_merchant,
            first_event_ts_us=txn.event_ts_us,
            txn_count=1,
            amount_sum=txn.amount,
            amount_sqsum=txn.amount * txn.amount,
            max_amount=txn.amount,
            last_fraud_score=fraud_score,
            updated_at_us=updated_at,
        )

    priors = {
        "first_event_ts_us": min(prev.first_event_ts_us, txn.event_ts_us),
        "txn_count": prev.txn_count + 1,
        "amount_sum": prev.amount_sum + txn.amount,
        "amount_sqsum": prev.amount_sqsum + txn.amount * txn.amount,
        "max_amount": max(prev.max_amount, txn.amount),
    }

    is_newer = (txn.event_ts_us, txn.event_id) >= prev.order_key
    if is_newer:
        anchor = {
            "last_event_id": txn.event_id,
            "last_event_ts_us": txn.event_ts_us,
            "last_step": txn.step,
            "last_txn_type": txn.txn_type,
            "last_amount": txn.amount,
            "last_oldbalance_org": txn.oldbalance_org,
            "last_newbalance_orig": txn.newbalance_orig,
            "last_dest_id": txn.dest_id,
            "last_dest_is_merchant": txn.dest_is_merchant,
            "last_fraud_score": fraud_score,
        }
    else:
        anchor = {
            "last_event_id": prev.last_event_id,
            "last_event_ts_us": prev.last_event_ts_us,
            "last_step": prev.last_step,
            "last_txn_type": prev.last_txn_type,
            "last_amount": prev.last_amount,
            "last_oldbalance_org": prev.last_oldbalance_org,
            "last_newbalance_orig": prev.last_newbalance_orig,
            "last_dest_id": prev.last_dest_id,
            "last_dest_is_merchant": prev.last_dest_is_merchant,
            "last_fraud_score": prev.last_fraud_score,
        }

    return AccountState(
        account_id=prev.account_id,
        state_version=STATE_SCHEMA_VERSION,
        updated_at_us=max(prev.updated_at_us, updated_at),
        **anchor,  # type: ignore[arg-type]
        **priors,  # type: ignore[arg-type]
    )


def compute(
    txn: TransactionEvent,
    prev: AccountState | None,
    *,
    fraud_score: float | None = None,
) -> tuple[FeatureVector, AccountState]:
    """Features for ``txn`` plus the state that should be stored after it.

    The online path calls this, scores the features, then re-attaches the score to
    the state before committing.
    """
    return (
        compute_features(txn, prev),
        advance_state(txn, prev, fraud_score=fraud_score),
    )


def compute_sequence(
    txns: Iterable[TransactionEvent],
    *,
    initial: AccountState | None = None,
) -> Iterator[tuple[FeatureVector, AccountState]]:
    """Fold over one account's chronologically-sorted transactions.

    This is the batch/training path. The caller is responsible for sorting -- passing
    unsorted transactions produces silently wrong recency features rather than an
    error, because out-of-order events are a legitimate production case handled by
    :func:`advance_state`.
    """
    state = initial
    for txn in txns:
        features = compute_features(txn, state)
        state = advance_state(txn, state)
        yield features, state
