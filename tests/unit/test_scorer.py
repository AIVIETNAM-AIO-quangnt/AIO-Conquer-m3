"""FraudScorer.score() logic, tested directly against fakes for the Redis state
store and event sink -- no model artifact, no Redis, no HTTP, so this stays fast
and infra-free, matching how core.features is tested directly. The full stack
(real MLflow, real Redis, the actual BentoML server) is proven in
tests/integration/test_serving_e2e.py; this file is where the batching/dry_run/
decision logic gets its coverage.

The `op` dispatch that used to live alongside this logic is gone: `/model_info`
and `/predict` are separate routes now, so the legacy-envelope multiplexer is
tested at the service level in test_serving_e2e.py instead.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import pandas as pd
import pytest

from conquer3.contracts.model_registry import ModelRef
from conquer3.core.types import AccountState, TransactionEvent
from conquer3.serving.scorer import FraudScorer

_REF = ModelRef(name="m", version="7", run_id="r1", alias="champion", tags={}, degraded=False)


class _FakePipe:
    """predict_proba: a fixed, known score -- deterministic on 'amount' so tests
    can pin exact expected decisions without depending on real sklearn output."""

    def predict_proba(self, rows: pd.DataFrame) -> Any:
        return pd.DataFrame(
            {
                0: [1 - self._proba(a) for a in rows["amount"]],
                1: [self._proba(a) for a in rows["amount"]],
            }
        ).to_numpy()

    @staticmethod
    def _proba(amount: float) -> float:
        return 0.9 if amount >= 500 else 0.1


class _ConstantPipe:
    """predict_proba never varies with input -- stands in for a freshly reloaded
    champion the reload test can distinguish from _FakePipe purely by its output,
    the same shape as this session's "fraud_score is always the same constant"
    incident this test guards against."""

    def __init__(self, proba: float) -> None:
        self._proba = proba

    def predict_proba(self, rows: pd.DataFrame) -> Any:
        n = len(rows)
        return pd.DataFrame({0: [1 - self._proba] * n, 1: [self._proba] * n}).to_numpy()


class _FakeStateStore:
    def __init__(self) -> None:
        self.data: dict[str, AccountState] = {}
        self.commits: list[AccountState] = []

    def get(self, account_id: str) -> AccountState | None:
        return self.data.get(account_id)

    def commit(self, state: AccountState) -> bool:
        self.data[state.account_id] = state
        self.commits.append(state)
        return True


class _FakeEventSink:
    def __init__(self) -> None:
        self.events: list[object] = []

    def append(self, event: object) -> None:
        self.events.append(event)


def _make_scorer(
    *, threshold: float = 0.5, ref: ModelRef = _REF
) -> tuple[FraudScorer, _FakeStateStore, _FakeEventSink]:
    state = _FakeStateStore()
    sink = _FakeEventSink()
    scorer = FraudScorer(
        pipe=_FakePipe(),
        ref=ref,
        threshold=threshold,
        state=state,  # type: ignore[arg-type]
        sink=sink,  # type: ignore[arg-type]
    )
    return scorer, state, sink


def _txn(
    *, event_id: str, account_id: str = "C1", amount: float = 100.0, event_ts_us: int
) -> TransactionEvent:
    return TransactionEvent(
        event_id=event_id,
        account_id=account_id,
        dest_id="C900",
        txn_type="TRANSFER",
        amount=amount,
        oldbalance_org=1000.0,
        newbalance_orig=1000.0 - amount,
        oldbalance_dest=0.0,
        newbalance_dest=amount,
        event_ts_us=event_ts_us,
        step=1,
    )


_T0 = 1_700_000_000_000_000


def test_scorer_exposes_the_ref_it_was_built_with() -> None:
    scorer, _state, _sink = _make_scorer()
    assert scorer.ref.name == "m"
    assert scorer.ref.version == "7"
    assert not scorer.ref.degraded


def test_first_transaction_is_cold_start_and_gets_committed_and_appended() -> None:
    scorer, state, sink = _make_scorer()
    (resp,) = scorer.score([_txn(event_id="e1", event_ts_us=_T0, amount=100.0)])

    assert not resp.had_prev_state
    assert resp.seconds_since_last_txn is None
    assert resp.decision == "LEGIT"  # amount=100 -> proba 0.1 < threshold 0.5
    assert resp.model_version == "7"
    assert not resp.degraded
    assert len(state.commits) == 1
    assert len(sink.events) == 1


def test_second_request_sees_state_committed_by_the_first() -> None:
    scorer, _state, _sink = _make_scorer()
    scorer.score([_txn(event_id="e1", event_ts_us=_T0)])
    (resp2,) = scorer.score([_txn(event_id="e2", event_ts_us=_T0 + 5_000_000)])
    assert resp2.had_prev_state
    assert resp2.seconds_since_last_txn == pytest.approx(5.0)


def test_decision_flips_to_fraud_above_threshold() -> None:
    scorer, _state, _sink = _make_scorer(threshold=0.5)
    (resp,) = scorer.score([_txn(event_id="e1", event_ts_us=_T0, amount=999.0)])
    assert resp.fraud_score == pytest.approx(0.9)
    assert resp.decision == "FRAUD"


def test_batch_same_account_folds_sequentially_within_one_call() -> None:
    """Two rows for the same account in ONE score() call must see each other's
    state even though the state store starts empty -- plan §8.4's batch semantics,
    independent of anything Redis persisted across separate requests."""
    scorer, state, sink = _make_scorer()
    resp1, resp2 = scorer.score(
        [
            _txn(event_id="b1", event_ts_us=_T0),
            _txn(event_id="b2", event_ts_us=_T0 + 5_000_000),
        ]
    )

    assert not resp1.had_prev_state
    assert resp2.had_prev_state
    assert resp2.seconds_since_last_txn == pytest.approx(5.0)
    # Exactly one commit per row, and the account's final committed state is the
    # one folded from the second (later) row, not the first.
    assert len(state.commits) == 2
    assert state.data["C1"].last_event_id == "b2"
    assert len(sink.events) == 2


def test_batch_out_of_request_order_is_folded_in_event_ts_us_order() -> None:
    """Rows for one account must be folded by event_ts_us, not by list position --
    the plan's ordering guarantee, not merely "first row wins"."""
    scorer, state, _sink = _make_scorer()
    results = scorer.score(
        [
            _txn(event_id="later", event_ts_us=_T0 + 5_000_000),
            _txn(event_id="earlier", event_ts_us=_T0),
        ]
    )
    by_id = {r.event_id: r for r in results}
    assert not by_id["earlier"].had_prev_state
    assert by_id["later"].had_prev_state
    assert by_id["later"].seconds_since_last_txn == pytest.approx(5.0)
    assert state.data["C1"].last_event_id == "later"


def test_results_are_returned_in_request_order_not_fold_order() -> None:
    """Reordering happens inside the fold; the caller still gets one result per
    input row, positionally aligned with what it sent."""
    scorer, _state, _sink = _make_scorer()
    results = scorer.score(
        [
            _txn(event_id="later", account_id="A", event_ts_us=_T0 + 5_000_000),
            _txn(event_id="other", account_id="B", event_ts_us=_T0),
            _txn(event_id="earlier", account_id="A", event_ts_us=_T0),
        ]
    )
    assert [r.event_id for r in results] == ["later", "other", "earlier"]


def test_batch_different_accounts_are_independent() -> None:
    scorer, _state, _sink = _make_scorer()
    results = scorer.score(
        [
            _txn(event_id="a1", account_id="A", event_ts_us=_T0),
            _txn(event_id="b1", account_id="B", event_ts_us=_T0),
        ]
    )
    assert all(not r.had_prev_state for r in results)


def test_dry_run_reads_state_but_never_commits_or_appends() -> None:
    scorer, state, sink = _make_scorer()
    # Establish real committed state first (a normal, non-dry-run call).
    scorer.score([_txn(event_id="e1", event_ts_us=_T0)])
    assert len(state.commits) == 1
    assert len(sink.events) == 1

    (resp,) = scorer.score([_txn(event_id="e2-dry", event_ts_us=_T0 + 5_000_000)], dry_run=True)
    assert resp.had_prev_state  # reads are not "touching" state
    assert resp.seconds_since_last_txn == pytest.approx(5.0)
    assert len(state.commits) == 1  # unchanged: no write
    assert len(sink.events) == 1  # unchanged: no event appended

    # A real follow-up call must still see e1 as its predecessor, not the
    # never-committed dry-run row -- proves the dry_run truly left no trace.
    (resp2,) = scorer.score([_txn(event_id="e3", event_ts_us=_T0 + 10_000_000)])
    assert resp2.seconds_since_last_txn == pytest.approx(10.0)


def test_response_carries_the_decision_threshold_used() -> None:
    scorer, _state, _sink = _make_scorer(threshold=0.5)
    (resp,) = scorer.score([_txn(event_id="e1", event_ts_us=_T0, amount=100.0)])
    assert resp.decision == "LEGIT"
    assert resp.fraud_score == pytest.approx(0.1)


def test_degraded_ref_is_reflected_in_every_response() -> None:
    scorer, _state, _sink = _make_scorer(ref=dataclasses.replace(_REF, degraded=True))
    (resp,) = scorer.score([_txn(event_id="e1", event_ts_us=_T0)])
    assert resp.degraded


def test_reload_swaps_the_serving_model_and_ref_in_place() -> None:
    """Direct regression test for this session's "fraud_score is always the same
    constant" incident: proves the scorer actually uses the newly reloaded model,
    not a stale reference to the one it was constructed with."""
    scorer, _state, _sink = _make_scorer()
    (before,) = scorer.score([_txn(event_id="e1", event_ts_us=_T0, amount=100.0)])
    assert before.fraud_score == pytest.approx(0.1)  # _FakePipe: amount < 500
    assert before.model_version == "7"

    new_ref = dataclasses.replace(_REF, version="8")
    scorer.reload(pipe=_ConstantPipe(0.99), ref=new_ref)
    assert scorer.ref.version == "8"

    (after,) = scorer.score([_txn(event_id="e2", event_ts_us=_T0 + 1_000_000, amount=100.0)])
    assert after.fraud_score == pytest.approx(0.99)
    assert after.model_version == "8"


def test_empty_batch_returns_no_results() -> None:
    scorer, state, sink = _make_scorer()
    assert scorer.score([]) == []
    assert not state.commits
    assert not sink.events


class _ColumnRecordingPipe:
    """predict_proba records exactly which columns (and values) it was called
    with, so tests can assert on the row _predict_proba built without depending
    on a real model's schema."""

    def __init__(self) -> None:
        self.seen_columns: list[str] | None = None
        self.seen_row: dict[str, Any] | None = None
        self.seen_dtypes: dict[str, str] | None = None

    def predict_proba(self, rows: pd.DataFrame) -> Any:
        self.seen_columns = list(rows.columns)
        self.seen_row = rows.iloc[0].to_dict()
        self.seen_dtypes = {col: str(rows[col].dtype) for col in rows.columns}
        return pd.DataFrame({0: [0.9], 1: [0.1]}).to_numpy()


def test_champion_with_partial_input_schema_is_fed_the_full_declared_schema() -> None:
    """A model registered outside publish_model (e.g. a different training
    pipeline) may declare input columns that only partially overlap
    FEATURE_NAMES/gold.txn_features. _predict_proba must feed it *every*
    declared column, in the model's own declared order -- not a reduced
    subset, so a model expecting a fixed-width input never sees a truncated
    row -- filling whatever it can't compute live with NaN, and not fail."""
    pipe = _ColumnRecordingPipe()
    scorer, _state, _sink = _make_scorer()
    scorer.reload(
        pipe=pipe,
        ref=dataclasses.replace(_REF, version="9"),
        input_columns=("amount", "not_a_real_feature", "hour_of_day"),
    )
    scorer.score([_txn(event_id="e1", event_ts_us=_T0, amount=100.0)])
    assert pipe.seen_columns == ["amount", "not_a_real_feature", "hour_of_day"]
    assert pipe.seen_row is not None
    assert pipe.seen_row["amount"] == 100.0
    assert math.isnan(pipe.seen_row["not_a_real_feature"])


def test_champion_with_only_external_model_feature_overlap_is_still_pluggable() -> None:
    """A model registered outside publish_model may declare columns that don't
    overlap FEATURE_NAMES (what core.features computes live) at all, but do
    overlap gold.txn_features's EXTERNAL_MODEL_FEATURES reservations -- e.g.
    paysim-fraud-lightgbm's real champion, whose signature is
    (step, type, amount, isFlaggedFraud, log_amount, hour_sin, hour_cos). That
    model is still a gold-schema citizen (6 of 7 columns are reserved
    gold.txn_features columns) and must be servable, not refused."""
    pipe = _ColumnRecordingPipe()
    scorer, _state, _sink = _make_scorer()
    scorer.reload(
        pipe=pipe,
        ref=dataclasses.replace(_REF, version="9"),
        input_columns=(
            "step",
            "type",
            "amount",
            "isFlaggedFraud",
            "log_amount",
            "hour_sin",
            "hour_cos",
        ),
    )
    scorer.score([_txn(event_id="e1", event_ts_us=_T0, amount=100.0)])
    assert pipe.seen_columns == [
        "step",
        "type",
        "amount",
        "isFlaggedFraud",
        "log_amount",
        "hour_sin",
        "hour_cos",
    ]
    assert pipe.seen_row is not None
    assert pipe.seen_row["amount"] == 100.0
    for name in ("step", "type", "isFlaggedFraud", "log_amount", "hour_sin", "hour_cos"):
        assert math.isnan(pipe.seen_row[name])


def test_categorical_columns_are_cast_to_category_dtype_before_scoring() -> None:
    """MLflow-declared string columns must land on the pipe as pandas
    `category` dtype, not plain float64/object -- the exact fix for the real
    `paysim-fraud-lightgbm` v4 bug (LightGBM's native categorical-feature
    validation rejects a plain-float stand-in for a trained categorical
    column)."""
    pipe = _ColumnRecordingPipe()
    scorer, _state, _sink = _make_scorer()
    scorer.reload(
        pipe=pipe,
        ref=dataclasses.replace(_REF, version="9"),
        input_columns=("amount", "type"),
        categorical_columns=frozenset({"type"}),
    )
    scorer.score([_txn(event_id="e1", event_ts_us=_T0, amount=100.0)])
    assert pipe.seen_dtypes is not None
    assert pipe.seen_dtypes["type"] == "category"
    assert pipe.seen_dtypes["amount"] != "category"


class _PredictOnlyPipe:
    """Stands in for a bare mlflow.pyfunc.PyFuncModel fallback (no native
    flavor -- see serving/service.py's _load_estimator): exposes .predict()
    only, deliberately no .predict_proba, mirroring a real registration like
    paysim-fraud-xgb-baseline (flavor python_function only)."""

    def predict(self, rows: pd.DataFrame) -> Any:
        return [0.42] * len(rows)


def test_pyfunc_only_pipe_is_scored_via_predict_not_predict_proba() -> None:
    scorer, _state, _sink = _make_scorer()
    scorer.reload(
        pipe=_PredictOnlyPipe(),
        ref=dataclasses.replace(_REF, version="9"),
        input_columns=("amount",),
    )
    (resp,) = scorer.score([_txn(event_id="e1", event_ts_us=_T0, amount=100.0)])
    assert resp.fraud_score == pytest.approx(0.42)


def test_champion_with_no_gold_schema_overlap_raises_clearly() -> None:
    """A model declaring only columns gold.txn_features has no room for at all
    -- foreign to this domain entirely, not merely unfed by a live feature --
    is refused outright rather than scored with an all-NaN row."""
    pipe = _ColumnRecordingPipe()
    scorer, _state, _sink = _make_scorer()
    scorer.reload(
        pipe=pipe,
        ref=dataclasses.replace(_REF, version="9"),
        input_columns=("totally_unrelated_column",),
    )
    with pytest.raises(ValueError, match="nothing to score with"):
        scorer.score([_txn(event_id="e1", event_ts_us=_T0, amount=100.0)])


def test_real_lightgbm_categorical_booster_scores_without_raising() -> None:
    """Regression test for the real `paysim-fraud-lightgbm` v4 bug: a booster
    trained with a pandas `category`-dtype column raises
    `ValueError: train and valid dataset categorical_feature do not match`
    when handed a plain float64 stand-in at predict time. The existing
    `_ColumnRecordingPipe`-based tests never exercise this because a fake pipe
    doesn't validate dtypes at all -- this is what actually caught the bug
    the fake-pipe tests missed.

    Mirrors the real signature exactly: one numeric column (`amount`, always
    live-computed) plus one string column (`type`, an EXTERNAL_MODEL_FEATURES
    reservation `core.features` never computes live, so it arrives as NaN)
    that the real booster was trained to treat as categorical.
    """
    import lightgbm as lgb

    x_train = pd.DataFrame(
        {
            "amount": [10.0, 20.0, 30.0, 999.0, 5.0, 15.0, 500.0, 8.0],
            "type": pd.Categorical(
                [
                    "CASH_IN",
                    "CASH_OUT",
                    "DEBIT",
                    "PAYMENT",
                    "TRANSFER",
                    "CASH_IN",
                    "DEBIT",
                    "PAYMENT",
                ]
            ),
        }
    )
    y_train = [0, 0, 0, 1, 0, 0, 1, 0]
    booster = lgb.LGBMClassifier(n_estimators=5, min_child_samples=1)
    booster.fit(x_train, y_train)

    scorer, _state, _sink = _make_scorer()
    scorer.reload(
        pipe=booster,
        ref=dataclasses.replace(_REF, version="9"),
        input_columns=("amount", "type"),
        categorical_columns=frozenset({"type"}),
    )
    # amount=100.0 is live-computed; "type" is not in FEATURE_NAMES, so
    # _predict_proba fills it with NaN -- exactly what happens for the real
    # v4 model's reserved-but-unfed `type` column.
    (resp,) = scorer.score([_txn(event_id="e1", event_ts_us=_T0, amount=100.0)])
    assert 0.0 <= resp.fraud_score <= 1.0
