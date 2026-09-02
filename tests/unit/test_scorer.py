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


def test_empty_batch_returns_no_results() -> None:
    scorer, state, sink = _make_scorer()
    assert scorer.score([]) == []
    assert not state.commits
    assert not sink.events
