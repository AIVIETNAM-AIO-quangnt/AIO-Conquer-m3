"""FraudScorerModel.predict() logic, tested directly against fakes for the Redis
state store and event sink -- bypasses load_context (which needs a real MLflow
artifact + Redis + the event dir) so this stays fast and infra-free, matching how
core.features is tested directly. The full stack (real MLflow, real Redis, the
actual scoring server) is proven in tests/integration/test_serving_e2e.py; this
file is where FraudScorerModel's own batching/dry_run/decision logic gets its
coverage.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from conquer3.contracts.model_registry import ModelRef
from conquer3.core.types import AccountState
from conquer3.serving.pyfunc_model import FraudScorerModel

_REF = ModelRef(name="m", version="7", run_id="r1", alias="champion", tags={}, degraded=False)


class _FakePipe:
    """predict_proba: a fixed, known score -- deterministic on 'amount' so tests
    can pin exact expected decisions without depending on real sklearn output."""

    def predict_proba(self, rows: pd.DataFrame) -> pd.DataFrame:
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


def _make_model(
    *, threshold: float = 0.5, ref: ModelRef = _REF
) -> tuple[FraudScorerModel, _FakeStateStore, _FakeEventSink]:
    model = FraudScorerModel()
    model._pipe = _FakePipe()  # type: ignore[attr-defined]
    model._ref = ref  # type: ignore[attr-defined]
    model._threshold = threshold  # type: ignore[attr-defined]
    state = _FakeStateStore()
    sink = _FakeEventSink()
    model._state = state  # type: ignore[attr-defined]
    model._sink = sink  # type: ignore[attr-defined]
    return model, state, sink


def _row(*, event_id: str, account_id: str = "C1", amount: float = 100.0, event_ts_us: int) -> dict:
    return {
        "event_id": event_id,
        "account_id": account_id,
        "dest_id": "C900",
        "txn_type": "TRANSFER",
        "amount": amount,
        "oldbalance_org": 1000.0,
        "newbalance_orig": 1000.0 - amount,
        "oldbalance_dest": 0.0,
        "newbalance_dest": amount,
        "event_ts_us": event_ts_us,
        "step": 1,
    }


_T0 = 1_700_000_000_000_000


def test_model_info_returns_the_resolved_ref_ignoring_model_input() -> None:
    model, _state, _sink = _make_model()
    row = pd.DataFrame([_row(event_id="ignored", event_ts_us=_T0)])
    out = model.predict(None, row, {"op": "model_info"})
    assert len(out) == 1
    row = out.iloc[0].to_dict()
    assert row["name"] == "m"
    assert row["version"] == "7"
    assert not row["degraded"]


def test_unknown_op_raises_value_error() -> None:
    model, _state, _sink = _make_model()
    with pytest.raises(ValueError, match="unknown op"):
        model.predict(None, pd.DataFrame([_row(event_id="e1", event_ts_us=_T0)]), {"op": "bogus"})


def test_first_transaction_is_cold_start_and_gets_committed_and_appended() -> None:
    model, state, sink = _make_model()
    out = model.predict(None, pd.DataFrame([_row(event_id="e1", event_ts_us=_T0, amount=100.0)]))
    resp = out.iloc[0].to_dict()

    assert not resp["had_prev_state"]
    assert resp["seconds_since_last_txn"] is None
    assert resp["decision"] == "LEGIT"  # amount=100 -> proba 0.1 < threshold 0.5
    assert resp["model_version"] == "7"
    assert not resp["degraded"]
    assert len(state.commits) == 1
    assert len(sink.events) == 1


def test_second_request_sees_state_committed_by_the_first() -> None:
    model, _state, _sink = _make_model()
    model.predict(None, pd.DataFrame([_row(event_id="e1", event_ts_us=_T0)]))
    out2 = model.predict(None, pd.DataFrame([_row(event_id="e2", event_ts_us=_T0 + 5_000_000)]))
    resp2 = out2.iloc[0].to_dict()
    assert resp2["had_prev_state"]
    assert resp2["seconds_since_last_txn"] == pytest.approx(5.0)


def test_decision_flips_to_fraud_above_threshold() -> None:
    model, _state, _sink = _make_model(threshold=0.5)
    out = model.predict(None, pd.DataFrame([_row(event_id="e1", event_ts_us=_T0, amount=999.0)]))
    resp = out.iloc[0].to_dict()
    assert resp["fraud_score"] == pytest.approx(0.9)
    assert resp["decision"] == "FRAUD"


def test_batch_same_account_folds_sequentially_within_one_call() -> None:
    """Two rows for the same account in ONE predict() call must see each other's
    state even though the state store starts empty -- plan §8.4's batch semantics,
    independent of anything Redis persisted across separate requests."""
    model, state, sink = _make_model()
    rows = [
        _row(event_id="b1", event_ts_us=_T0),
        _row(event_id="b2", event_ts_us=_T0 + 5_000_000),
    ]
    out = model.predict(None, pd.DataFrame(rows))
    resp1, resp2 = out.iloc[0].to_dict(), out.iloc[1].to_dict()

    assert not resp1["had_prev_state"]
    assert resp2["had_prev_state"]
    assert resp2["seconds_since_last_txn"] == pytest.approx(5.0)
    # Exactly one commit per row, and the account's final committed state is the
    # one folded from the second (later) row, not the first.
    assert len(state.commits) == 2
    assert state.data["C1"].last_event_id == "b2"
    assert len(sink.events) == 2


def test_batch_out_of_request_order_is_folded_in_event_ts_us_order() -> None:
    """Rows for one account must be folded by event_ts_us, not by array position --
    the plan's ordering guarantee, not merely "first row wins"."""
    model, state, _sink = _make_model()
    rows = [
        _row(event_id="later", event_ts_us=_T0 + 5_000_000),
        _row(event_id="earlier", event_ts_us=_T0),
    ]
    out = model.predict(None, pd.DataFrame(rows))
    by_id = {r["event_id"]: r for r in out.to_dict(orient="records")}
    assert not by_id["earlier"]["had_prev_state"]
    assert by_id["later"]["had_prev_state"]
    assert by_id["later"]["seconds_since_last_txn"] == pytest.approx(5.0)
    assert state.data["C1"].last_event_id == "later"


def test_batch_different_accounts_are_independent() -> None:
    model, _state, _sink = _make_model()
    rows = [
        _row(event_id="a1", account_id="A", event_ts_us=_T0),
        _row(event_id="b1", account_id="B", event_ts_us=_T0),
    ]
    out = model.predict(None, pd.DataFrame(rows))
    assert all(not r["had_prev_state"] for r in out.to_dict(orient="records"))


def test_dry_run_reads_state_but_never_commits_or_appends() -> None:
    model, state, sink = _make_model()
    # Establish real committed state first (a normal, non-dry-run call).
    model.predict(None, pd.DataFrame([_row(event_id="e1", event_ts_us=_T0)]))
    assert len(state.commits) == 1
    assert len(sink.events) == 1

    out = model.predict(
        None,
        pd.DataFrame([_row(event_id="e2-dry", event_ts_us=_T0 + 5_000_000)]),
        {"dry_run": True},
    )
    resp = out.iloc[0].to_dict()
    assert resp["had_prev_state"]  # reads are not "touching" state
    assert resp["seconds_since_last_txn"] == pytest.approx(5.0)
    assert len(state.commits) == 1  # unchanged: no write
    assert len(sink.events) == 1  # unchanged: no event appended

    # A real follow-up call must still see e1 as its predecessor, not the
    # never-committed dry-run row -- proves the dry_run truly left no trace.
    out2 = model.predict(None, pd.DataFrame([_row(event_id="e3", event_ts_us=_T0 + 10_000_000)]))
    assert out2.iloc[0]["seconds_since_last_txn"] == pytest.approx(10.0)


def test_response_carries_the_decision_threshold_used() -> None:
    model, _state, _sink = _make_model(threshold=0.5)
    out = model.predict(None, pd.DataFrame([_row(event_id="e1", event_ts_us=_T0, amount=100.0)]))
    resp = out.iloc[0].to_dict()
    assert resp["decision"] == "LEGIT"
    assert resp["fraud_score"] == pytest.approx(0.1)


def test_degraded_ref_is_reflected_in_every_response() -> None:
    degraded_ref = dataclasses.replace(_REF, degraded=True)
    model, _state, _sink = _make_model(ref=degraded_ref)
    out = model.predict(None, pd.DataFrame([_row(event_id="e1", event_ts_us=_T0)]))
    assert bool(out.iloc[0]["degraded"])
