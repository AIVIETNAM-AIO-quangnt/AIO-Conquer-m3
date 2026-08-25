"""The three data shapes every consumer agrees on.

stdlib + typing only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from conquer3.core.schema import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, STATE_SCHEMA_VERSION

FeatureValue = float | int | str | None


class TxnType(StrEnum):
    """PaySim transaction types."""

    CASH_IN = "CASH_IN"
    CASH_OUT = "CASH_OUT"
    DEBIT = "DEBIT"
    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"


# In PaySim, fraud only ever occurs in these two types. Useful as a feature and as a
# data-quality assertion, but note the model still scores every type.
FRAUD_CAPABLE_TYPES: Final[frozenset[str]] = frozenset({TxnType.CASH_OUT, TxnType.TRANSFER})

# PaySim merchant accounts are prefixed 'M'.
MERCHANT_PREFIX: Final[str] = "M"


@dataclass(frozen=True, slots=True)
class TransactionEvent:
    """One transaction, as it arrives at the scorer.

    ``event_ts_us`` is always supplied by the caller -- never read from the clock in
    here -- so that batch and online paths are bit-identical.
    """

    event_id: str
    account_id: str  # PaySim nameOrig
    dest_id: str  # PaySim nameDest
    txn_type: str
    amount: float
    oldbalance_org: float
    newbalance_orig: float
    oldbalance_dest: float
    newbalance_dest: float
    event_ts_us: int
    step: int

    @property
    def dest_is_merchant(self) -> bool:
        return self.dest_id.startswith(MERCHANT_PREFIX)


@dataclass(frozen=True, slots=True)
class AccountState:
    """Everything needed to featurize the NEXT transaction for one account.

    Split into two halves with different merge semantics:

    * the ``last_*`` anchor block -- resolved by the total order
      ``(last_event_ts_us, last_event_id)``; and
    * the running priors -- associative aggregates (sum/count/max/min).

    That split is what makes the Pathway accumulator's ``update()`` associative, and
    therefore what makes streaming state agree with the sequential batch fold. See
    ``pipelines/pathway/accumulators.py``.
    """

    account_id: str
    state_version: int = STATE_SCHEMA_VERSION

    # -- last transaction: the "window" anchor -----------------------------
    last_event_id: str = ""
    last_event_ts_us: int = 0
    last_step: int = 0
    last_txn_type: str = ""
    last_amount: float = 0.0
    last_oldbalance_org: float = 0.0
    last_newbalance_orig: float = 0.0
    last_dest_id: str = ""
    last_dest_is_merchant: bool = False

    # -- running priors: associative, order-independent ---------------------
    first_event_ts_us: int = 0
    txn_count: int = 0
    amount_sum: float = 0.0
    amount_sqsum: float = 0.0
    max_amount: float = 0.0

    # -- feedback loop ------------------------------------------------------
    last_fraud_score: float | None = None
    updated_at_us: int = 0

    @property
    def order_key(self) -> tuple[int, str]:
        """Total order used to pick the winning ``last_*`` block on merge."""
        return (self.last_event_ts_us, self.last_event_id)


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """Computed features for one transaction.

    ``values`` is keyed by :data:`conquer3.core.schema.FEATURE_NAMES`. Holding a
    mapping rather than 34 explicit dataclass fields keeps FEATURE_NAMES the single
    source of truth -- there is no parallel list to drift out of sync.
    """

    event_id: str
    account_id: str
    event_ts_us: int
    values: Mapping[str, FeatureValue] = field(default_factory=dict)
    feature_schema_version: int = FEATURE_SCHEMA_VERSION

    def model_inputs(self) -> dict[str, FeatureValue]:
        """Only the model's input columns, in FEATURE_NAMES order.

        Never includes identity fields -- passing ``account_id`` to the model would
        both leak and fail to generalise.
        """
        return {name: self.values[name] for name in FEATURE_NAMES}

    def as_dict(self) -> dict[str, FeatureValue]:
        """Identity + features, for JSONL logging and the gold table."""
        out: dict[str, FeatureValue] = {
            "event_id": self.event_id,
            "account_id": self.account_id,
            "event_ts_us": self.event_ts_us,
            "feature_schema_version": self.feature_schema_version,
        }
        out.update(self.model_inputs())
        return out
