"""``ops.accounts`` read/execute for the UI's single-transaction form (Layer 9).

Goes through ``db.engine.pg_connection`` -- the single Postgres entry point -- never a
second ``psycopg2.connect`` call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import psycopg2

__all__ = [
    "DEFAULT_STARTING_BALANCE",
    "AccountError",
    "InsufficientBalanceError",
    "TransferResult",
    "execute_transfer",
]

# A fresh `name_orig` auto-provisions at this balance rather than requiring a separate
# account-creation step -- the UI's ask is "type an amount and two account names", not
# "set up an account first". A fresh `name_dest`, by contrast, auto-provisions at 0:
# that mirrors PaySim's own `dest_is_new`/`orig_balance_was_zero` fraud signal in
# core/schema.py, where a brand-new destination legitimately starts at zero.
DEFAULT_STARTING_BALANCE: Final[float] = 10_000.00


class AccountError(Exception):
    """Base for ledger-validation failures."""


class InsufficientBalanceError(AccountError):
    def __init__(self, name_acc: str, amount: float, balance: float) -> None:
        self.name_acc = name_acc
        self.amount = amount
        self.balance = balance
        super().__init__(f"{name_acc!r} has balance {balance:.2f}, cannot send {amount:.2f}")


@dataclass(frozen=True, slots=True)
class TransferResult:
    oldbalance_org: float
    newbalance_orig: float
    oldbalance_dest: float
    newbalance_dest: float


def execute_transfer(
    conn: psycopg2.extensions.connection, *, name_orig: str, name_dest: str, amount: float
) -> TransferResult:
    """Validates and executes one transfer against ``ops.accounts``, atomically.

    Auto-provisions ``name_orig`` (at :data:`DEFAULT_STARTING_BALANCE`) and
    ``name_dest`` (at 0) on first use. Raises :class:`InsufficientBalanceError` --
    leaving both balances unchanged -- when ``amount`` exceeds ``name_orig``'s balance.

    Both rows are locked in a single, name-sorted ``SELECT ... FOR UPDATE`` so that two
    concurrent transfers between the same pair of accounts (in either direction) always
    acquire their locks in the same order and can never deadlock each other.
    """
    if amount <= 0:
        raise ValueError(f"amount must be positive, got {amount!r}")
    if name_orig == name_dest:
        raise ValueError(f"name_orig and name_dest must differ, got {name_orig!r} for both")

    prev_autocommit = conn.autocommit
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ops.accounts (name_acc, balance) VALUES (%s, %s) "
                "ON CONFLICT (name_acc) DO NOTHING",
                (name_orig, DEFAULT_STARTING_BALANCE),
            )
            cur.execute(
                "INSERT INTO ops.accounts (name_acc, balance) VALUES (%s, 0) "
                "ON CONFLICT (name_acc) DO NOTHING",
                (name_dest,),
            )
            cur.execute(
                "SELECT name_acc, balance FROM ops.accounts "
                "WHERE name_acc = ANY(%s) ORDER BY name_acc FOR UPDATE",
                ([name_orig, name_dest],),
            )
            balances = {name: float(balance) for name, balance in cur.fetchall()}
            oldbalance_org = balances[name_orig]
            oldbalance_dest = balances[name_dest]

            if amount > oldbalance_org:
                raise InsufficientBalanceError(name_orig, amount, oldbalance_org)

            newbalance_orig = oldbalance_org - amount
            newbalance_dest = oldbalance_dest + amount

            cur.execute(
                "UPDATE ops.accounts SET balance = %s, updated_at = now() WHERE name_acc = %s",
                (newbalance_orig, name_orig),
            )
            cur.execute(
                "UPDATE ops.accounts SET balance = %s, updated_at = now() WHERE name_acc = %s",
                (newbalance_dest, name_dest),
            )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.autocommit = prev_autocommit

    return TransferResult(
        oldbalance_org=oldbalance_org,
        newbalance_orig=newbalance_orig,
        oldbalance_dest=oldbalance_dest,
        newbalance_dest=newbalance_dest,
    )
