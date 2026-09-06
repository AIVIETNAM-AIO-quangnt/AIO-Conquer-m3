"""ops.accounts against a real ephemeral Postgres (testcontainers, same fixture
philosophy as test_warehouse_pipeline.py) -- auto-provisioning, balance
validation/execution atomicity, and the sorted-lock-order deadlock guard.

Needs Docker (testcontainers spins up a real postgres:17-alpine container).
"""

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest

pytest.importorskip("testcontainers")

from testcontainers.community.postgres import PostgresContainer

from conquer3.config.settings import get_settings
from conquer3.db.accounts import (
    DEFAULT_STARTING_BALANCE,
    InsufficientBalanceError,
    execute_transfer,
)
from conquer3.db.bootstrap import apply_ddl
from conquer3.db.engine import pg_connection

pytestmark = pytest.mark.integration


@pytest.fixture
def ledger(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    with PostgresContainer("postgres:17-alpine") as pg:
        monkeypatch.setenv("POSTGRES_HOST", pg.get_container_host_ip())
        monkeypatch.setenv("POSTGRES_PORT", str(pg.get_exposed_port(5432)))
        monkeypatch.setenv("POSTGRES_DB", pg.dbname)
        monkeypatch.setenv("POSTGRES_USER", pg.username)
        monkeypatch.setenv("POSTGRES_PASSWORD", pg.password)
        # The testcontainers postgres:17-alpine image has no SSL configured.
        monkeypatch.setenv("POSTGRES_SSLMODE", "disable")
        monkeypatch.setenv("POSTGRES_CHANNEL_BINDING", "disable")
        get_settings.cache_clear()
        try:
            with pg_connection() as conn:
                apply_ddl(conn)
            yield
        finally:
            get_settings.cache_clear()


def _balance(name_acc: str) -> float:
    with pg_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT balance FROM ops.accounts WHERE name_acc = %s", (name_acc,))
        row = cur.fetchone()
        assert row is not None
        return float(row[0])


def test_first_transfer_auto_provisions_both_accounts(ledger: None) -> None:
    with pg_connection() as conn:
        result = execute_transfer(conn, name_orig="C1", name_dest="M900", amount=181.0)

    assert result.oldbalance_org == DEFAULT_STARTING_BALANCE
    assert result.newbalance_orig == DEFAULT_STARTING_BALANCE - 181.0
    assert result.oldbalance_dest == 0.0
    assert result.newbalance_dest == 181.0
    assert _balance("C1") == DEFAULT_STARTING_BALANCE - 181.0
    assert _balance("M900") == 181.0


def test_balances_persist_and_compound_across_transfers(ledger: None) -> None:
    with pg_connection() as conn:
        first = execute_transfer(conn, name_orig="C1", name_dest="M900", amount=181.0)
    with pg_connection() as conn:
        second = execute_transfer(conn, name_orig="C1", name_dest="M901", amount=100.0)

    assert second.oldbalance_org == first.newbalance_orig
    assert second.newbalance_orig == first.newbalance_orig - 100.0
    assert _balance("C1") == DEFAULT_STARTING_BALANCE - 181.0 - 100.0


def test_amount_over_balance_is_rejected_with_no_partial_write(ledger: None) -> None:
    too_much = DEFAULT_STARTING_BALANCE + 1.0
    with pg_connection() as conn, pytest.raises(InsufficientBalanceError) as exc_info:
        execute_transfer(conn, name_orig="C1", name_dest="M900", amount=too_much)
    # The error still reports the real (in-transaction) balance the attempt saw.
    assert f"{DEFAULT_STARTING_BALANCE:.2f}" in str(exc_info.value)

    # The whole attempt rolled back atomically -- even the auto-provisioning inserts
    # that ran earlier in the same transaction were undone, so neither account exists.
    with pg_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM ops.accounts WHERE name_acc IN ('C1', 'M900')")
        row = cur.fetchone()
        assert row is not None
        assert row[0] == 0


def test_opposite_direction_concurrent_transfers_never_deadlock_and_conserve_the_total(
    ledger: None,
) -> None:
    # One real transfer first, so both accounts exist and C2 has something to send
    # back -- auto-provisioning itself isn't what this test is exercising.
    with pg_connection() as conn:
        execute_transfer(conn, name_orig="C1", name_dest="C2", amount=1.0)

    total_before = _balance("C1") + _balance("C2")
    errors: list[BaseException] = []

    def _run(name_orig: str, name_dest: str) -> None:
        for _ in range(20):
            try:
                with pg_connection() as conn:
                    execute_transfer(conn, name_orig=name_orig, name_dest=name_dest, amount=1.0)
            except InsufficientBalanceError:
                # Expected under contention: the other thread may not have credited
                # the sending side yet. Not the invariant this test is checking.
                pass
            except BaseException as exc:
                errors.append(exc)

    t1 = threading.Thread(target=_run, args=("C1", "C2"))
    t2 = threading.Thread(target=_run, args=("C2", "C1"))
    t1.start()
    t2.start()
    t1.join(timeout=30)
    t2.join(timeout=30)

    assert not t1.is_alive() and not t2.is_alive(), "a transfer deadlocked"
    assert not errors, errors
    assert _balance("C1") + _balance("C2") == total_before
