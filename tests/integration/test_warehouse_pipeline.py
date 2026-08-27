"""End-to-end Layer 2 pipeline against an ephemeral Postgres: ingest -> bronze ->
silver -> gold. The real correctness check: gold must match calling
conquer3.core.features.compute_sequence directly on the same synthetic events,
independently re-deriving event_ts_us/event_id the same way
pipelines/transforms/bronze_to_silver.py's SQL does (that formula has its own
dedicated parity test in tests/parity/test_event_ts_us_sql.py).

Needs Docker (testcontainers spins up a real postgres:17-alpine container).
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("testcontainers")

from testcontainers.community.postgres import PostgresContainer

from conquer3.config.settings import get_settings
from conquer3.core import timeref
from conquer3.core.features import compute_sequence
from conquer3.core.types import TransactionEvent
from conquer3.db.bootstrap import apply_ddl
from conquer3.db.engine import pg_connection
from conquer3.pipelines.ingest.bronze import load_csv_to_bronze
from conquer3.pipelines.transforms.bronze_to_silver import bronze_to_silver
from conquer3.pipelines.transforms.silver_to_gold import silver_to_gold

pytestmark = pytest.mark.integration

_CSV_HEADER = [
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
]

# Two accounts (C100, C101) with multiple transactions each, sharing steps 1 and 2
# so the intra-step tiebreak in event_ts_us is actually exercised.
_ROWS: list[tuple[int, str, float, str, float, float, str, float, float, int, int]] = [
    (1, "PAYMENT", 9839.64, "C100", 170136.0, 160296.36, "M200", 0.0, 0.0, 0, 0),
    (1, "PAYMENT", 1864.28, "C101", 21249.0, 19384.72, "M201", 0.0, 0.0, 0, 0),
    (1, "TRANSFER", 181.0, "C100", 160296.36, 0.0, "C555", 0.0, 181.0, 1, 0),
    (2, "CASH_OUT", 5000.0, "C100", 0.0, 0.0, "C777", 2000.0, 7000.0, 0, 0),
    (2, "CASH_OUT", 181.0, "C101", 19384.72, 19203.72, "C888", 21182.0, 21363.72, 1, 0),
    (3, "DEBIT", 250.5, "C101", 19203.72, 18953.22, "M300", 0.0, 0.0, 0, 0),
]


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    path = tmp_path / "paysim_sample.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_CSV_HEADER)
        writer.writerows(_ROWS)
    return path


@pytest.fixture
def warehouse(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    with PostgresContainer("postgres:17-alpine") as pg:
        monkeypatch.setenv("POSTGRES_HOST", pg.get_container_host_ip())
        monkeypatch.setenv("POSTGRES_PORT", str(pg.get_exposed_port(5432)))
        monkeypatch.setenv("POSTGRES_DB", pg.dbname)
        monkeypatch.setenv("POSTGRES_USER", pg.username)
        monkeypatch.setenv("POSTGRES_PASSWORD", pg.password)
        monkeypatch.setenv("C3_DUCKDB_PATH", str(tmp_path / "analytics.duckdb"))
        monkeypatch.setenv("C3_DUCKDB_TEMP_DIR", str(tmp_path / "duckdb_tmp"))
        get_settings.cache_clear()
        try:
            with pg_connection() as conn:
                apply_ddl(conn)
            yield
        finally:
            get_settings.cache_clear()


def _expected_features_by_event_id() -> dict[str, dict[str, object]]:
    """The independent oracle: derive event_ts_us/event_id the way bronze_to_silver's
    SQL does, then fold each account's events through core.features directly."""
    rows_by_step: dict[int, list[int]] = {}
    for i, row in enumerate(_ROWS, start=1):
        rows_by_step.setdefault(row[0], []).append(i)

    by_account: dict[str, list[TransactionEvent]] = {}
    for i, row in enumerate(_ROWS, start=1):
        step, txn_type, amount, name_orig, old_org, new_org, name_dest, old_dest, new_dest = row[:9]
        seq = rows_by_step[step].index(i) + 1
        card = len(rows_by_step[step])
        txn = TransactionEvent(
            event_id=f"ps-{i:010d}",
            account_id=name_orig,
            dest_id=name_dest,
            txn_type=txn_type,
            amount=amount,
            oldbalance_org=old_org,
            newbalance_orig=new_org,
            oldbalance_dest=old_dest,
            newbalance_dest=new_dest,
            event_ts_us=timeref.derive_event_ts_us(step, seq, card),
            step=step,
        )
        by_account.setdefault(txn.account_id, []).append(txn)

    expected: dict[str, dict[str, object]] = {}
    for txns in by_account.values():
        ordered = sorted(txns, key=lambda t: (t.event_ts_us, t.event_id))
        for features, _state in compute_sequence(ordered):
            expected[features.event_id] = features.as_dict()
    return expected


def test_pipeline_matches_core_features_directly(warehouse: None, csv_path: Path) -> None:
    assert load_csv_to_bronze(csv_path) == len(_ROWS)
    assert bronze_to_silver() == len(_ROWS)
    assert silver_to_gold() == len(_ROWS)

    expected = _expected_features_by_event_id()

    with pg_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT event_id, account_id, event_ts_us, amount, is_first_txn, "
            "txn_count_prior, is_fraud, is_flagged_fraud FROM gold.txn_features"
        )
        actual = {r[0]: r for r in cur.fetchall()}

    assert set(actual) == set(expected)

    fraud_by_event_id = {f"ps-{i:010d}": bool(row[9]) for i, row in enumerate(_ROWS, start=1)}
    flagged_by_event_id = {f"ps-{i:010d}": bool(row[10]) for i, row in enumerate(_ROWS, start=1)}

    for event_id, exp in expected.items():
        _, account_id, event_ts_us, amount, is_first_txn, txn_count_prior, is_fraud, is_flagged = (
            actual[event_id]
        )
        assert account_id == exp["account_id"]
        assert event_ts_us == exp["event_ts_us"]
        assert amount == pytest.approx(exp["amount"])
        assert is_first_txn == exp["is_first_txn"]
        assert txn_count_prior == exp["txn_count_prior"]
        assert is_fraud == fraud_by_event_id[event_id]
        assert is_flagged == flagged_by_event_id[event_id]
