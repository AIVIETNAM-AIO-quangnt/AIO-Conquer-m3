"""Landing raw PaySim1 CSV rows into ``bronze.txn_raw``.

Full refresh: PaySim1 is a static one-shot dataset (not an incrementally-arriving
stream -- that's Layer 3b/Pathway's job), so each run TRUNCATEs before loading.
"""

from __future__ import annotations

from pathlib import Path

from conquer3.db import ops
from conquer3.db.engine import get_ibis_connection, pg_connection

__all__ = ["load_csv_to_bronze"]

_INSERT_SQL = """
INSERT INTO pg.bronze.txn_raw (
    row_num, step, type, amount, name_orig, oldbalance_org, newbalance_orig,
    name_dest, oldbalance_dest, newbalance_dest, is_fraud, is_flagged_fraud, source_file
)
SELECT
    row_number() OVER () AS row_num,
    step, type, amount,
    nameOrig AS name_orig, oldbalanceOrg AS oldbalance_org, newbalanceOrig AS newbalance_orig,
    nameDest AS name_dest, oldbalanceDest AS oldbalance_dest, newbalanceDest AS newbalance_dest,
    isFraud::BOOLEAN AS is_fraud, isFlaggedFraud::BOOLEAN AS is_flagged_fraud,
    ? AS source_file
FROM read_csv_auto(?)
"""


def load_csv_to_bronze(csv_path: Path | str) -> int:
    """TRUNCATEs ``bronze.txn_raw``, then bulk-loads ``csv_path`` into it.

    Returns the row count loaded.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"PaySim1 CSV not found at {csv_path}")

    with pg_connection() as conn, ops.track_run(conn, layer="bronze_ingest") as run:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE bronze.txn_raw")

        duck = get_ibis_connection()
        try:
            duck.con.execute(_INSERT_SQL, [csv_path.name, str(csv_path)])
            row = duck.con.execute("SELECT count(*) FROM pg.bronze.txn_raw").fetchone()
            assert row is not None
            row_count = int(row[0])
        finally:
            duck.disconnect()

        run.rows_in = row_count
        run.rows_out = row_count

    return row_count
