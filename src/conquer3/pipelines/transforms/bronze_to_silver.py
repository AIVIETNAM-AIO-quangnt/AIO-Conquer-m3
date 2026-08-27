"""Bronze -> silver: typing, cleaning, and deriving ``event_ts_us``.

The ``event_ts_us`` formula below is DuckDB SQL reproducing
``conquer3.core.timeref.derive_event_ts_us`` exactly (integer floor-division, using
the same ``SIM_EPOCH_US``/``US_PER_HOUR`` constants) -- see that module's docstring
for why this must stay bit-for-bit identical, and
``tests/parity/test_event_ts_us_sql.py`` for the check that it is.

Full refresh: TRUNCATEs ``silver.txn`` before reloading, matching bronze's semantics.
"""

from __future__ import annotations

from conquer3.core.timeref import SIM_EPOCH_US, US_PER_HOUR
from conquer3.db import ops
from conquer3.db.engine import get_ibis_connection, pg_connection

__all__ = ["bronze_to_silver"]

_TRANSFORM_SQL = f"""
INSERT INTO pg.silver.txn (
    event_id, account_id, dest_id, txn_type, amount,
    oldbalance_org, newbalance_orig, oldbalance_dest, newbalance_dest,
    step, event_ts_us, is_fraud, is_flagged_fraud, bronze_row_num
)
SELECT
    'ps-' || lpad(row_num::VARCHAR, 10, '0') AS event_id,
    name_orig AS account_id,
    name_dest AS dest_id,
    type AS txn_type,
    amount,
    oldbalance_org,
    newbalance_orig,
    oldbalance_dest,
    newbalance_dest,
    step,
    CAST({SIM_EPOCH_US} AS BIGINT)
        + CAST(step - 1 AS BIGINT) * CAST({US_PER_HOUR} AS BIGINT)
        + (CAST(intra_step_seq - 1 AS BIGINT) * CAST({US_PER_HOUR} AS BIGINT))
          // CAST(GREATEST(step_cardinality, 1) AS BIGINT) AS event_ts_us,
    is_fraud,
    is_flagged_fraud,
    row_num AS bronze_row_num
FROM (
    SELECT
        *,
        row_number() OVER (PARTITION BY step ORDER BY row_num) AS intra_step_seq,
        count(*) OVER (PARTITION BY step) AS step_cardinality
    FROM pg.bronze.txn_raw
) staged
"""


def bronze_to_silver() -> int:
    """TRUNCATEs ``silver.txn``, then transforms all of ``bronze.txn_raw`` into it.

    Returns the row count written.
    """
    with pg_connection() as conn, ops.track_run(conn, layer="bronze_to_silver") as run:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM bronze.txn_raw")
            row = cur.fetchone()
            assert row is not None
            run.rows_in = row[0]
            cur.execute("TRUNCATE silver.txn")

        duck = get_ibis_connection()
        try:
            duck.raw_sql(_TRANSFORM_SQL)
            row_count = int(duck.raw_sql("SELECT count(*) FROM pg.silver.txn").fetchone()[0])
        finally:
            duck.disconnect()

        run.rows_out = row_count

    return row_count
