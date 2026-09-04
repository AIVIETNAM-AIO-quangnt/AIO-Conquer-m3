"""Applies the Layer 2 DDL files to a running Postgres.

``docker-entrypoint-initdb.d`` only runs on a fresh volume, so this is the only way
to get the schema onto a Postgres that already has data in it (e.g. the Layer 1
volume). Every statement in ``db/ddl`` is ``CREATE ... IF NOT EXISTS``, so this is
idempotent -- safe to call from the CLI, the smoke gate, and integration tests alike.
"""

from __future__ import annotations

from pathlib import Path

import psycopg2

from conquer3.db.ddl_gen import DDL_DIR

__all__ = ["apply_ddl"]


def apply_ddl(conn: psycopg2.extensions.connection, *, ddl_dir: Path = DDL_DIR) -> list[str]:
    """Runs every ``*.sql`` file in ``ddl_dir``, in filename order.

    Returns the filenames applied, in the order they ran.
    """
    applied = []
    for sql_file in sorted(ddl_dir.glob("*.sql")):
        with conn.cursor() as cur:
            cur.execute(sql_file.read_text())
        applied.append(sql_file.name)
    return applied
