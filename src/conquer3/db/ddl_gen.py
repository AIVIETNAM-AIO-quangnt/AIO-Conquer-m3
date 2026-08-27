"""Generates ``db/ddl/30_gold.sql`` from ``core.schema`` -- never hand-maintained in
parallel (see ``core/schema.py``'s module docstring).

After changing ``FEATURE_NAMES`` (or a feature's Postgres type), regenerate with
``conquer3 db gen-gold-ddl`` and commit the result alongside the bump to
``FEATURE_SCHEMA_VERSION``. ``tests/unit/test_ddl_gen.py`` fails the build if the
committed file ever drifts from what this module would generate.
"""

from __future__ import annotations

from pathlib import Path

from conquer3.core.schema import FEATURE_NAMES, pg_column_type

DDL_DIR = Path(__file__).parent / "ddl"
GOLD_DDL_PATH = DDL_DIR / "30_gold.sql"

_HEADER = """\
-- GENERATED FILE -- do not hand-edit.
-- Regenerate with `conquer3 db gen-gold-ddl` after changing core/schema.py's
-- FEATURE_NAMES or pg_column_type(), then bump FEATURE_SCHEMA_VERSION and commit
-- both. tests/unit/test_ddl_gen.py enforces this file stays in sync.
--
-- Feature columns are nullable: conquer3.core.features leaves window features
-- undefined (NULL) on an account's first transaction -- see COLD_START_NULL_FEATURES
-- in core/schema.py.
CREATE TABLE IF NOT EXISTS gold.txn_features (
    event_id                 TEXT NOT NULL PRIMARY KEY,
    account_id                TEXT NOT NULL,
    event_ts_us                BIGINT NOT NULL,
    feature_schema_version     INTEGER NOT NULL,
"""

_FOOTER = """\
    is_fraud                   BOOLEAN NOT NULL,
    is_flagged_fraud           BOOLEAN NOT NULL
);
"""


def render_gold_ddl() -> str:
    """The full contents of ``30_gold.sql``, generated from ``FEATURE_NAMES``."""
    body = "".join(f"    {name:<27} {pg_column_type(name)},\n" for name in FEATURE_NAMES)
    return _HEADER + body + _FOOTER


def write_gold_ddl(path: Path = GOLD_DDL_PATH) -> None:
    path.write_text(render_gold_ddl())


if __name__ == "__main__":
    write_gold_ddl()
    print(f"wrote {GOLD_DDL_PATH}")
