"""Generates ``db/ddl/30_gold.sql`` from ``core.schema`` -- never hand-maintained in
parallel (see ``core/schema.py``'s module docstring).

After changing ``FEATURE_NAMES`` (or a feature's Postgres type), regenerate with
``conquer3 db gen-gold-ddl`` and commit the result alongside the bump to
``FEATURE_SCHEMA_VERSION``. ``tests/unit/test_ddl_gen.py`` fails the build if the
committed file ever drifts from what this module would generate.

Also emits ``EXTERNAL_MODEL_FEATURES`` as a second, nullable block reserved for other
MLflow-registered models (see that constant's docstring in ``core/schema.py``).
Changing that list needs the same regenerate-and-commit step, but never a
``FEATURE_SCHEMA_VERSION`` bump -- it isn't part of the served model's contract.
"""

from __future__ import annotations

from pathlib import Path

from conquer3.core.schema import EXTERNAL_MODEL_FEATURES, FEATURE_NAMES, pg_column_type

DDL_DIR = Path(__file__).parent / "ddl"
GOLD_DDL_PATH = DDL_DIR / "30_gold.sql"

_HEADER = """\
-- GENERATED FILE -- do not hand-edit.
-- Regenerate with `conquer3 db gen-gold-ddl` after changing core/schema.py's
-- FEATURE_NAMES, pg_column_type(), or EXTERNAL_MODEL_FEATURES, then commit the
-- result (bump FEATURE_SCHEMA_VERSION too if FEATURE_NAMES changed).
-- tests/unit/test_ddl_gen.py enforces this file stays in sync.
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

_EXTERNAL_HEADER = """\
    --
    -- Below: reserved for other MLflow-registered models' own feature names (see
    -- core/schema.py's EXTERNAL_MODEL_FEATURES). conquer3.core.features never
    -- computes these and the champion scorer never reads them -- always NULL until
    -- a job backfills that specific model's features.
"""

_FOOTER = """\
    is_fraud                   BOOLEAN NOT NULL,
    is_flagged_fraud           BOOLEAN NOT NULL
);
"""


def render_gold_ddl() -> str:
    """The full contents of ``30_gold.sql``: ``FEATURE_NAMES`` plus the reserved
    ``EXTERNAL_MODEL_FEATURES`` columns."""
    body = "".join(f"    {name:<27} {pg_column_type(name)},\n" for name in FEATURE_NAMES)
    external = "".join(f"    {name:<27} {pg_type},\n" for name, pg_type in EXTERNAL_MODEL_FEATURES)
    return _HEADER + body + _EXTERNAL_HEADER + external + _FOOTER


def write_gold_ddl(path: Path = GOLD_DDL_PATH) -> None:
    path.write_text(render_gold_ddl())


if __name__ == "__main__":
    write_gold_ddl()
    print(f"wrote {GOLD_DDL_PATH}")
