"""The committed db/ddl/30_gold.sql must never drift from core.schema."""

from __future__ import annotations

import re

from conquer3.core.schema import FEATURE_NAMES
from conquer3.db.ddl_gen import GOLD_DDL_PATH, render_gold_ddl

_COLUMN_NAME_RE = re.compile(r"^\s{4}(\w+)\s", re.MULTILINE)


def test_committed_gold_ddl_matches_generator() -> None:
    assert GOLD_DDL_PATH.read_text() == render_gold_ddl(), (
        "db/ddl/30_gold.sql is out of date -- run `conquer3 db gen-gold-ddl` and commit it"
    )


def test_rendered_ddl_has_one_column_per_feature() -> None:
    columns = set(_COLUMN_NAME_RE.findall(render_gold_ddl()))
    assert set(FEATURE_NAMES) <= columns
