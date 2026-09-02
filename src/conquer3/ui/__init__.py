"""Streamlit console (Layer 9): a client of the scorer, never a second scorer.

Every module in this package holds no model and computes no feature -- see the
"ui talks to serving over HTTP, never by import" import-linter contract. Launched
via ``conquer3 ui`` (``cli.py``), which shells out to ``streamlit run ui/app.py``.
"""

from __future__ import annotations
