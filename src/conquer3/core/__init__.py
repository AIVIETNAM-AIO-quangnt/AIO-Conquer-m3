"""Tier-0: the dependency-light feature core.

Everything in this subpackage may import **stdlib and typing only**. It is the one
module that ships to Google Colab via ``pip install "conquer3[train] @ git+..."``,
and it is the single place a feature may be computed.

Import submodules explicitly (``from conquer3.core import features``); this file
deliberately re-exports nothing so that importing one submodule does not pull the
rest.
"""
