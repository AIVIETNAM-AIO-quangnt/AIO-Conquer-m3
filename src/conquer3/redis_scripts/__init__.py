"""Shared Lua scripts for Redis, used identically by Pathway (Layer 3b) and BentoML
(Layer 5, not yet built) -- so the monotonic-CAS write semantics can never drift
between the two writers of account state. stdlib only: no redis import here, since
this package only hands back text; callers build their own redis.Redis client and
call `.register_script(load_script(...))`.
"""

from __future__ import annotations

from importlib import resources

__all__ = ["load_script"]


def load_script(name: str) -> str:
    """Reads a bundled ``.lua`` script by name (without extension)."""
    return resources.files(__package__).joinpath(f"{name}.lua").read_text(encoding="utf-8")
