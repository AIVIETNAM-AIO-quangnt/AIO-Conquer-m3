"""Serialisation for AccountState, plus the Redis key layout.

stdlib ``json`` only -- this crosses the BentoML/Pathway boundary, so both sides must
agree byte-for-byte without a shared heavyweight dependency.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from conquer3.core.schema import STATE_SCHEMA_VERSION
from conquer3.core.types import AccountState, FeatureVector

__all__ = [
    "features_to_row",
    "redis_state_key",
    "state_from_dict",
    "state_from_json",
    "state_to_dict",
    "state_to_json",
]


class StateVersionMismatchError(ValueError):
    """A stored state document was written by a different STATE_SCHEMA_VERSION."""


def state_to_dict(state: AccountState) -> dict[str, Any]:
    return asdict(state)


def state_from_dict(payload: Mapping[str, Any]) -> AccountState:
    """Rebuild an AccountState, rejecting documents from another schema version.

    Raises:
        StateVersionMismatchError: if the document's version differs from the running one.
    """
    version = payload.get("state_version")
    if version != STATE_SCHEMA_VERSION:
        raise StateVersionMismatchError(
            f"state_version {version!r} != running {STATE_SCHEMA_VERSION}"
        )
    known = {f: payload[f] for f in AccountState.__slots__ if f in payload}
    return AccountState(**known)


def state_to_json(state: AccountState) -> str:
    # separators: compact, and stable across Python versions so that two writers
    # produce identical bytes for identical state.
    return json.dumps(state_to_dict(state), separators=(",", ":"), sort_keys=True)


def state_from_json(raw: str | bytes | None) -> AccountState | None:
    """Lenient read path: returns ``None`` rather than raising on junk.

    A corrupt or version-stale Redis value must degrade to a cold start (which is
    observable via ``c3_state_miss_total``), never take down the scorer.
    """
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return state_from_dict(payload)
    except (StateVersionMismatchError, TypeError):
        return None


def redis_state_key(account_id: str, *, prefix: str = "c3") -> str:
    """Versioned key, so a STATE_SCHEMA_VERSION bump is a clean cutover.

    Old keys are simply never read again and expire via TTL -- no migration step.
    """
    return f"{prefix}:acct:v{STATE_SCHEMA_VERSION}:{account_id}"


def features_to_row(features: FeatureVector) -> dict[str, Any]:
    """Flat dict suitable for a JSONL line or a gold.txn_features insert."""
    return features.as_dict()
