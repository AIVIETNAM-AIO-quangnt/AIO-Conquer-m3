"""Redis-backed account state for the online path.

Writes through the exact same monotonic-CAS Lua script
``pipelines/pathway/sinks/redis_sink.py`` uses -- imported from the same
``conquer3.redis_scripts`` bundle, not re-authored here -- so Pathway (a repairer)
and the live scorer (the primary writer) can never drift onto different CAS
semantics. A rejected write (a fresher `updated_at_us` already stored) is an
expected outcome under concurrency, not an error: same-account races are real
*within* one uvicorn worker process now that ``/invocations`` runs ``predict`` on
parallel threads (plan §8.4), and the CAS script is what makes that safe.
"""

from __future__ import annotations

import redis

from conquer3.config.settings import RedisSettings, StateSettings
from conquer3.core.serde import redis_state_key, state_from_json, state_to_json
from conquer3.core.types import AccountState
from conquer3.redis_scripts import load_script
from conquer3.telemetry.otel import get_meter

__all__ = ["RedisStateStore"]

_CAS_SCRIPT = load_script("monotonic_cas")


class RedisStateStore:
    """Thread-safe by construction: ``redis.Redis`` pools its own connections
    internally and a registered ``Script`` is a stateless callable, so concurrent
    request threads inside one worker process may share a single instance without
    a lock (see plan §8.4 -- ``FraudScorer`` holds no per-request mutable
    state of its own either)."""

    def __init__(self, *, redis_settings: RedisSettings, state_settings: StateSettings) -> None:
        self._client = redis.Redis(
            host=redis_settings.host,
            username=redis_settings.username,
            port=redis_settings.port,
            db=redis_settings.db,
            password=redis_settings.password,
            ssl=redis_settings.tls,
            decode_responses=True,
        )
        self._cas = self._client.register_script(_CAS_SCRIPT)
        self._ttl_s = state_settings.ttl_days * 86400
        self._key_prefix = state_settings.key_prefix

        meter = get_meter(__name__)
        self._hit_counter = meter.create_counter(
            "c3_state_hit_total", description="Redis reads that found existing account state"
        )
        self._miss_counter = meter.create_counter(
            "c3_state_miss_total", description="Redis reads with no existing account state"
        )
        self._cas_rejected_counter = meter.create_counter(
            "c3_state_cas_rejected_total",
            description="Monotonic-CAS writes rejected by a fresher concurrent write",
        )

    def get(self, account_id: str) -> AccountState | None:
        """Read-only; safe to call even during a dry_run (dry_run only skips the
        *write* side -- see FraudScorer.score)."""
        raw = self._client.get(redis_state_key(account_id, prefix=self._key_prefix))
        state = state_from_json(raw)
        (self._hit_counter if state is not None else self._miss_counter).add(1)
        return state

    def commit(self, state: AccountState) -> bool:
        """Monotonic-CAS write. Returns False if a fresher write already landed --
        never raises; a lost race is expected, not exceptional."""
        key = redis_state_key(state.account_id, prefix=self._key_prefix)
        accepted = bool(
            self._cas(keys=[key], args=[state_to_json(state), state.updated_at_us, self._ttl_s])
        )
        if not accepted:
            self._cas_rejected_counter.add(1)
        return accepted

    def close(self) -> None:
        self._client.close()
