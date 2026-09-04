"""Pathway's Redis output connector: the same monotonic-CAS Lua script BentoML
(Layer 5) will use, so Pathway can restore state Redis lost but never overwrite
state a live request just wrote ("Pathway is a repairer, never an overwriter").
"""

from __future__ import annotations

from typing import Any

import pathway as pw
import redis

from conquer3.config.settings import RedisSettings, StateSettings
from conquer3.core.serde import redis_state_key
from conquer3.redis_scripts import load_script

__all__ = ["RedisStateObserver"]

_CAS_SCRIPT = load_script("monotonic_cas")


class RedisStateObserver(pw.io.python.ConnectorObserver):
    def __init__(self, *, redis_settings: RedisSettings, state_settings: StateSettings) -> None:
        self._client = redis.Redis(
            host=redis_settings.host,
            username=redis_settings.username,
            port=redis_settings.port,
            db=redis_settings.db,
            password=redis_settings.password,
            ssl=redis_settings.tls,
        )
        self._cas = self._client.register_script(_CAS_SCRIPT)
        self._ttl_s = state_settings.ttl_days * 86400
        self._key_prefix = state_settings.key_prefix

    def on_change(self, key: Any, row: dict[str, Any], time: int, is_addition: bool) -> None:
        if not is_addition:
            return  # never delete on retraction -- Pathway is a repairer, not an overwriter
        redis_key = redis_state_key(row["account_id"], prefix=self._key_prefix)
        self._cas(keys=[redis_key], args=[row["state_json"], row["updated_at_us"], self._ttl_s])

    def on_end(self) -> None:
        self._client.close()
