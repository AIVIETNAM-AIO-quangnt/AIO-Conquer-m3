"""RedisStateStore against a real ephemeral Redis (testcontainers, same
philosophy as Layer 3b's gate) -- get/commit round-tripping, the monotonic-CAS
rejection path, and the hit/miss/cas_rejected counters. The CAS *script* itself
(monotonic_cas.lua) is already proven correct by Pathway's tests; this file
proves RedisStateStore's own glue around it (get, commit, metrics) is correct.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import pytest

pytest.importorskip("testcontainers")

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from testcontainers.community.redis import RedisContainer

from conquer3.config.settings import RedisSettings, StateSettings
from conquer3.core.serde import redis_state_key
from conquer3.core.types import AccountState
from conquer3.serving.state_store import RedisStateStore

pytestmark = [pytest.mark.integration]


@pytest.fixture(scope="module")
def metric_reader() -> InMemoryMetricReader:
    """The OTel MeterProvider is a process-wide singleton that only accepts
    being set once (later calls are silently ignored with a warning) -- so this
    is set up ONCE for the module and every test reads *deltas* off it, rather
    than each test racing to install its own reader."""
    reader = InMemoryMetricReader()
    metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))
    return reader


@pytest.fixture
def store(metric_reader: InMemoryMetricReader) -> Iterator[RedisStateStore]:
    with RedisContainer("redis:7-alpine") as redis_c:
        redis_settings = RedisSettings(
            host=redis_c.get_container_host_ip(), port=int(redis_c.get_exposed_port(6379))
        )
        state_settings = StateSettings(key_prefix="c3test", ttl_days=1)
        rss = RedisStateStore(redis_settings=redis_settings, state_settings=state_settings)
        try:
            yield rss
        finally:
            rss.close()


def _counter_value(reader: InMemoryMetricReader, name: str) -> float:
    data = reader.get_metrics_data()
    if data is None:
        return 0.0
    total = 0.0
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                if m.name == name:
                    total += sum(dp.value for dp in m.data.data_points)
    return total


def _state(account_id: str, *, updated_at_us: int, last_event_id: str = "e1") -> AccountState:
    return AccountState(
        account_id=account_id,
        last_event_id=last_event_id,
        last_event_ts_us=updated_at_us,
        updated_at_us=updated_at_us,
        txn_count=1,
        amount_sum=100.0,
    )


def test_get_on_a_missing_key_returns_none_and_increments_miss(
    store: RedisStateStore, metric_reader: InMemoryMetricReader
) -> None:
    before = _counter_value(metric_reader, "c3_state_miss_total")
    assert store.get("no-such-account") is None
    assert _counter_value(metric_reader, "c3_state_miss_total") == before + 1


def test_commit_then_get_round_trips(
    store: RedisStateStore, metric_reader: InMemoryMetricReader
) -> None:
    before = _counter_value(metric_reader, "c3_state_hit_total")
    state = _state("A", updated_at_us=1000)
    assert store.commit(state) is True

    fetched = store.get("A")
    assert fetched is not None
    assert fetched.account_id == "A"
    assert fetched.last_event_id == "e1"
    assert fetched.txn_count == 1
    assert fetched.amount_sum == pytest.approx(100.0)
    assert _counter_value(metric_reader, "c3_state_hit_total") == before + 1


def test_commit_rejects_a_stale_write_and_accepts_a_fresher_one(
    store: RedisStateStore, metric_reader: InMemoryMetricReader
) -> None:
    before = _counter_value(metric_reader, "c3_state_cas_rejected_total")

    fresh = _state("A", updated_at_us=2000, last_event_id="fresh")
    assert store.commit(fresh) is True

    stale = _state("A", updated_at_us=1000, last_event_id="stale")
    assert store.commit(stale) is False
    assert _counter_value(metric_reader, "c3_state_cas_rejected_total") == before + 1

    # The rejected write must not have overwritten the fresher one.
    fetched = store.get("A")
    assert fetched is not None
    assert fetched.last_event_id == "fresh"

    fresher = replace(fresh, updated_at_us=3000, last_event_id="fresher")
    assert store.commit(fresher) is True
    fetched = store.get("A")
    assert fetched is not None
    assert fetched.last_event_id == "fresher"
    assert _counter_value(metric_reader, "c3_state_cas_rejected_total") == before + 1  # unchanged


def test_ttl_is_set_on_commit(store: RedisStateStore) -> None:
    store.commit(_state("A", updated_at_us=1000))
    ttl = store._client.ttl(redis_state_key("A", prefix="c3test"))
    assert 0 < ttl <= 86400
