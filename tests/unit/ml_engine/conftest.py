"""
Shared test fixtures for ml_engine unit tests.

The async-Redis migration goes through a dual-API transitional phase: some
classes expose both sync and async methods, backed by two Redis clients that
point at the same server. These fixtures wire up that setup.

``fake_server`` is the single fakeredis ``FakeServer`` backing both clients.
``redis_sync`` and ``redis_async`` are the two clients, sharing state via the
server. Tests that only care about one API request just that fixture;
dual-API tests take both.

``redis`` is a back-compat alias for ``redis_sync`` so existing tests keep
passing unchanged. It will be removed once the sync API is deleted (Phase 7).
"""

from __future__ import annotations

import json

import fakeredis
import fakeredis.aioredis
import pytest

from ml_engine.agent.stream_consumer import stream_key


def read_stream_events(redis_sync, run_id: str, event_type: str) -> list:
    """
    Read all events of a given type from the per-run agent stream.

    Uses the sync client so callers don't need async wiring for readback.
    The sync and async clients share the same FakeServer, so writes from
    async code are immediately visible here.
    """
    key = stream_key(run_id)
    entries = redis_sync.xrange(key)
    result = []
    for _, data in entries:
        raw = data.get(b"data", data.get("data", "{}"))
        if isinstance(raw, bytes):
            raw = raw.decode()
        event = json.loads(raw)
        if event.get("type") == event_type:
            result.append(event)
    return result


@pytest.fixture
def fake_server():
    """One FakeServer per test, shared between sync and async clients."""
    return fakeredis.FakeServer()


@pytest.fixture
def redis_sync(fake_server):
    return fakeredis.FakeRedis(server=fake_server, decode_responses=False)


@pytest.fixture
def redis_async(fake_server):
    return fakeredis.aioredis.FakeRedis(server=fake_server, decode_responses=False)
