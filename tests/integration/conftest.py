"""
Shared fixtures for integration tests.

All Redis-backed fixtures use fakeredis with a shared FakeServer so the async
and (if needed) sync clients see the same in-memory state within a single test.
"""

from __future__ import annotations

import uuid

import fakeredis
import fakeredis.aioredis
import pytest
import pytest_asyncio


@pytest.fixture
def fake_server() -> fakeredis.FakeServer:
    """One FakeServer per test -- state never bleeds across tests."""
    return fakeredis.FakeServer()


@pytest_asyncio.fixture
async def redis_async(fake_server: fakeredis.FakeServer) -> fakeredis.aioredis.FakeRedis:
    """Async fakeredis client backed by the shared FakeServer."""
    client = fakeredis.aioredis.FakeRedis(server=fake_server, decode_responses=False)
    yield client
    await client.aclose()


@pytest.fixture
def redis_sync(fake_server: fakeredis.FakeServer) -> fakeredis.FakeRedis:
    """Sync fakeredis client backed by the same FakeServer (for dequeue assertions)."""
    client = fakeredis.FakeRedis(server=fake_server, decode_responses=False)
    yield client
    client.close()


@pytest.fixture
def run_id() -> str:
    """Unique run_id per test to avoid cross-test key collisions."""
    return f"test-run-{uuid.uuid4().hex[:8]}"
