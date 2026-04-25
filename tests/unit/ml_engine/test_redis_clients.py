"""
Unit tests for ml_engine.agent.redis_clients.

The module caches one async Redis client per URL. Tests cover:
- Same URL returns the same instance (caching)
- Different URLs get different instances
- max_connections is applied on first construction, ignored thereafter
- close_async_redis_client removes the cached entry and calls aclose()
- close_async_redis_client() with no args closes every cached client
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ml_engine.agent import redis_clients


@pytest.fixture(autouse=True)
def clear_cache():
    """Ensure each test starts with an empty client cache."""
    redis_clients._clients.clear()
    yield
    redis_clients._clients.clear()


# ---------------------------------------------------------------------------
# get_async_redis_client
# ---------------------------------------------------------------------------


class TestGetAsyncRedisClient:
    r""""""

    def test_same_url_returns_cached_instance(self):
        with patch.object(redis_clients._aredis.Redis, "from_url") as mock_from_url:
            mock_from_url.return_value = MagicMock(name="client-1")
            c1 = redis_clients.get_async_redis_client("redis://host-a")
            c2 = redis_clients.get_async_redis_client("redis://host-a")
        assert c1 is c2
        assert mock_from_url.call_count == 1

    def test_different_urls_get_different_instances(self):
        with patch.object(redis_clients._aredis.Redis, "from_url") as mock_from_url:
            mock_from_url.side_effect = [MagicMock(name="a"), MagicMock(name="b")]
            c_a = redis_clients.get_async_redis_client("redis://host-a")
            c_b = redis_clients.get_async_redis_client("redis://host-b")
        assert c_a is not c_b
        assert mock_from_url.call_count == 2

    def test_default_max_connections_applied_on_first_call(self):
        with patch.object(redis_clients._aredis.Redis, "from_url") as mock_from_url:
            mock_from_url.return_value = MagicMock()
            redis_clients.get_async_redis_client("redis://host-a")
        _, kwargs = mock_from_url.call_args
        assert kwargs["max_connections"] == redis_clients._DEFAULT_MAX_CONNECTIONS
        assert kwargs["decode_responses"] is False

    def test_custom_max_connections_applied_on_first_call(self):
        with patch.object(redis_clients._aredis.Redis, "from_url") as mock_from_url:
            mock_from_url.return_value = MagicMock()
            redis_clients.get_async_redis_client("redis://host-a", max_connections=42)
        _, kwargs = mock_from_url.call_args
        assert kwargs["max_connections"] == 42

    def test_subsequent_calls_ignore_max_connections(self):
        """Per the docstring: subsequent calls ignore the pool-size argument."""
        with patch.object(redis_clients._aredis.Redis, "from_url") as mock_from_url:
            first = MagicMock(name="first")
            mock_from_url.return_value = first
            c1 = redis_clients.get_async_redis_client("redis://host-a", max_connections=5)
            c2 = redis_clients.get_async_redis_client("redis://host-a", max_connections=999)
        assert c1 is c2
        assert mock_from_url.call_count == 1


# ---------------------------------------------------------------------------
# close_async_redis_client
# ---------------------------------------------------------------------------


class TestCloseAsyncRedisClient:
    @pytest.mark.asyncio
    async def test_close_specific_url_removes_and_aclose(self):
        client_a = MagicMock()
        client_a.aclose = AsyncMock()
        client_b = MagicMock()
        client_b.aclose = AsyncMock()
        redis_clients._clients["redis://a"] = client_a
        redis_clients._clients["redis://b"] = client_b

        await redis_clients.close_async_redis_client("redis://a")

        client_a.aclose.assert_awaited_once()
        client_b.aclose.assert_not_awaited()
        assert "redis://a" not in redis_clients._clients
        assert "redis://b" in redis_clients._clients

    @pytest.mark.asyncio
    async def test_close_unknown_url_is_noop(self):
        """Closing a URL that was never cached must not raise."""
        await redis_clients.close_async_redis_client("redis://never-cached")

    @pytest.mark.asyncio
    async def test_close_all_closes_every_cached_client(self):
        client_a = MagicMock()
        client_a.aclose = AsyncMock()
        client_b = MagicMock()
        client_b.aclose = AsyncMock()
        redis_clients._clients["redis://a"] = client_a
        redis_clients._clients["redis://b"] = client_b

        await redis_clients.close_async_redis_client()

        client_a.aclose.assert_awaited_once()
        client_b.aclose.assert_awaited_once()
        assert redis_clients._clients == {}

    @pytest.mark.asyncio
    async def test_close_all_swallows_exceptions(self):
        """One failing aclose() must not prevent others from closing."""
        client_ok = MagicMock()
        client_ok.aclose = AsyncMock()
        client_bad = MagicMock()
        client_bad.aclose = AsyncMock(side_effect=RuntimeError("boom"))
        redis_clients._clients["redis://ok"] = client_ok
        redis_clients._clients["redis://bad"] = client_bad

        await redis_clients.close_async_redis_client()

        client_ok.aclose.assert_awaited_once()
        client_bad.aclose.assert_awaited_once()
        assert redis_clients._clients == {}
