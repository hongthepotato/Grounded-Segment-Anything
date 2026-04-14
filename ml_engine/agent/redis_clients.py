"""
Async Redis client lifecycle for the agent subsystem.

One async Redis client per `redis_url`, cached process-wide. Callers (Coordinator,
workers, tools, FastAPI route handlers) obtain the shared client via
:func:`get_async_redis_client` and release it at process shutdown via
:func:`close_async_redis_client`.

Why cache by URL:
- `redis.asyncio.Redis` holds its own connection pool. Creating a new client per
  caller means N pools for the same Redis server, which leaks connections on
  pipeline cancellation and can exhaust the server's max-clients limit.
- Matches the existing sync-side pattern of `get_job_manager(redis_url)` in
  `ml_engine/jobs/manager.py`.

Pool sizing: default `max_connections=10`. The agent has at most three concurrent
consumers (AgentLoop, ExecutorWorker, EvaluatorWorker) plus Coordinator tools;
10 is comfortably above working-set. Override only if observed saturation.
"""

from __future__ import annotations

import logging
from typing import Dict

import redis.asyncio as _aredis

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CONNECTIONS = 10

# Process-wide cache: one client per url.
_clients: Dict[str, _aredis.Redis] = {}


def get_async_redis_client(
    redis_url: str,
    max_connections: int = _DEFAULT_MAX_CONNECTIONS,
) -> _aredis.Redis:
    """
    Return the cached async Redis client for ``redis_url``.

    The first call constructs the client with ``max_connections``; subsequent
    calls ignore the pool-size argument and return the existing instance.
    """
    client = _clients.get(redis_url)
    if client is not None:
        return client
    client = _aredis.Redis.from_url(
        redis_url,
        max_connections=max_connections,
        decode_responses=False,
    )
    _clients[redis_url] = client
    logger.debug("Created async Redis client for %s (pool=%d)", redis_url, max_connections)
    return client


async def close_async_redis_client(redis_url: str | None = None) -> None:
    """
    Close the async Redis client(s). If ``redis_url`` is given, close only that
    one; otherwise close every cached client (graceful process shutdown).
    """
    if redis_url is not None:
        client = _clients.pop(redis_url, None)
        if client is not None:
            await client.aclose()
        return
    while _clients:
        url, client = _clients.popitem()
        try:
            await client.aclose()
        except Exception as e:
            logger.warning("Error closing async Redis client for %s: %s", url, e)
