"""
MemoryStore -- structured memory backed by plain Redis HASHes.

No RedisJSON required (redis:8.6.1-alpine has no modules).
Python-side filtering for queries.

Four memory types: user, project, feedback, reference.
Keys are prefixed: mem:{type}:{key}
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import redis.asyncio as _aredis

logger = logging.getLogger(__name__)

MEMORY_TYPES = {"user", "project", "feedback", "reference"}


def _validate_type(type_: str) -> None:
    if type_ not in MEMORY_TYPES:
        raise ValueError(f"Unknown memory type: {type_!r}. Must be one of {MEMORY_TYPES}")


def _build_record(type_: str, key: str, content: Dict[str, Any]) -> Dict[str, str]:
    """Build the Redis HASH payload for a memory record."""
    return {
        "type": type_,
        "key": key,
        "content": json.dumps(content),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _parse_record(raw: Dict) -> Dict[str, Any]:
    """Decode a raw Redis HASH into a typed record dict."""
    record = {
        k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v
        for k, v in raw.items()
    }
    content_str = record.get("content", "{}")
    try:
        record["content"] = json.loads(content_str)
    except json.JSONDecodeError:
        record["content"] = {"raw": content_str}
    return record


class MemoryStore:
    """
    Structured memory store using plain Redis HASHes.

    Each record is stored at `mem:{type}:{key}` as a Redis HASH.
    An index at `mem:{type}:_index` is a Redis SET of keys for that type.
    """

    def __init__(self, redis_async: _aredis.Redis):
        # `Any` workaround for redis-py's Awaitable[T] | T overload artifact —
        # see ml_engine/jobs/redis_store.py for the full rationale.
        self._r: Any = redis_async

    async def write(self, type_: str, key: str, content: Dict[str, Any]) -> None:
        """
        Write a memory record.

        Args:
            type_: One of "user", "project", "feedback", "reference".
            key: Unique key within this type (e.g. "teacher_training_lora_r_insight").
            content: Arbitrary dict of memory content.
        """
        _validate_type(type_)
        redis_key = f"mem:{type_}:{key}"
        async with self._r.pipeline() as pipe:
            pipe.hset(redis_key, mapping=_build_record(type_, key, content))
            pipe.sadd(f"mem:{type_}:_index", key)
            await pipe.execute()
        logger.debug("Memory written: %s/%s", type_, key)

    async def read(self, type_: str, key: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Read memory records.

        Args:
            type_: Memory type to read.
            key: If given, return only that record (as a list). If None, return all.

        Returns:
            List of parsed record dicts.
        """
        _validate_type(type_)

        if key is not None:
            raw = await self._r.hgetall(f"mem:{type_}:{key}")
            if not raw:
                return []
            return [_parse_record(raw)]

        keys_raw = await self._r.smembers(f"mem:{type_}:_index")
        results = []
        for k_raw in keys_raw:
            k = k_raw.decode() if isinstance(k_raw, bytes) else k_raw
            raw = await self._r.hgetall(f"mem:{type_}:{k}")
            if raw:
                results.append(_parse_record(raw))
        return results

    async def to_llm_context(self, types: Optional[List[str]] = None) -> str:
        """
        Format selected memory types as a readable string for LLM injection.

        Args:
            types: Which types to include. Defaults to all.
        """
        types = types or list(MEMORY_TYPES)
        lines = []
        for t in types:
            records = await self.read(t)
            if not records:
                continue
            lines.append(f"## Memory: {t}")
            for rec in records:
                content = rec.get("content", {})
                lines.append(f"  [{rec['key']}] {json.dumps(content)}")
        return "\n".join(lines) if lines else "(no memory)"
