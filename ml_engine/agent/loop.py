"""
Event-driven agent loop backed by Redis Streams.

Uses XADD/XREADGROUP (NOT pub/sub) so events survive subscriber downtime.
On restart, the loop resumes from the last acknowledged Stream position.

Key decisions:
- max_turns_per_event: caps LLM calls per event (prevents runaway spend)
- State persisted to Redis after every event turn
- Between events: nothing runs, zero cost
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import redis as _redis

logger = logging.getLogger(__name__)

# Stream key for pipeline-level events: agent:{run_id}:events
_STREAM_PREFIX = "agent:"
_STREAM_SUFFIX = ":events"

# Agent loop state key: agent:{run_id}:loop_state
_STATE_PREFIX = "agent:"
_STATE_SUFFIX = ":loop_state"

# Consumer group name (one per run)
_GROUP_NAME = "coordinator"

MAX_TURNS_PER_EVENT = 5
STREAM_BLOCK_MS = 5000   # 5s block timeout on XREADGROUP


@dataclass
class LoopState:
    """Persisted agent loop state. Survives Docker restarts."""
    run_id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    last_event_id: str = "0-0"
    stage_just_completed: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "messages": json.dumps(self.messages),
            "last_event_id": self.last_event_id,
            "stage_just_completed": self.stage_just_completed or "",
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LoopState":
        messages_raw = d.get("messages", "[]")
        messages = json.loads(messages_raw) if isinstance(messages_raw, str) else messages_raw
        return cls(
            run_id=d.get("run_id", ""),
            messages=messages,
            last_event_id=d.get("last_event_id", "0-0"),
            stage_just_completed=d.get("stage_just_completed") or None,
        )


def stream_key(run_id: str) -> str:
    return f"{_STREAM_PREFIX}{run_id}{_STREAM_SUFFIX}"


def state_key(run_id: str) -> str:
    return f"{_STATE_PREFIX}{run_id}{_STATE_SUFFIX}"


def publish_event(redis_client: _redis.Redis, run_id: str, event: Dict[str, Any]) -> str:
    """
    Publish an event to the pipeline's Redis Stream.

    Returns the Stream entry ID (e.g. "1234567890123-0").
    """
    key = stream_key(run_id)
    entry_id = redis_client.xadd(key, {"data": json.dumps(event)})
    logger.debug("Published event to %s: %s", key, event.get("type"))
    return entry_id.decode() if isinstance(entry_id, bytes) else entry_id


def ensure_consumer_group(redis_client: _redis.Redis, run_id: str) -> None:
    """Create the consumer group if it doesn't exist. Idempotent."""
    key = stream_key(run_id)
    try:
        redis_client.xgroup_create(key, _GROUP_NAME, id="0", mkstream=True)
        logger.debug("Created consumer group %s on %s", _GROUP_NAME, key)
    except _redis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


class AgentLoop:
    """
    Event-driven loop for one pipeline run.

    Call run() once; it blocks reading from the Redis Stream until:
    - A terminal state is reached (done, failed_unrecoverable, escalated)
    - cancel_check() returns True
    - A fatal exception occurs

    Each event triggers at most MAX_TURNS_PER_EVENT LLM calls. State is
    persisted after every turn. The loop auto-resumes from the last ACKed
    position on restart.
    """

    def __init__(
        self,
        redis_client: _redis.Redis,
        run_id: str,
        on_event: Callable[[Dict[str, Any], LoopState], asyncio.Coroutine],
        consumer_name: str = "coordinator-0",
    ):
        self._r = redis_client
        self.run_id = run_id
        self._on_event = on_event
        self._consumer_name = consumer_name

    async def run(
        self,
        cancel_check: Optional[Callable[[], bool]] = None,
        max_events: Optional[int] = None,
    ) -> None:
        """Block on the Stream and dispatch each event to on_event."""
        ensure_consumer_group(self._r, self.run_id)
        key = stream_key(self.run_id)
        events_processed = 0

        logger.info("AgentLoop started for run %s", self.run_id)

        while True:
            if cancel_check and cancel_check():
                logger.info("AgentLoop cancelled for run %s", self.run_id)
                break

            if max_events is not None and events_processed >= max_events:
                break

            # Load or initialize loop state
            state = self._load_state()

            # XREADGROUP: blocks up to STREAM_BLOCK_MS, then retries
            try:
                entries = self._r.xreadgroup(
                    groupname=_GROUP_NAME,
                    consumername=self._consumer_name,
                    streams={key: ">"}, # '>' means: give me only new messages I haven't ACKed yet
                    count=1,
                    block=STREAM_BLOCK_MS, # block for STREAM_BLOCK_MS milliseconds if no messages, then return empty to loop again
                )
            except _redis.RedisError as e:
                logger.error("Redis error reading stream: %s", e)
                await asyncio.sleep(1)
                continue

            if not entries:
                continue  # timeout, loop again

            for _stream_key, messages in entries:
                for entry_id, entry_data in messages:
                    entry_id_str = entry_id.decode() if isinstance(entry_id, bytes) else entry_id
                    data_raw = entry_data.get(b"data", entry_data.get("data", "{}"))
                    if isinstance(data_raw, bytes):
                        data_raw = data_raw.decode()
                    event = json.loads(data_raw)

                    logger.info(
                        "Run %s received event: %s (id=%s)",
                        self.run_id, event.get("type"), entry_id_str,
                    )

                    try:
                        # Inject event into loop state
                        state.messages.append({
                            "role": "user",
                            "content": f"[EVENT] {json.dumps(event)}",
                        })
                        state.last_event_id = entry_id_str

                        # Dispatch to handler (Coordinator.on_event)
                        await self._on_event(event, state)

                        # Persist state
                        self._save_state(state)

                        # ACK the message
                        self._r.xack(key, _GROUP_NAME, entry_id_str)
                        events_processed += 1

                    except Exception as e:
                        logger.error(
                            "Error handling event %s for run %s: %s",
                            entry_id_str, self.run_id, e, exc_info=True,
                        )
                        # Still ACK to avoid infinite retry loop on poison messages
                        self._r.xack(key, _GROUP_NAME, entry_id_str)

    def _load_state(self) -> LoopState:
        key = state_key(self.run_id)
        raw = self._r.hgetall(key)
        if not raw:
            return LoopState(run_id=self.run_id)
        decoded = {
            k.decode() if isinstance(k, bytes) else k:
            v.decode() if isinstance(v, bytes) else v
            for k, v in raw.items()
        }
        return LoopState.from_dict(decoded)

    def _save_state(self, state: LoopState) -> None:
        key = state_key(self.run_id)
        self._r.hset(key, mapping=state.to_dict())
