"""
Async event-driven agent loop backed by Redis Streams.

Uses XADD/XREADGROUP so events survive subscriber downtime.
On restart, the loop resumes from the last acknowledged Stream position.

Key decisions:
- max_turns_per_event: caps LLM calls per event (prevents runaway spend)
- State persisted to Redis after every event turn
- Between events: nothing runs, zero cost

Stream-level machinery (XREADGROUP loop, PEL recovery, RedisError retry, always-ACK
poison safety) lives in ``stream_consumer.StreamConsumer``. This module owns the
agent-loop specialization: LoopState persistence and the user-provided ``on_event``
handler wiring.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

import redis.asyncio as _aredis

from ml_engine.agent.state_machine import TERMINAL_STATES, StateMachine
from ml_engine.agent.stream_consumer import (
    StreamConsumer,
    stream_key,
)
from ml_engine.agent.stream_consumer import (
    ensure_consumer_group as _ensure_consumer_group_shared,
)

logger = logging.getLogger(__name__)

# Agent loop state key: agent:{run_id}:loop_state
_STATE_PREFIX = "agent:"
_STATE_SUFFIX = ":loop_state"

# Consumer group name (one per run)
_GROUP_NAME = "coordinator"

MAX_TURNS_PER_EVENT = 5

__all__ = [
    "AgentLoop",
    "LoopState",
    "MAX_TURNS_PER_EVENT",
    "apublish_event",
    "ensure_consumer_group",
    "state_key",
]


@dataclass
class LoopState:
    """Persisted agent loop state. Survives Docker restarts."""

    run_id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    last_event_id: str = "0-0"
    stage_just_completed: Optional[str] = None
    stage_dispatch_overrides: Dict[str, Any] = field(default_factory=dict)
    stage_start_idx: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        r"""Returns a dict of string keys and values suitable for Redis HSET."""
        return {
            "run_id": self.run_id,
            "messages": json.dumps(self.messages),
            "last_event_id": self.last_event_id,
            "stage_just_completed": self.stage_just_completed or "",
            "stage_dispatch_overrides": json.dumps(self.stage_dispatch_overrides),
            "stage_start_idx": "" if self.stage_start_idx is None else str(self.stage_start_idx),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> LoopState:
        r"""Create a LoopState instance from a dictionary of string keys and values."""
        messages_raw = d.get("messages", "[]")
        messages = json.loads(messages_raw) if isinstance(messages_raw, str) else messages_raw
        raw_idx = d.get("stage_start_idx", "")
        return cls(
            run_id=d.get("run_id", ""),
            messages=messages,
            last_event_id=d.get("last_event_id", "0-0"),
            stage_just_completed=d.get("stage_just_completed") or None,
            stage_dispatch_overrides=json.loads(d.get("stage_dispatch_overrides") or "{}"),
            stage_start_idx=int(raw_idx) if raw_idx else None,
        )


def state_key(run_id: str) -> str:
    r"""Returns the Redis key for the LoopState hash for a given run_id."""
    return f"{_STATE_PREFIX}{run_id}{_STATE_SUFFIX}"


async def apublish_event(redis_client: _aredis.Redis, run_id: str, event: Dict[str, Any]) -> str:
    """Publish an event to the pipeline's Redis Stream."""
    key = stream_key(run_id)
    entry_id = await redis_client.xadd(key, {"data": json.dumps(event)})
    logger.debug("Published event to %s: %s", key, event.get("type"))
    return entry_id.decode() if isinstance(entry_id, bytes) else entry_id


async def ensure_consumer_group(redis_client: _aredis.Redis, run_id: str) -> None:
    """
    Create the "coordinator" consumer group for a pipeline run.

    This is the correct entry point for AgentLoop callers. Other workers
    (ExecutorWorker, EvaluatorWorker) use their own CONSUMER_GROUP names and
    call ``stream_consumer.ensure_consumer_group`` directly.
    """
    await _ensure_consumer_group_shared(redis_client, run_id, _GROUP_NAME)


class AgentLoop(StreamConsumer):
    """
    Event-driven loop for one pipeline run.

    Reads events from the per-run Redis Stream, appends each as a user message
    into ``LoopState.messages``, persists BEFORE dispatching so the event trace
    survives handler crashes, then calls the user-provided ``on_event``
    coroutine. State is persisted again after the handler returns to capture
    any mutations.

    Each event triggers at most MAX_TURNS_PER_EVENT LLM calls (enforced by the
    handler, not this loop). Stream-level guarantees (PEL recovery on restart,
    always-ACK on handler exception, RedisError backoff) are inherited from
    :class:`StreamConsumer`.
    """

    CONSUMER_GROUP = _GROUP_NAME
    BATCH_SIZE = 1  # Coordinator processes one event at a time (serial LLM turns)

    def __init__(
        self,
        redis_client: _aredis.Redis,
        run_id: str,
        on_event: Callable[[Dict[str, Any], LoopState], Awaitable[None]],
        consumer_name: str = "coordinator-0",
    ):
        super().__init__(redis_client, run_id, consumer_name)
        self._on_event = on_event
        # Populated in on_start(); this worker is the sole writer, so the
        # in-memory copy is authoritative across iterations.
        self._state: Optional[LoopState] = None

    async def should_stop(self) -> bool:
        """Stop when the pipeline run reaches a terminal state."""
        sm = StateMachine(run_id=self.run_id, redis_async=self._r)
        try:
            if await sm.current_state() in TERMINAL_STATES:
                logger.info(
                    "Run %s reached terminal state, AgentLoop stopping",
                    self.run_id,
                )
                return True
        except KeyError:
            pass  # State not initialized yet -- keep running
        return False

    async def on_start(self) -> None:
        self._state = await self._load_state()

    async def handle_event(self, event: Dict[str, Any], entry_id_str: str) -> None:
        assert self._state is not None, "on_start must run before handle_event"
        state = self._state
        logger.info(
            "Run %s received event: %s (id=%s)",
            self.run_id,
            event.get("type"),
            entry_id_str,
        )

        # Inject event into loop state and persist BEFORE handler runs,
        # so the event trace survives handler crashes. Costs one extra
        # HSET per event; worth it for debuggability.
        state.messages.append(
            {
                "role": "user",
                "content": f"[EVENT] {json.dumps(event)}",
            }
        )
        state.last_event_id = entry_id_str
        await self._save_state(state)

        # Dispatch to handler (Coordinator.on_event)
        await self._on_event(event, state)

        # Persist again to capture handler mutations
        await self._save_state(state)

    async def _load_state(self) -> LoopState:
        key = state_key(self.run_id)
        raw = await self._r.hgetall(key)
        if not raw:
            return LoopState(run_id=self.run_id)
        decoded = {
            k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v
            for k, v in raw.items()
        }
        return LoopState.from_dict(decoded)

    async def _save_state(self, state: LoopState) -> None:
        key = state_key(self.run_id)
        await self._r.hset(key, mapping=state.to_dict())
