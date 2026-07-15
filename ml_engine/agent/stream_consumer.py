"""
Shared async Redis Streams consumer-group machinery.

Before this module existed, AgentLoop, ExecutorWorker, and EvaluatorWorker each
open-coded the same pattern: ensure_consumer_group, while-loop with cancel_check,
xreadgroup with block timeout, RedisError backoff, decode+dispatch+ACK. Any fix
(PEL recovery, async Redis, backoff strategy) had to land in three places. This
base class consolidates that skeleton so subclasses only provide the event
handler and (optionally) a terminal-state stop check.

Cancellation semantics:
  A cancelled handler (asyncio.CancelledError) is NOT poison. It means the
  caller pulled the plug mid-work. The message is deliberately left unacked so
  PEL recovery on next start replays it. Any other exception is treated as
  poison and always ACKed.

This module also owns the low-level stream helpers (``stream_key``,
``ensure_consumer_group``, ``STREAM_BLOCK_MS``) since they are used by every
consumer. loop.py re-exports them for backward-compatible imports.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Optional

import redis  # exception classes live at redis.RedisError / redis.ResponseError
import redis.asyncio as _aredis

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stream naming + setup
# ---------------------------------------------------------------------------

# Per-run stream key: agent:{run_id}:events
_STREAM_PREFIX = "agent:"
_STREAM_SUFFIX = ":events"

# 5s block timeout on XREADGROUP. Balances responsiveness to cancel_check
# against Redis QPS when idle.
STREAM_BLOCK_MS = 5000


def stream_key(run_id: str) -> str:
    """Return the per-run Stream key used by every agent worker."""
    return f"{_STREAM_PREFIX}{run_id}{_STREAM_SUFFIX}"


async def ensure_consumer_group(
    redis_client: _aredis.Redis,
    run_id: str,
    group_name: str = "coordinator",
) -> None:
    """
    Create a consumer group on the run's Stream if it doesn't exist. Idempotent.

    ``group_name`` defaults to ``"coordinator"`` for backward compatibility with
    earlier callers (AgentLoop / workers.py) that used a bare two-arg form.
    New call sites should pass an explicit group name matching their subclass
    ``CONSUMER_GROUP``.
    """
    key = stream_key(run_id)
    try:
        await redis_client.xgroup_create(key, group_name, id="0", mkstream=True)
        logger.debug("Created consumer group %s on %s", group_name, key)
    except redis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class StreamConsumer(ABC):
    """
    Base class for consumers of the per-run agent Redis Stream.

    Subclasses must set ``CONSUMER_GROUP`` and implement ``handle_event``. They
    may override ``should_stop`` (custom stop conditions), ``on_start`` (one-shot
    setup), and ``on_event_error`` (recover from handler exceptions, e.g.
    publishing an escalation event).

    Core guarantees:
      * Every delivered message is ACKed on successful dispatch OR on a
        non-cancellation handler exception (poison-message safety).
      * A cancelled handler leaves the message unacked; PEL recovery replays it.
      * On startup, pending-but-unACKed messages for this ``consumer_name`` are
        reclaimed via XPENDING + XCLAIM before new-message processing begins.
      * Transient RedisError on xreadgroup triggers a 1s backoff + retry, not
        a crash.
    """

    CONSUMER_GROUP: ClassVar[str] = ""
    BATCH_SIZE: ClassVar[int] = 5

    def __init__(
        self,
        redis_client: _aredis.Redis,
        run_id: str,
        consumer_name: str,
    ):
        assert self.CONSUMER_GROUP, f"{type(self).__name__} must set CONSUMER_GROUP"
        # `Any` workaround for redis-py's Awaitable[T] | T overload artifact
        # — see ml_engine/jobs/redis_store.py for the full rationale.
        # Subclasses (AgentLoop, etc.) inherit this typing for `self._r`.
        self._r: Any = redis_client
        self.run_id = run_id
        self._consumer_name = consumer_name

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    async def run(
        self,
        cancel_check: Optional[Callable[[], bool]] = None,
        max_events: Optional[int] = None,
    ) -> None:
        """Block on the Stream and dispatch each event to ``handle_event``."""
        await ensure_consumer_group(self._r, self.run_id, self.CONSUMER_GROUP)
        key = stream_key(self.run_id)  # agent:{run_id}:events
        events_processed = 0

        await self.on_start()  # subclass hook for one-shot setup before the main loop

        logger.info(
            "%s started for run %s (consumer=%s)",
            type(self).__name__,
            self.run_id,
            self._consumer_name,
        )

        events_processed = await self._drain_pel(key, events_processed, cancel_check, max_events)

        while True:
            if cancel_check and cancel_check():
                logger.info("%s cancelled for run %s", type(self).__name__, self.run_id)
                break
            if max_events is not None and events_processed >= max_events:
                break
            if await self.should_stop():
                logger.info(
                    "%s stop condition met for run %s",
                    type(self).__name__,
                    self.run_id,
                )
                break

            try:
                entries = await self._r.xreadgroup(
                    groupname=self.CONSUMER_GROUP,
                    consumername=self._consumer_name,
                    streams={key: ">"},  # only new messages
                    count=self.BATCH_SIZE,
                    block=STREAM_BLOCK_MS,
                )
            except redis.RedisError as e:
                logger.error("%s Redis error: %s", type(self).__name__, e)
                await asyncio.sleep(1)
                continue

            if not entries:
                continue

            events_processed = await self._process_entries(entries, key, events_processed)

    # ------------------------------------------------------------------
    # PEL reclamation
    # ------------------------------------------------------------------

    async def _drain_pel(
        self,
        key: str,
        events_processed: int,
        cancel_check: Optional[Callable[[], bool]],
        max_events: Optional[int],
    ) -> int:
        """
        Reclaim messages delivered to this consumer but never ACKed.

        Uses XPENDING + XCLAIM rather than ``XREADGROUP streams={key: "0"}``
        because the latter has inconsistent PEL-replay behavior across Redis
        implementations (notably fakeredis 2.x). XCLAIM with min_idle_time=0
        is safe here because this method runs only at startup, before any
        other workers under the same consumer_name can be active.
        """
        while True:
            if cancel_check and cancel_check():
                return events_processed
            if max_events is not None and events_processed >= max_events:
                return events_processed

            try:
                pending = await self._r.xpending_range(
                    key,
                    self.CONSUMER_GROUP,
                    min="-",
                    max="+",
                    count=self.BATCH_SIZE * 2,
                    consumername=self._consumer_name,
                )
            except redis.RedisError as e:
                logger.error(
                    "%s Redis error reading PEL: %s",
                    type(self).__name__,
                    e,
                )
                return events_processed

            if not pending:
                return events_processed

            ids = [p["message_id"] for p in pending]

            try:
                claimed = await self._r.xclaim(
                    key,
                    self.CONSUMER_GROUP,
                    self._consumer_name,
                    min_idle_time=0,
                    message_ids=ids,
                )
            except redis.RedisError as e:
                logger.error(
                    "%s Redis error claiming PEL: %s",
                    type(self).__name__,
                    e,
                )
                return events_processed

            if not claimed:
                return events_processed

            logger.info(
                "%s reclaiming %d PEL entries for consumer %s",
                type(self).__name__,
                len(claimed),
                self._consumer_name,
            )
            # xclaim returns [(id, data), ...] -- wrap to match xreadgroup shape
            wrapped = [(key, claimed)]
            events_processed = await self._process_entries(wrapped, key, events_processed)

    # ------------------------------------------------------------------
    # Entry processing
    # ------------------------------------------------------------------

    async def _process_entries(
        self,
        entries: Any,
        key: str,
        events_processed: int,
    ) -> int:
        """Decode each message, dispatch to handle_event, ACK. Returns counter."""
        for _, messages in entries:
            for entry_id, entry_data in messages:
                entry_id_str = entry_id.decode() if isinstance(entry_id, bytes) else entry_id
                raw = entry_data.get(b"data", entry_data.get("data", "{}"))
                if isinstance(raw, bytes):
                    raw = raw.decode()
                event = json.loads(raw)

                await self._dispatch(event, entry_id_str, key)
                events_processed += 1
        return events_processed

    async def _dispatch(
        self,
        event: Dict[str, Any],
        entry_id_str: str,
        key: str,
    ) -> None:
        """
        Invoke handle_event with poison-safety: ACK on success OR on a
        non-cancellation exception. Cancellation leaves the message unacked so
        the PEL replays it.
        """
        try:
            await self.handle_event(event, entry_id_str)
        except asyncio.CancelledError:
            # Do not ACK, do not swallow -- PEL recovery picks this up later.
            raise
        except Exception as e:
            logger.error(
                "%s handler error for event %s: %s",
                type(self).__name__,
                entry_id_str,
                e,
                exc_info=True,
            )
            try:
                await self.on_event_error(event, entry_id_str, e)
            except asyncio.CancelledError:
                raise
            except Exception as inner:
                logger.error(
                    "%s on_event_error raised: %s",
                    type(self).__name__,
                    inner,
                    exc_info=True,
                )
        # Handler succeeded or raised non-cancellation -> ACK.
        await self._r.xack(key, self.CONSUMER_GROUP, entry_id_str)

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    async def on_start(self) -> None:
        """One-shot setup before the main loop. Override as needed."""
        return None

    async def should_stop(self) -> bool:
        """Custom stop condition (e.g., terminal state). Default: never stop."""
        return False

    @abstractmethod
    async def handle_event(self, event: Dict[str, Any], entry_id_str: str) -> None:
        """Subclass event dispatch. Event has been JSON-decoded."""
        raise NotImplementedError

    async def on_event_error(self, event: Dict[str, Any], entry_id_str: str, exc: BaseException) -> None:
        """Hook called when handle_event raises. Default: no-op (error is logged)."""
        return None
