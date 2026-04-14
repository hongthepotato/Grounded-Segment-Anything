"""
Unit tests for ml_engine.agent.stream_consumer.

Tests cover the shared StreamConsumer base class behaviors that are NOT
already exercised through AgentLoop (test_agent_loop.py) or EvaluatorWorker
(test_evaluator_worker.py):

- stream_key / ensure_consumer_group helpers
- CONSUMER_GROUP assertion on instantiation
- RECLAIM_PEL_ON_START=False skips PEL drain
- on_start hook called before main loop
- should_stop() returning True exits immediately
- on_event_error default (no-op) + message still ACKed
- on_event_error raising CancelledError re-raised (message NOT ACKed)
- on_event_error raising regular Exception is swallowed + message ACKed
- _drain_pel resilience: RedisError on xpending_range / xclaim returns gracefully
- _drain_pel breaks out when xclaim returns empty list
- _drain_pel respects cancel_check and max_events
- Multiple events in one batch -- events_processed counter increments correctly
- cancel_check stops the main loop

Uses a minimal ``RecordingConsumer`` concrete subclass so we can exercise
the base-class machinery without pulling in Coordinator or AgentLoop logic.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import pytest
import redis as _redis

from ml_engine.agent.stream_consumer import (
    StreamConsumer,
    ensure_consumer_group,
    stream_key,
)
from ml_engine.agent.loop import apublish_event


# ---------------------------------------------------------------------------
# Fixtures (redis_async from conftest)
# ---------------------------------------------------------------------------


@pytest.fixture
def run_id():
    return "sc-test-run-001"


# ---------------------------------------------------------------------------
# Concrete subclass for testing
# ---------------------------------------------------------------------------


class RecordingConsumer(StreamConsumer):
    """Minimal StreamConsumer that records every event it handles."""

    CONSUMER_GROUP = "test-group"

    def __init__(self, redis_client, run_id, *, stop_after: int = 0, **kwargs):
        super().__init__(redis_client, run_id, consumer_name="test-0", **kwargs)
        self.received: List[Dict[str, Any]] = []
        self._stop_after = stop_after   # should_stop() returns True after N events
        self._started = False

    async def on_start(self) -> None:
        self._started = True

    async def should_stop(self) -> bool:
        if self._stop_after and len(self.received) >= self._stop_after:
            return True
        return False

    async def handle_event(self, event: Dict[str, Any], entry_id_str: str) -> None:
        self.received.append(event)


# ---------------------------------------------------------------------------
# stream_key / ensure_consumer_group
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_stream_key_format(self):
        assert stream_key("abc123") == "agent:abc123:events"

    @pytest.mark.asyncio
    async def test_ensure_consumer_group_creates_group(self, redis_async, run_id):
        await ensure_consumer_group(redis_async, run_id, group_name="my-group")
        key = stream_key(run_id)
        groups = await redis_async.xinfo_groups(key)
        names = [g["name"] if isinstance(g["name"], str) else g["name"].decode() for g in groups]
        assert "my-group" in names

    @pytest.mark.asyncio
    async def test_ensure_consumer_group_idempotent(self, redis_async, run_id):
        """Calling twice must not raise (BUSYGROUP is silently swallowed)."""
        await ensure_consumer_group(redis_async, run_id, group_name="dup-group")
        await ensure_consumer_group(redis_async, run_id, group_name="dup-group")

    @pytest.mark.asyncio
    async def test_ensure_consumer_group_custom_name(self, redis_async, run_id):
        """Non-default group name is created, not "coordinator"."""
        await ensure_consumer_group(redis_async, run_id, group_name="executor")
        key = stream_key(run_id)
        groups = await redis_async.xinfo_groups(key)
        names = [g["name"] if isinstance(g["name"], str) else g["name"].decode() for g in groups]
        assert "executor" in names
        assert "coordinator" not in names


# ---------------------------------------------------------------------------
# Instantiation guard
# ---------------------------------------------------------------------------


class TestInstantiationGuard:
    def test_missing_consumer_group_raises(self, redis_async, run_id):
        """Subclass with empty CONSUMER_GROUP must raise AssertionError."""

        class BadConsumer(StreamConsumer):
            CONSUMER_GROUP = ""

            async def handle_event(self, event, entry_id_str):
                pass

        with pytest.raises(AssertionError):
            BadConsumer(redis_async, run_id, consumer_name="x")


# ---------------------------------------------------------------------------
# on_start hook
# ---------------------------------------------------------------------------


class TestOnStartHook:
    @pytest.mark.asyncio
    async def test_on_start_called_before_first_event(self, redis_async, run_id):
        """on_start must run before handle_event is ever called."""
        await apublish_event(redis_async, run_id, {"type": "ping"})
        consumer = RecordingConsumer(redis_async, run_id)
        assert not consumer._started
        await consumer.run(max_events=1)
        assert consumer._started
        assert consumer.received


# ---------------------------------------------------------------------------
# should_stop
# ---------------------------------------------------------------------------


class TestShouldStop:
    @pytest.mark.asyncio
    async def test_should_stop_exits_without_processing(self, redis_async, run_id):
        """If should_stop() returns True before the first read, loop exits cleanly."""
        await apublish_event(redis_async, run_id, {"type": "ignored"})

        class ImmediateStop(RecordingConsumer):
            async def should_stop(self):
                return True

        consumer = ImmediateStop(redis_async, run_id)
        await consumer.run()
        assert consumer.received == []

    @pytest.mark.asyncio
    async def test_should_stop_after_n_events(self, redis_async, run_id):
        """
        should_stop() fires between batches. With BATCH_SIZE=1 each event is
        its own batch, so the loop stops after exactly N events.
        """
        for i in range(5):
            await apublish_event(redis_async, run_id, {"type": "e", "i": i})

        class OneBatchConsumer(RecordingConsumer):
            BATCH_SIZE = 1

        consumer = OneBatchConsumer(redis_async, run_id, stop_after=2)
        await consumer.run()
        assert len(consumer.received) == 2


# ---------------------------------------------------------------------------
# RECLAIM_PEL_ON_START
# ---------------------------------------------------------------------------


class TestNoPELOnStart:
    @pytest.mark.asyncio
    async def test_reclaim_false_skips_pel_drain(self, redis_async, run_id):
        """
        With RECLAIM_PEL_ON_START=False, a message stuck in the PEL is NOT
        re-delivered on restart -- the consumer skips straight to new (">") messages.
        """
        class NoPELConsumer(RecordingConsumer):
            RECLAIM_PEL_ON_START = False

        run = run_id + "-nopel"

        # Publish and consume-without-ACK to put a message in the PEL.
        await ensure_consumer_group(redis_async, run, group_name="test-group")
        await apublish_event(redis_async, run, {"type": "stuck_in_pel"})
        key = stream_key(run)
        await redis_async.xreadgroup(
            groupname="test-group",
            consumername="test-0",
            streams={key: ">"},
            count=1,
        )
        # Do NOT ack -- message now in PEL.

        consumer = NoPELConsumer(redis_async, run)
        # max_events=0 with no new messages: loop exits immediately after
        # cancel_check or should_stop; nothing is re-delivered.
        await consumer.run(max_events=0)

        assert consumer.received == [], "PEL message must NOT be replayed when RECLAIM_PEL_ON_START=False"


# ---------------------------------------------------------------------------
# cancel_check
# ---------------------------------------------------------------------------


class TestCancelCheck:
    @pytest.mark.asyncio
    async def test_cancel_check_stops_loop(self, redis_async, run_id):
        run = run_id + "-cc"
        await apublish_event(redis_async, run, {"type": "after_cancel"})

        consumer = RecordingConsumer(redis_async, run)
        await consumer.run(cancel_check=lambda: True)  # already cancelled at first check
        assert consumer.received == []


# ---------------------------------------------------------------------------
# on_event_error hooks
# ---------------------------------------------------------------------------


class TestOnEventError:
    @pytest.mark.asyncio
    async def test_default_on_event_error_acks_message(self, redis_async, run_id):
        """
        When handle_event raises a regular exception, the default on_event_error
        is a no-op; the message is still ACKed (poison safety).
        """
        run = run_id + "-oee-default"
        await apublish_event(redis_async, run, {"type": "boom"})

        class ExplodingConsumer(RecordingConsumer):
            async def handle_event(self, event, entry_id_str):
                raise RuntimeError("boom")

        consumer = ExplodingConsumer(redis_async, run)
        await consumer.run(max_events=1)

        key = stream_key(run)
        pending = await redis_async.xpending_range(
            key, "test-group", min="-", max="+", count=10,
            consumername="test-0",
        )
        assert pending == [], "message must be ACKed even when handler raises"

    @pytest.mark.asyncio
    async def test_on_event_error_raising_cancelled_error_reraises(self, redis_async, run_id):
        """
        If on_event_error itself raises CancelledError, it propagates out so
        the caller (asyncio task) can observe the cancellation. The message is
        NOT ACKed.
        """
        run = run_id + "-oee-cancel"
        await apublish_event(redis_async, run, {"type": "cancel_in_error_hook"})

        class CancelInErrorConsumer(RecordingConsumer):
            async def handle_event(self, event, entry_id_str):
                raise RuntimeError("trigger error hook")

            async def on_event_error(self, event, entry_id_str, exc):
                raise asyncio.CancelledError()

        consumer = CancelInErrorConsumer(redis_async, run)
        with pytest.raises(asyncio.CancelledError):
            await consumer.run(max_events=1)

        # Message must remain in PEL (not ACKed).
        key = stream_key(run)
        pending = await redis_async.xpending_range(
            key, "test-group", min="-", max="+", count=10,
            consumername="test-0",
        )
        assert len(pending) == 1, "message must remain unacked when on_event_error raises CancelledError"

    @pytest.mark.asyncio
    async def test_on_event_error_raising_regular_exception_is_swallowed(self, redis_async, run_id):
        """
        If on_event_error raises a regular exception, it is logged and swallowed.
        The message is still ACKed so we don't loop forever on a poison message.
        """
        run = run_id + "-oee-swallow"
        await apublish_event(redis_async, run, {"type": "double_trouble"})

        class DoubleFailConsumer(RecordingConsumer):
            async def handle_event(self, event, entry_id_str):
                raise RuntimeError("first failure")

            async def on_event_error(self, event, entry_id_str, exc):
                raise ValueError("error in error handler")

        consumer = DoubleFailConsumer(redis_async, run)
        # Must not raise, must not hang.
        await consumer.run(max_events=1)

        key = stream_key(run)
        pending = await redis_async.xpending_range(
            key, "test-group", min="-", max="+", count=10,
            consumername="test-0",
        )
        assert pending == [], "message must be ACKed even when on_event_error raises"


# ---------------------------------------------------------------------------
# _drain_pel resilience
# ---------------------------------------------------------------------------


class TestDrainPELResilience:
    @pytest.mark.asyncio
    async def test_redis_error_on_xpending_returns_gracefully(self, redis_async, run_id):
        """RedisError during xpending_range must not crash -- PEL drain bails out."""
        run = run_id + "-pel-xpend-err"
        await apublish_event(redis_async, run, {"type": "new_event"})

        consumer = RecordingConsumer(redis_async, run)

        with patch.object(
            redis_async, "xpending_range",
            new=AsyncMock(side_effect=_redis.RedisError("xpending boom")),
        ):
            await consumer.run(max_events=1)

        # Loop continues to ">" and picks up the new message.
        assert len(consumer.received) == 1
        assert consumer.received[0]["type"] == "new_event"

    @pytest.mark.asyncio
    async def test_redis_error_on_xclaim_returns_gracefully(self, redis_async, run_id):
        """RedisError during xclaim must not crash -- PEL drain bails out early."""
        run = run_id + "-pel-xclaim-err"

        # Create a real PEL entry to trigger the xclaim path.
        await ensure_consumer_group(redis_async, run, group_name="test-group")
        await apublish_event(redis_async, run, {"type": "stuck"})
        key = stream_key(run)
        await redis_async.xreadgroup(
            groupname="test-group", consumername="test-0",
            streams={key: ">"}, count=1,
        )
        # PEL has 1 entry. Now publish a new message too.
        await apublish_event(redis_async, run, {"type": "new_after_stuck"})

        consumer = RecordingConsumer(redis_async, run)

        real_xclaim = redis_async.xclaim

        async def failing_xclaim(*args, **kwargs):
            raise _redis.RedisError("xclaim boom")

        with patch.object(redis_async, "xclaim", side_effect=failing_xclaim):
            await consumer.run(max_events=1)

        # xclaim failed so PEL entry is NOT re-delivered; the new message is.
        assert len(consumer.received) == 1
        assert consumer.received[0]["type"] == "new_after_stuck"

    @pytest.mark.asyncio
    async def test_empty_xclaim_result_exits_pel_drain(self, redis_async, run_id):
        """
        If xclaim returns an empty list (messages claimed by another consumer
        between xpending and xclaim), the PEL drain should exit cleanly.
        """
        run = run_id + "-pel-empty-xclaim"

        await ensure_consumer_group(redis_async, run, group_name="test-group")
        await apublish_event(redis_async, run, {"type": "race_condition"})
        key = stream_key(run)
        await redis_async.xreadgroup(
            groupname="test-group", consumername="test-0",
            streams={key: ">"}, count=1,
        )

        consumer = RecordingConsumer(redis_async, run)

        with patch.object(redis_async, "xclaim", new=AsyncMock(return_value=[])):
            # max_events=0: after PEL drain exits, loop does one iteration then stops.
            await consumer.run(max_events=0)

        assert consumer.received == []

    @pytest.mark.asyncio
    async def test_cancel_check_during_pel_drain(self, redis_async, run_id):
        """cancel_check=True during PEL drain exits before delivering any message."""
        run = run_id + "-pel-cancel-drain"

        await ensure_consumer_group(redis_async, run, group_name="test-group")
        await apublish_event(redis_async, run, {"type": "should_not_arrive"})
        key = stream_key(run)
        await redis_async.xreadgroup(
            groupname="test-group", consumername="test-0",
            streams={key: ">"}, count=1,
        )

        consumer = RecordingConsumer(redis_async, run)
        await consumer.run(cancel_check=lambda: True)

        assert consumer.received == []


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


class TestBatchProcessing:
    @pytest.mark.asyncio
    async def test_multiple_events_in_batch_all_delivered(self, redis_async, run_id):
        """All events in a multi-message xreadgroup response must be dispatched."""
        run = run_id + "-batch"
        for i in range(4):
            await apublish_event(redis_async, run, {"type": "batch", "i": i})

        consumer = RecordingConsumer(redis_async, run)
        await consumer.run(max_events=4)

        assert len(consumer.received) == 4
        indices = sorted(e["i"] for e in consumer.received)
        assert indices == [0, 1, 2, 3]

    @pytest.mark.asyncio
    async def test_max_events_limits_total_across_batches(self, redis_async, run_id):
        """
        max_events is checked between batches. With BATCH_SIZE=1 each event is
        its own batch, so the loop stops after exactly max_events events.
        """
        run = run_id + "-maxev"
        for i in range(5):
            await apublish_event(redis_async, run, {"type": "item", "i": i})

        class OneBatchConsumer(RecordingConsumer):
            BATCH_SIZE = 1

        consumer = OneBatchConsumer(redis_async, run)
        await consumer.run(max_events=2)

        assert len(consumer.received) == 2
