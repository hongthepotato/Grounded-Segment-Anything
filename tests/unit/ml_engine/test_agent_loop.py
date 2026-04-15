"""
Unit tests for ml_engine.agent.loop.

Tests AgentLoop event dispatch, LoopState persistence/serialization,
apublish_event, and ensure_consumer_group.

Uses the async fakeredis fixture from conftest (``redis_async``). Tests that
drive AgentLoop through its ``run()`` loop are marked ``@pytest.mark.asyncio``
so that everything shares one event loop and one async client instance.
"""

from __future__ import annotations

import asyncio
import json
import pytest

from unittest.mock import patch

import redis as _redis

from ml_engine.agent.loop import (
    AgentLoop,
    LoopState,
    apublish_event,
    ensure_consumer_group,
    state_key,
)
from ml_engine.agent.stream_consumer import stream_key


# ---------------------------------------------------------------------------
# Fixtures (redis_async provided by conftest.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def run_id():
    return "loop-test-run-abc"


# ---------------------------------------------------------------------------
# apublish_event
# ---------------------------------------------------------------------------

class TestPublishEvent:
    @pytest.mark.asyncio
    async def test_returns_string_entry_id(self, redis_async, run_id):
        entry_id = await apublish_event(redis_async, run_id, {"type": "test_event"})
        assert isinstance(entry_id, str)
        assert "-" in entry_id  # Redis stream IDs are "ms-seq"

    @pytest.mark.asyncio
    async def test_event_readable_from_stream(self, redis_async, run_id):
        await apublish_event(redis_async, run_id, {"type": "hello", "payload": 42})
        key = stream_key(run_id)
        entries = await redis_async.xrange(key)
        assert len(entries) == 1
        raw = entries[0][1][b"data"]
        event = json.loads(raw)
        assert event["type"] == "hello"
        assert event["payload"] == 42

    @pytest.mark.asyncio
    async def test_multiple_events_ordered(self, redis_async, run_id):
        for i in range(3):
            await apublish_event(redis_async, run_id, {"type": "event", "seq": i})
        key = stream_key(run_id)
        entries = await redis_async.xrange(key)
        assert len(entries) == 3
        for i, (_, data) in enumerate(entries):
            event = json.loads(data[b"data"])
            assert event["seq"] == i


# ---------------------------------------------------------------------------
# ensure_consumer_group
# ---------------------------------------------------------------------------

class TestEnsureConsumerGroup:
    @pytest.mark.asyncio
    async def test_creates_group_idempotent(self, redis_async, run_id):
        await ensure_consumer_group(redis_async, run_id)
        # Second call must not raise
        await ensure_consumer_group(redis_async, run_id)

    @pytest.mark.asyncio
    async def test_group_exists_after_creation(self, redis_async, run_id):
        await ensure_consumer_group(redis_async, run_id)
        key = stream_key(run_id)
        groups = await redis_async.xinfo_groups(key)
        names = [g[b"name"] if b"name" in g else g["name"] for g in groups]
        assert b"coordinator" in names or "coordinator" in names


# ---------------------------------------------------------------------------
# LoopState serialization
# ---------------------------------------------------------------------------

class TestLoopStateSerialization:
    def test_roundtrip_empty(self, run_id):
        state = LoopState(run_id=run_id)
        d = state.to_dict()
        recovered = LoopState.from_dict(d)
        assert recovered.run_id == run_id
        assert recovered.messages == []
        assert recovered.last_event_id == "0-0"
        assert recovered.stage_just_completed is None

    def test_roundtrip_with_messages(self, run_id):
        state = LoopState(
            run_id=run_id,
            messages=[{"role": "user", "content": "hello"}],
            last_event_id="12345-0",
            stage_just_completed="teacher_training",
        )
        recovered = LoopState.from_dict(state.to_dict())
        assert recovered.messages[0]["content"] == "hello"
        assert recovered.last_event_id == "12345-0"
        assert recovered.stage_just_completed == "teacher_training"

    def test_stage_just_completed_empty_string_becomes_none(self, run_id):
        d = {
            "run_id": run_id,
            "messages": "[]",
            "last_event_id": "0-0",
            "stage_just_completed": "",
        }
        state = LoopState.from_dict(d)
        assert state.stage_just_completed is None

    def test_messages_as_raw_list_accepted(self, run_id):
        """from_dict accepts pre-parsed list (not only JSON string)."""
        d = {
            "run_id": run_id,
            "messages": [{"role": "user", "content": "hi"}],
            "last_event_id": "0-0",
            "stage_just_completed": "",
        }
        state = LoopState.from_dict(d)
        assert state.messages[0]["role"] == "user"


# ---------------------------------------------------------------------------
# AgentLoop._load_state / _save_state
# ---------------------------------------------------------------------------

class TestAgentLoopStatePersistence:
    def _make_loop(self, redis_async, run_id):
        async def noop(event, state):
            pass
        return AgentLoop(redis_async, run_id, on_event=noop)

    @pytest.mark.asyncio
    async def test_load_state_returns_default_when_no_state(self, redis_async, run_id):
        loop = self._make_loop(redis_async, run_id)
        state = await loop._load_state()
        assert state.run_id == run_id
        assert state.messages == []

    @pytest.mark.asyncio
    async def test_save_and_load_roundtrip(self, redis_async, run_id):
        loop = self._make_loop(redis_async, run_id)
        state = LoopState(
            run_id=run_id,
            messages=[{"role": "user", "content": "event1"}],
            last_event_id="99-0",
        )
        await loop._save_state(state)
        recovered = await loop._load_state()
        assert recovered.messages[0]["content"] == "event1"
        assert recovered.last_event_id == "99-0"


# ---------------------------------------------------------------------------
# AgentLoop.run() -- event dispatch
# ---------------------------------------------------------------------------

class TestAgentLoopRun:
    """
    These tests publish events to Redis and verify the on_event handler
    receives them. We use max_events=N to bound the loop.
    """

    @pytest.mark.asyncio
    async def test_single_event_dispatched(self, redis_async, run_id):
        received = []

        async def handler(event, state):
            received.append(event)

        loop = AgentLoop(redis_async, run_id, on_event=handler)
        await apublish_event(redis_async, run_id, {"type": "contract_approved"})

        await loop.run(max_events=1)

        assert len(received) == 1
        assert received[0]["type"] == "contract_approved"

    @pytest.mark.asyncio
    async def test_multiple_events_dispatched_in_order(self, redis_async, run_id):
        received = []

        async def handler(event, state):
            received.append(event["seq"])

        loop = AgentLoop(redis_async, run_id + "-multi", on_event=handler)
        for i in range(3):
            await apublish_event(redis_async, run_id + "-multi", {"type": "ping", "seq": i})

        await loop.run(max_events=3)
        assert received == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_event_appended_to_messages(self, redis_async, run_id):
        captured_state = []

        async def handler(event, state):
            captured_state.append(list(state.messages))

        loop = AgentLoop(redis_async, run_id + "-msgs", on_event=handler)
        await apublish_event(redis_async, run_id + "-msgs", {"type": "test_event"})
        await loop.run(max_events=1)

        assert len(captured_state) == 1
        messages = captured_state[0]
        assert any("test_event" in m.get("content", "") for m in messages)

    @pytest.mark.asyncio
    async def test_state_persisted_after_event(self, redis_async, run_id):
        async def handler(event, state):
            state.stage_just_completed = "teacher_training"

        run = run_id + "-persist"
        loop = AgentLoop(redis_async, run, on_event=handler)
        await apublish_event(redis_async, run, {"type": "job_completed"})
        await loop.run(max_events=1)

        # State should be in Redis
        raw = await redis_async.hgetall(state_key(run))
        assert raw  # non-empty

    @pytest.mark.asyncio
    async def test_ack_prevents_redelivery(self, redis_async, run_id):
        """Events ACK'd in first run must not appear in second run."""
        event_seen = []

        async def handler(event, state):
            event_seen.append(event["type"])

        run = run_id + "-ack"
        await apublish_event(redis_async, run, {"type": "once"})

        loop1 = AgentLoop(redis_async, run, on_event=handler)
        await loop1.run(max_events=1)

        loop2 = AgentLoop(redis_async, run, on_event=handler)
        # Publish a second distinct event so max_events=1 terminates
        await apublish_event(redis_async, run, {"type": "second"})
        await loop2.run(max_events=1)

        # handler called exactly twice (once per loop, one event each)
        assert len(event_seen) == 2
        assert event_seen == ["once", "second"]

    @pytest.mark.asyncio
    async def test_cancel_check_stops_loop(self, redis_async, run_id):
        called = [0]

        async def handler(event, state):
            called[0] += 1

        run = run_id + "-cancel"
        # Do NOT publish any events -- loop should exit via cancel_check
        loop = AgentLoop(redis_async, run, on_event=handler, consumer_name="test-consumer")

        cancel = [False]

        def check():
            cancel[0] = True
            return True  # cancel immediately

        await loop.run(cancel_check=check, max_events=10)
        assert called[0] == 0

    @pytest.mark.asyncio
    async def test_handler_exception_does_not_crash_loop(self, redis_async, run_id):
        """A crashing handler should be swallowed; loop continues and ACKs."""
        call_count = [0]

        async def bad_handler(event, state):
            call_count[0] += 1
            raise RuntimeError("boom")

        run = run_id + "-exc"
        await apublish_event(redis_async, run, {"type": "bad"})
        await apublish_event(redis_async, run, {"type": "also_bad"})

        loop = AgentLoop(redis_async, run, on_event=bad_handler)
        # Should not raise
        await loop.run(max_events=2)
        assert call_count[0] == 2


# ---------------------------------------------------------------------------
# New field roundtrips (stage_start_idx, stage_dispatch_overrides)
# ---------------------------------------------------------------------------

class TestLoopStateNewFieldsRoundtrip:
    """Regression tests for fields added after the initial LoopState schema."""

    def test_stage_start_idx_roundtrip(self, run_id):
        """stage_start_idx survives to_dict/from_dict and handles None."""
        state = LoopState(run_id=run_id, stage_start_idx=3)
        recovered = LoopState.from_dict(state.to_dict())
        assert recovered.stage_start_idx == 3

        # None round-trips through empty string
        none_state = LoopState(run_id=run_id, stage_start_idx=None)
        serialized = none_state.to_dict()
        assert serialized["stage_start_idx"] == ""
        assert LoopState.from_dict(serialized).stage_start_idx is None

    def test_stage_dispatch_overrides_roundtrip(self, run_id):
        """stage_dispatch_overrides is JSON-encoded and survives roundtrip."""
        overrides = {"batch_size": 8, "models.grounding_dino.lora.r": 16}
        state = LoopState(
            run_id=run_id,
            stage_dispatch_overrides=overrides,
        )
        recovered = LoopState.from_dict(state.to_dict())
        assert recovered.stage_dispatch_overrides == overrides

        # Empty dict round-trips cleanly too
        empty = LoopState(run_id=run_id)
        assert LoopState.from_dict(empty.to_dict()).stage_dispatch_overrides == {}


# ---------------------------------------------------------------------------
# Redis error retry path
# ---------------------------------------------------------------------------

class TestRedisErrorRetry:
    @pytest.mark.asyncio
    async def test_redis_error_retries(self, redis_async, run_id):
        """Transient RedisError on xreadgroup should not kill the loop."""
        run = run_id + "-redis-err"
        await apublish_event(redis_async, run, {"type": "survived"})

        received = []

        async def handler(event, state):
            received.append(event)

        loop = AgentLoop(redis_async, run, on_event=handler)

        # Wrap xreadgroup to raise RedisError the first time it's called with
        # ">" (the main loop), then delegate to the real method. The PEL sweep
        # uses id="0" so the flaky guard only trips the main-loop read.
        real_xreadgroup = redis_async.xreadgroup
        raised = {"count": 0}

        async def flaky(*args, **kwargs):
            streams = kwargs.get("streams", {})
            if ">" in streams.values() and raised["count"] == 0:
                raised["count"] += 1
                raise _redis.RedisError("transient network blip")
            return await real_xreadgroup(*args, **kwargs)

        with patch.object(redis_async, "xreadgroup", side_effect=flaky):
            async def no_sleep(_t):
                return None
            with patch("ml_engine.agent.stream_consumer.asyncio.sleep", new=no_sleep):
                await loop.run(max_events=1)

        assert raised["count"] == 1, "expected exactly one injected failure"
        assert len(received) == 1
        assert received[0]["type"] == "survived"


# ---------------------------------------------------------------------------
# PEL recovery on restart
# ---------------------------------------------------------------------------

class TestPELRecovery:
    @pytest.mark.asyncio
    async def test_pel_recovery_on_restart(self, redis_async, run_id):
        """
        Simulate a process crash mid-event: read via XREADGROUP but never ACK.
        A new AgentLoop with the same consumer_name must re-deliver the event
        via the startup PEL sweep.
        """
        run = run_id + "-pel"
        await ensure_consumer_group(redis_async, run)
        await apublish_event(redis_async, run, {"type": "crashed_event"})

        # Simulate a prior process that read the event but crashed before ACK.
        # The message is now in the PEL for consumer "coordinator-0".
        key = stream_key(run)
        entries = await redis_async.xreadgroup(
            groupname="coordinator",
            consumername="coordinator-0",
            streams={key: ">"},
            count=1,
        )
        assert entries and entries[0][1], "message should have been delivered"
        # Deliberately do NOT xack -- this message is now stuck in the PEL.

        # A second loop with id=">" alone would never see this message.
        # The startup PEL sweep in run() must reclaim it.
        received = []

        async def handler(event, state):
            received.append(event)

        loop = AgentLoop(redis_async, run, on_event=handler, consumer_name="coordinator-0")
        await loop.run(max_events=1)

        assert len(received) == 1, "PEL entry was not replayed on restart"
        assert received[0]["type"] == "crashed_event"


# ---------------------------------------------------------------------------
# Cancellation semantics (new in async migration)
# ---------------------------------------------------------------------------

class TestCancellationSemantics:
    @pytest.mark.asyncio
    async def test_cancelled_handler_leaves_message_unacked(self, redis_async, run_id):
        """
        If the handler raises asyncio.CancelledError, the message must NOT be
        ACKed -- the PEL will replay it on next start. Non-cancellation
        exceptions still ACK (poison safety), but cancellation means
        ``work was interrupted, try again later``.
        """
        run = run_id + "-cancel-pel"
        await apublish_event(redis_async, run, {"type": "interrupted"})

        async def cancelled_handler(event, state):
            raise asyncio.CancelledError()

        loop = AgentLoop(redis_async, run, on_event=cancelled_handler)

        # Run should propagate CancelledError out of the _dispatch path.
        with pytest.raises(asyncio.CancelledError):
            await loop.run(max_events=1)

        # Message must still be pending (unacked) -- PEL recovery will replay it.
        key = stream_key(run)
        pending = await redis_async.xpending_range(
            key, "coordinator",
            min="-", max="+", count=10,
            consumername="coordinator-0",
        )
        assert len(pending) == 1, "cancelled handler's message must remain in PEL"

    @pytest.mark.asyncio
    async def test_cancelled_then_restart_replays_event(self, redis_async, run_id):
        """End-to-end: cancel mid-handler, restart, verify event re-delivered."""
        run = run_id + "-cancel-replay"
        await apublish_event(redis_async, run, {"type": "survive_cancel"})

        first_attempts = []

        async def cancelled_handler(event, state):
            first_attempts.append(event)
            raise asyncio.CancelledError()

        loop1 = AgentLoop(redis_async, run, on_event=cancelled_handler)
        with pytest.raises(asyncio.CancelledError):
            await loop1.run(max_events=1)
        assert len(first_attempts) == 1

        # Restart with a handler that succeeds. PEL sweep should replay.
        second_attempts = []

        async def good_handler(event, state):
            second_attempts.append(event)

        loop2 = AgentLoop(redis_async, run, on_event=good_handler, consumer_name="coordinator-0")
        await loop2.run(max_events=1)
        assert len(second_attempts) == 1
        assert second_attempts[0]["type"] == "survive_cancel"
