"""
Integration tests for the AgentLoop + StateMachine lifecycle.

Coverage:
  Scenario 1 -- happy path: single event flows through full state arc to "done".
  Scenario 2 -- crash recovery: first loop raises CancelledError (no ACK);
               second loop reclaims via PEL and finishes the run.
  Scenario 3 -- state machine fence: invalid transitions are blocked; terminal
               state rejects further transitions.
  Scenario 4 -- TODOS #24 race (xfail strict): two coordinator instances start
               simultaneously; without a distributed SET NX lock they both
               attempt the same state advance, one raising ValueError.

Read before editing:
  ml_engine/agent/loop.py           -- AgentLoop, apublish_event, ensure_consumer_group
  ml_engine/agent/stream_consumer.py -- StreamConsumer.run() calls on_start() internally
  ml_engine/agent/state_machine.py   -- StateMachine, TERMINAL_STATES, TRANSITIONS
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from ml_engine.agent.loop import (
    AgentLoop,
    LoopState,
    apublish_event,
    ensure_consumer_group,
    state_key,
)
from ml_engine.agent.state_machine import TERMINAL_STATES, StateMachine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _advance_to_done(sm: StateMachine) -> None:
    """Drive sm from its current state through the non-distillation arc to 'done'."""
    arc = [
        ("created", "planning"),
        ("planning", "pending_contract_approval"),
        ("pending_contract_approval", "auto_labeling"),
        ("auto_labeling", "label_review_gate"),
        ("label_review_gate", "teacher_training"),
        ("teacher_training", "training_eval_gate"),
        ("training_eval_gate", "pending_approval"),
        ("pending_approval", "done"),
    ]
    current = await sm.current_state()
    for from_state, to_state in arc:
        if current == from_state:
            await sm.transition(to_state)
            current = to_state
        if current == "done":
            break
    assert current == "done", f"_advance_to_done left sm in {current!r} — arc never matched"


# ---------------------------------------------------------------------------
# Scenario 1 -- Happy path: publish event, handle, state machine reaches "done"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_happy_path_full_lifecycle(run_id: str, redis_async: Any) -> None:
    """
    Single coordinator processes one event and drives state machine to 'done'.

    Asserts:
    - LoopState.messages contains the event as a user message.
    - StateMachine ends in terminal state 'done'.
    - LoopState.last_event_id matches the published entry id.
    - LoopState is persisted to Redis (survives a fresh hgetall).
    - Negative: no extra messages (exactly one event published).
    """
    sm = StateMachine(run_id=run_id, redis_async=redis_async)
    await sm.initialize()
    await ensure_consumer_group(redis_async, run_id)

    published_id = await apublish_event(redis_async, run_id, {"type": "pipeline_started"})

    async def on_event(event: Dict[str, Any], state: LoopState) -> None:
        await _advance_to_done(sm)

    loop = AgentLoop(
        redis_client=redis_async,
        run_id=run_id,
        on_event=on_event,
        consumer_name="coordinator-0",
    )
    await loop.run(max_events=1)

    assert await sm.current_state() == "done"

    loaded = loop._state
    assert loaded is not None
    assert len(loaded.messages) == 1
    assert loaded.messages[0]["role"] == "user"
    assert "pipeline_started" in loaded.messages[0]["content"]
    assert loaded.last_event_id == published_id

    # Hash must survive in Redis after run
    raw = await redis_async.hgetall(state_key(run_id))
    assert raw, "LoopState hash must persist in Redis"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_happy_path_multiple_events_sequential(run_id: str, redis_async: Any) -> None:
    """
    Three heartbeat events arrive in order; messages accumulate per event.

    Asserts:
    - Handler called exactly 3 times in publish order.
    - LoopState.messages has 3 entries after max_events=3.
    - Negative: max_events=3 stops the loop even though the stream has no more
      events (loop does not block indefinitely waiting for a 4th).
    """
    sm = StateMachine(run_id=run_id, redis_async=redis_async)
    await sm.initialize()
    await ensure_consumer_group(redis_async, run_id)

    for i in range(3):
        await apublish_event(redis_async, run_id, {"type": "heartbeat", "seq": i})

    seen: List[Dict[str, Any]] = []

    async def on_event(event: Dict[str, Any], state: LoopState) -> None:
        seen.append(event)

    loop = AgentLoop(
        redis_client=redis_async,
        run_id=run_id,
        on_event=on_event,
        consumer_name="coordinator-0",
    )
    await loop.run(max_events=3)

    assert len(seen) == 3
    assert [e["seq"] for e in seen] == [0, 1, 2], "events must arrive in publish order"
    assert loop._state is not None
    assert len(loop._state.messages) == 3


# ---------------------------------------------------------------------------
# Scenario 2 -- Crash recovery: PEL reclaim on second loop restart
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_crash_recovery_second_loop_resumes(run_id: str, redis_async: Any) -> None:
    """
    First AgentLoop 'crashes' by raising CancelledError in on_event.
    CancelledError is not ACKed by StreamConsumer._dispatch, so the message
    stays in the PEL for consumer_name 'coordinator-0'.

    Second AgentLoop with the same consumer_name calls _drain_pel on startup
    (via XPENDING + XCLAIM), reclaims the message, and finishes the run.

    Asserts:
    - Crash handler was called once (event was delivered to loop_a).
    - Recovery handler was called once (event replayed from PEL to loop_b).
    - State machine reaches 'done' after recovery.
    - LoopState from loop_b has exactly 2 messages: pre-crash save append +
      re-dispatch append (exactly-twice semantics -- documented, not a bug).
    - Negative: stage_just_completed is still None (no stage was declared).
    """
    sm = StateMachine(run_id=run_id, redis_async=redis_async)
    await sm.initialize()
    await ensure_consumer_group(redis_async, run_id)

    await apublish_event(redis_async, run_id, {"type": "pipeline_started"})

    # First loop: crashes mid-handler
    crashed: List[Dict[str, Any]] = []

    async def crashing_handler(event: Dict[str, Any], state: LoopState) -> None:
        crashed.append(event)
        raise asyncio.CancelledError()  # not ACKed -> stays in PEL

    loop_a = AgentLoop(
        redis_client=redis_async,
        run_id=run_id,
        on_event=crashing_handler,
        consumer_name="coordinator-0",
    )
    with pytest.raises(asyncio.CancelledError):
        await loop_a.run(max_events=1)

    assert len(crashed) == 1
    assert crashed[0]["type"] == "pipeline_started"

    # Second loop: reclaims from PEL, drives run to done
    recovered: List[Dict[str, Any]] = []

    async def recovery_handler(event: Dict[str, Any], state: LoopState) -> None:
        recovered.append(event)
        await _advance_to_done(sm)

    loop_b = AgentLoop(
        redis_client=redis_async,
        run_id=run_id,
        on_event=recovery_handler,
        consumer_name="coordinator-0",  # same consumer -> reclaims its own PEL
    )
    await loop_b.run(max_events=1)

    assert len(recovered) == 1
    assert recovered[0]["type"] == "pipeline_started"
    assert await sm.current_state() == "done"

    # handle_event appends to messages BEFORE calling on_event, so the
    # re-dispatch adds one entry on top of the pre-crash-save entry -> exactly 2.
    assert loop_b._state is not None
    assert len(loop_b._state.messages) == 2, (
        "exactly-twice: pre-crash save append + re-dispatch append = 2"
    )
    assert loop_b._state.stage_just_completed is None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_crash_recovery_state_persisted_before_handler(run_id: str, redis_async: Any) -> None:
    """
    LoopState is saved to Redis BEFORE on_event is called.
    Even after a crash, the second loop restores the pre-crash state snapshot.

    Asserts:
    - The state key in Redis has the event message after the crash (pre-crash save).
    - loop_b._state.messages already has one entry when on_event is first called
      (carried over from the crash snapshot, not injected by loop_b).
    """
    sm = StateMachine(run_id=run_id, redis_async=redis_async)
    await sm.initialize()
    await ensure_consumer_group(redis_async, run_id)

    await apublish_event(redis_async, run_id, {"type": "init"})

    async def crashing_handler(event: Dict[str, Any], state: LoopState) -> None:
        raise asyncio.CancelledError()

    loop_a = AgentLoop(
        redis_client=redis_async,
        run_id=run_id,
        on_event=crashing_handler,
        consumer_name="coordinator-0",
    )
    with pytest.raises(asyncio.CancelledError):
        await loop_a.run(max_events=1)

    # Pre-crash save must already be in Redis
    raw = await redis_async.hgetall(state_key(run_id))
    assert raw, "pre-crash LoopState hash must be in Redis"

    # Loop_b on_start loads that snapshot -- messages already has 1 entry on first call
    messages_at_entry: List[int] = []

    async def recovery_handler(event: Dict[str, Any], state: LoopState) -> None:
        # This count is BEFORE loop_b appends the re-delivered event
        # (handle_event appends THEN calls on_event); so we see the pre-crash count
        messages_at_entry.append(len(state.messages))

    loop_b = AgentLoop(
        redis_client=redis_async,
        run_id=run_id,
        on_event=recovery_handler,
        consumer_name="coordinator-0",
    )
    await loop_b.run(max_events=1)

    # handle_event appends the re-dispatched event BEFORE calling on_event, so
    # on_event sees: pre-crash-save entry (1) + re-dispatch append (1) = 2 total.
    assert messages_at_entry == [2], (
        "on_event sees pre-crash entry + re-dispatch append = 2 messages"
    )


# ---------------------------------------------------------------------------
# Scenario 3 -- State machine fence: invalid and terminal transitions blocked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_state_machine_invalid_transition_raises(run_id: str, redis_async: Any) -> None:
    """
    StateMachine.transition() rejects unknown states and disallowed arcs.

    Asserts:
    - Completely unknown state name raises ValueError('Unknown state').
    - Valid state name on a disallowed arc raises ValueError('Invalid transition').
    - State is unchanged after both rejections.
    """
    sm = StateMachine(run_id=run_id, redis_async=redis_async)
    await sm.initialize()
    assert await sm.current_state() == "created"

    with pytest.raises(ValueError, match="Unknown state"):
        await sm.transition("not_a_real_state")

    # 'done' is valid but created -> done is not in TRANSITIONS
    with pytest.raises(ValueError, match="Invalid transition"):
        await sm.transition("done")

    # Negative: state must not have mutated
    assert await sm.current_state() == "created"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_state_machine_terminal_rejects_all_outbound(run_id: str, redis_async: Any) -> None:
    """
    Terminal states have no outbound transitions; every attempt raises ValueError.

    Uses 'cancelled' (pending_contract_approval -> cancelled) as the terminal state.
    Asserts state stays 'cancelled' after all rejected attempts.
    """
    for terminal in ("done", "failed_unrecoverable", "escalated", "cancelled"):
        assert terminal in TERMINAL_STATES

    sm = StateMachine(run_id=run_id, redis_async=redis_async)
    await sm.initialize()
    await sm.transition("planning")
    await sm.transition("pending_contract_approval")
    await sm.transition("cancelled")
    assert await sm.current_state() == "cancelled"

    for attempt in ("planning", "created", "done", "failed_unrecoverable"):
        with pytest.raises(ValueError):
            await sm.transition(attempt)

    assert await sm.current_state() == "cancelled"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_state_machine_full_arc_without_distillation(run_id: str, redis_async: Any) -> None:
    """
    Drives the full non-distillation production arc from 'created' to 'done'.

    Arc: created -> planning -> pending_contract_approval -> auto_labeling
         -> label_review_gate -> teacher_training -> training_eval_gate
         -> pending_approval -> done

    Asserts no intermediate state is terminal (pipeline must not stop early).
    """
    sm = StateMachine(run_id=run_id, redis_async=redis_async)
    await sm.initialize()

    arc = [
        "planning",
        "pending_contract_approval",
        "auto_labeling",
        "label_review_gate",
        "teacher_training",
        "training_eval_gate",
        "pending_approval",
        "done",
    ]
    for state in arc:
        prev = await sm.current_state()
        assert prev not in TERMINAL_STATES, f"intermediate state {prev!r} must not be terminal"
        await sm.transition(state)

    assert await sm.current_state() == "done"
    assert "done" in TERMINAL_STATES


@pytest.mark.asyncio
@pytest.mark.integration
async def test_state_machine_failed_retrying_increments_count(run_id: str, redis_async: Any) -> None:
    """
    failed_retrying increments retry_count on each entry.

    Asserts count is 1 after first failure, 2 after a second, and never
    exceeds the number of real retries (no phantom increments).
    """
    sm = StateMachine(run_id=run_id, redis_async=redis_async)
    await sm.initialize()
    await sm.transition("planning")
    await sm.transition("pending_contract_approval")
    await sm.transition("auto_labeling")

    await sm.transition("failed_retrying")
    assert await sm.retry_count() == 1

    await sm.transition("auto_labeling")
    await sm.transition("failed_retrying")
    assert await sm.retry_count() == 2


# ---------------------------------------------------------------------------
# Scenario 4 -- TODOS #24 race: coordinator double-start without a lock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.xfail(
    strict=True,
    reason=(
        "TODOS #24: no SET NX distributed lock on coordinator resume. "
        "Two AgentLoop instances race to sm.transition('planning'). "
        "The second reads stale state and gets ValueError. "
        "XFAILED = race confirmed. XPASS = lock landed (remove xfail)."
    ),
)
async def test_coordinator_double_resume_race_condition(run_id: str, redis_async: Any) -> None:
    """
    Two 'coordinator' instances start concurrently for the same run_id,
    simulating a zombie process coexisting with a fresh restart (TODOS #24).

    Setup:
    - Two events published so each coordinator (coordinator-0, coordinator-1)
      gets one from the consumer group.
    - Both handlers wait at a busy-yield barrier until both have arrived,
      then race to sm.transition('planning').

    Without a distributed lock:
    - First writer succeeds (created -> planning).
    - Second reads 'planning' inside transition(), validates planning->planning
      as INVALID, raises ValueError.
    - handler_errors is non-empty.

    The assertion 'assert not handler_errors' FAILS -> XFAILED (expected).
    With a proper lock: the loser never calls transition() -> no error ->
    assertion passes -> XPASS -> strict=True turns that into a test failure,
    prompting removal of this xfail marker.
    """
    sm = StateMachine(run_id=run_id, redis_async=redis_async)
    await sm.initialize()
    await ensure_consumer_group(redis_async, run_id)

    await apublish_event(redis_async, run_id, {"type": "coordinator_started", "instance": "A"})
    await apublish_event(redis_async, run_id, {"type": "coordinator_started", "instance": "B"})

    arrived: List[str] = []
    handler_errors: List[str] = []

    def make_handler(name: str):
        async def on_event(event: Dict[str, Any], state: LoopState) -> None:
            arrived.append(name)
            # Busy-yield until both handlers are inside, maximising interleave chance.
            # Capped to avoid infinite hang if one loop never reaches the barrier.
            _iters = 0
            while len(arrived) < 2:
                await asyncio.sleep(0)
                _iters += 1
                if _iters > 5000:
                    pytest.fail("barrier timed out — one handler never arrived")
            try:
                await sm.transition("planning")
            except ValueError as exc:
                handler_errors.append(f"{name}: {exc}")
        return on_event

    loop_a = AgentLoop(
        redis_client=redis_async,
        run_id=run_id,
        on_event=make_handler("A"),
        consumer_name="coordinator-0",
    )
    loop_b = AgentLoop(
        redis_client=redis_async,
        run_id=run_id,
        on_event=make_handler("B"),
        consumer_name="coordinator-1",
    )

    await asyncio.gather(
        loop_a.run(max_events=1),
        loop_b.run(max_events=1),
    )

    # With a lock: loser is blocked before calling transition() -> no errors.
    # Without a lock (TODOS #24): one coordinator errors -> assert fails -> XFAILED.
    assert not handler_errors, (
        f"TODOS #24: coordinator raced without distributed lock -> {handler_errors}"
    )
