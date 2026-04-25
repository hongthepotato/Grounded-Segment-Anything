"""
Unit tests for ml_engine.agent.state_machine.StateMachine.

Async-only after Phase 7 (sync API removed). Uses the ``redis_async``
fixture from conftest.
"""

from __future__ import annotations

import asyncio
import json

import pytest
import pytest_asyncio

from ml_engine.agent.state_machine import (
    STATES,
    TERMINAL_STATES,
    TRANSITIONS,
    StateMachine,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def run_id():
    return "test-run-0000-1111-2222"


@pytest_asyncio.fixture
async def sm(redis_async, run_id):
    """Return an initialized StateMachine."""
    machine = StateMachine(run_id=run_id, redis_async=redis_async)
    await machine.initialize()
    return machine


@pytest.fixture
def contract_dict():
    return {
        "id": "contract-abc",
        "target": {"class_names": ["defect"], "output_mode": "detection"},
        "budget": {"max_retries": 2},
    }


# ---------------------------------------------------------------------------
# initialize()
# ---------------------------------------------------------------------------


class TestInitialize:
    @pytest.mark.asyncio
    async def test_sets_created_state(self, sm):
        assert await sm.current_state() == "created"
        assert await sm.retry_count() == 0
        assert await sm.get_proposed_contract() is None

    @pytest.mark.asyncio
    async def test_stores_contract(self, redis_async, run_id, contract_dict):
        machine = StateMachine(run_id=run_id + "-c", redis_async=redis_async)
        await machine.initialize(contract=contract_dict)
        stored = await machine.get_proposed_contract()
        assert stored is not None
        assert stored["id"] == "contract-abc"

    @pytest.mark.asyncio
    async def test_initialize_sets_required_fields(self, redis_async, run_id):
        machine = StateMachine(run_id=run_id + "-fields", redis_async=redis_async)
        await machine.initialize()
        data = await machine.load()
        assert "created_at" in data
        assert "updated_at" in data
        assert data["run_id"] == run_id + "-fields"
        assert data["contract_id"] == ""
        assert data["error_message"] == ""
        assert data["stage_summaries"] == "[]"
        assert data["state"] == "created"
        assert data["retry_count"] == "0"


# ---------------------------------------------------------------------------
# current_state()
# ---------------------------------------------------------------------------


class TestCurrentState:
    @pytest.mark.asyncio
    async def test_raises_key_error_for_unknown_run(self, redis_async):
        machine = StateMachine(run_id="no-such-run", redis_async=redis_async)
        with pytest.raises(KeyError):
            await machine.current_state()


class TestLoad:
    @pytest.mark.asyncio
    async def test_raises_key_error_for_unknown_run(self, redis_async):
        machine = StateMachine(run_id="never-initialized", redis_async=redis_async)
        with pytest.raises(KeyError):
            await machine.load()


# ---------------------------------------------------------------------------
# transition()
# ---------------------------------------------------------------------------


class TestTransition:
    @pytest.mark.asyncio
    async def test_valid_transition_created_to_planning(self, sm):
        await sm.transition("planning")
        assert await sm.current_state() == "planning"

    # sorted() to make parametrize IDs deterministic across xdist workers.
    # STATES is a set; iteration order is hash-randomized per process.
    ALL_POSSIBLE_PAIRS = [(src, dst) for src in sorted(STATES) for dst in sorted(STATES)]

    @pytest.mark.parametrize("src,dst", ALL_POSSIBLE_PAIRS)
    @pytest.mark.asyncio
    async def test_state_transitions(self, sm, src, dst):
        """Every possible (src, dst): valid -> success; invalid -> ValueError."""
        await sm._r.hset(sm._key, "state", src)
        is_allowed = dst in TRANSITIONS.get(src, [])
        if is_allowed:
            await sm.transition(dst)
            assert await sm.current_state() == dst
        else:
            if src in TERMINAL_STATES:
                with pytest.raises(ValueError, match="terminal state"):
                    await sm.transition(dst)
            else:
                with pytest.raises(ValueError, match="Invalid transition"):
                    await sm.transition(dst)

    @pytest.mark.asyncio
    async def test_unknown_state_raises_value_error(self, sm):
        with pytest.raises(ValueError, match="Unknown state"):
            await sm.transition("not_a_real_state")

    ALL_STATES = sorted(STATES)  # deterministic across xdist workers

    @pytest.mark.parametrize("terminal_state", ALL_STATES)
    @pytest.mark.asyncio
    async def test_terminal_state_blocks_further_transitions(self, sm, terminal_state):
        await sm.transition("planning")
        await sm.transition("pending_contract_approval")
        await sm.transition("teacher_training")
        await sm.transition("training_eval_gate")
        await sm.transition("pending_approval")
        await sm.transition("done")
        with pytest.raises(ValueError, match="terminal state"):
            await sm.transition(terminal_state)

    @pytest.mark.asyncio
    async def test_failed_retrying_increments_retry_count(self, sm):
        await sm.transition("planning")
        await sm.transition("pending_contract_approval")
        await sm.transition("auto_labeling")
        await sm.transition("failed_retrying")
        assert await sm.retry_count() == 1

    @pytest.mark.asyncio
    async def test_retry_count_accumulates(self, sm):
        await sm.transition("planning")
        await sm.transition("pending_contract_approval")
        await sm.transition("auto_labeling")
        await sm.transition("failed_retrying")
        await sm.transition("auto_labeling")
        await sm.transition("failed_retrying")
        assert await sm.retry_count() == 2

    @pytest.mark.asyncio
    async def test_transition_stores_error_message(self, sm):
        await sm.transition("planning")
        await sm.transition("failed_unrecoverable", error_message="OOM on epoch 3")
        data = await sm.load()
        assert data["error_message"] == "OOM on epoch 3"

    @pytest.mark.asyncio
    async def test_transition_stores_metadata(self, sm):
        await sm.transition("planning", metadata={"note": "test"})
        data = await sm.load()
        assert json.loads(data["metadata"]) == {"note": "test"}

    ALL_TERMINAL_PAIRS = [(src, dst) for src in sorted(TERMINAL_STATES) for dst in sorted(STATES)]

    @pytest.mark.parametrize("src, dst", ALL_TERMINAL_PAIRS)
    @pytest.mark.asyncio
    async def test_all_terminal_states_are_blocked(self, redis_async, src, dst):
        """Every state in TERMINAL_STATES should refuse further transitions."""
        m = StateMachine(run_id="run-" + src, redis_async=redis_async)
        await m.initialize()
        await redis_async.hset(m._key, "state", src)
        with pytest.raises(ValueError):
            await m.transition(dst)


# ---------------------------------------------------------------------------
# Full valid pipeline walk
# ---------------------------------------------------------------------------


class TestFullPipelineWalk:
    @pytest.mark.asyncio
    async def test_happy_path_detection_pipeline(self, redis_async, run_id):
        sm = StateMachine(run_id=run_id + "-happy", redis_async=redis_async)
        await sm.initialize()
        path = [
            "planning",
            "pending_contract_approval",
            "teacher_training",
            "training_eval_gate",
            "pending_approval",
            "done",
        ]
        for state in path:
            old_time = await sm._r.hget(sm._key, "updated_at")
            await asyncio.sleep(0.02)
            await sm.transition(state)
            assert await sm.current_state() == state
            assert await sm._r.hget(sm._key, "updated_at") != old_time
        assert await sm.current_state() == "done"

    @pytest.mark.asyncio
    async def test_happy_path_with_distillation(self, redis_async, run_id):
        sm = StateMachine(run_id=run_id + "-distill", redis_async=redis_async)
        await sm.initialize()
        path = [
            "planning",
            "pending_contract_approval",
            "auto_labeling",
            "label_review_gate",
            "teacher_training",
            "training_eval_gate",
            "student_distillation",
            "distill_eval_gate",
            "pending_approval",
            "done",
        ]
        for state in path:
            old_time = await sm._r.hget(sm._key, "updated_at")
            await asyncio.sleep(0.02)
            await sm.transition(state)
            assert await sm.current_state() == state
            assert await sm._r.hget(sm._key, "updated_at") != old_time
        assert await sm.current_state() == "done"

    @pytest.mark.asyncio
    async def test_retry_then_escalate(self, redis_async, run_id):
        sm = StateMachine(run_id=run_id + "-retry", redis_async=redis_async)
        await sm.initialize()
        await sm.transition("planning")
        await sm.transition("pending_contract_approval")
        await sm.transition("teacher_training")
        await sm.transition("failed_retrying")
        await sm.transition("teacher_training")
        await sm.transition("failed_unrecoverable")
        assert await sm.current_state() == "failed_unrecoverable"

    @pytest.mark.asyncio
    async def test_cancel_from_pending_approval(self, redis_async, run_id):
        sm = StateMachine(run_id=run_id + "-cancel", redis_async=redis_async)
        await sm.initialize()
        await sm.transition("planning")
        await sm.transition("pending_contract_approval")
        await sm.transition("teacher_training")
        await sm.transition("training_eval_gate")
        await sm.transition("pending_approval")
        await sm.transition("cancelled")
        assert await sm.current_state() == "cancelled"


# ---------------------------------------------------------------------------
# stage_summaries
# ---------------------------------------------------------------------------


class TestStageSummaries:
    @pytest.mark.asyncio
    async def test_append_and_get(self, sm):
        await sm.append_stage_summary(
            {"stage": "teacher_training", "status": "pass", "metrics": {"mAP50": 0.72}},
        )
        summaries = await sm.get_stage_summaries()
        assert len(summaries) == 1
        assert summaries[0]["stage"] == "teacher_training"

    @pytest.mark.asyncio
    async def test_append_multiple(self, sm):
        await sm.append_stage_summary({"stage": "auto_labeling", "status": "pass"})
        await sm.append_stage_summary({"stage": "teacher_training", "status": "pass"})
        assert len(await sm.get_stage_summaries()) == 2

    @pytest.mark.asyncio
    async def test_empty_initially(self, sm):
        assert await sm.get_stage_summaries() == []


# ---------------------------------------------------------------------------
# get_proposed_contract()
# ---------------------------------------------------------------------------


class TestGetProposedContract:
    @pytest.mark.asyncio
    async def test_returns_none_when_not_set(self, redis_async, run_id):
        m = StateMachine(run_id=run_id + "-nocontract", redis_async=redis_async)
        await m.initialize()
        assert await m.get_proposed_contract() is None

    @pytest.mark.asyncio
    async def test_returns_contract_when_set(self, redis_async, run_id, contract_dict):
        m = StateMachine(run_id=run_id + "-withcontract", redis_async=redis_async)
        await m.initialize(contract=contract_dict)
        result = await m.get_proposed_contract()
        assert result["id"] == "contract-abc"

    @pytest.mark.asyncio
    async def test_handles_corrupt_json_gracefully(self, redis_async, run_id):
        m = StateMachine(run_id=run_id + "-corrupt", redis_async=redis_async)
        await m.initialize()
        await redis_async.hset(m._key, "proposed_contract", b"not valid json {{{")
        assert await m.get_proposed_contract() is None

    @pytest.mark.asyncio
    async def test_handles_empty_json_object(self, redis_async, run_id):
        m = StateMachine(run_id=run_id + "-empty", redis_async=redis_async)
        await m.initialize()
        await redis_async.hset(m._key, "proposed_contract", b"{}")
        assert await m.get_proposed_contract() is None


# ---------------------------------------------------------------------------
# exists()
# ---------------------------------------------------------------------------


class TestExists:
    @pytest.mark.asyncio
    async def test_true_after_initialize(self, redis_async, run_id):
        m = StateMachine(run_id=run_id + "-exists", redis_async=redis_async)
        await m.initialize()
        assert await StateMachine.exists(redis_async, run_id + "-exists") is True

    @pytest.mark.asyncio
    async def test_false_before_initialize(self, redis_async):
        assert await StateMachine.exists(redis_async, "never-initialized") is False
