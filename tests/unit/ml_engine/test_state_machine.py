"""
Unit tests for ml_engine.agent.state_machine.StateMachine.

Uses fakeredis so no real Redis instance is required.
"""

from __future__ import annotations

import json
import pytest
import fakeredis

from ml_engine.agent.state_machine import (
    StateMachine,
    STATES,
    TERMINAL_STATES,
    TRANSITIONS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def redis():
    r"""Returns a fresh fakeredis instance for each test."""
    return fakeredis.FakeRedis(decode_responses=False)


@pytest.fixture
def run_id():
    r"""Returns a fixed run_id for testing."""
    return "test-run-0000-1111-2222"


@pytest.fixture
def sm(redis, run_id):
    r"""Returns an initialized StateMachine for testing."""
    machine = StateMachine(redis, run_id)
    machine.initialize()
    return machine


@pytest.fixture
def contract_dict():
    r"""Example contract dict for testing get_proposed_contract()."""
    return {
        "id": "contract-abc",
        "target": {"class_names": ["defect"], "output_mode": "detection"},
        "budget": {"max_retries": 2},
    }


# ---------------------------------------------------------------------------
# initialize()
# ---------------------------------------------------------------------------

class TestInitialize:
    r"""Tests for StateMachine.initialize()."""
    def test_sets_created_state(self, sm):
        r"""initialize() should set the initial state to "created"."""
        assert sm.current_state == "created"
        assert sm.retry_count == 0
        assert sm.get_proposed_contract() is None

    def test_stores_contract(self, redis, run_id, contract_dict):
        r"""initialize() should store the proposed contract in Redis."""
        machine = StateMachine(redis, run_id + "-c")
        machine.initialize(contract=contract_dict)
        stored = machine.get_proposed_contract()
        assert stored is not None
        assert stored["id"] == "contract-abc"

    def test_initialize_sets_required_fields(self, redis, run_id):
        r"""initialize() should set required fields like created_at and updated_at."""
        machine = StateMachine(redis, run_id + "-fields")
        machine.initialize()
        data = machine.load()
        assert "created_at" in data
        assert "updated_at" in data
        assert data["run_id"] == run_id + "-fields"
        assert data["contract_id"] == ""
        assert data["error_message"] == ""
        assert data["stage_summaries"] == "[]"
        assert data["state"] == "created"
        assert data["retry_count"] == "0"


# ---------------------------------------------------------------------------
# current_state property
# ---------------------------------------------------------------------------

class TestCurrentState:
    r"""current_state should reflect the "state" field in Redis, and raise KeyError if not found."""

    def test_raises_key_error_for_unknown_run(self, redis):
        r"""Accessing current_state for a uninitialized run_id should raise KeyError."""
        machine = StateMachine(redis, "no-such-run")
        with pytest.raises(KeyError):
            _ = machine.current_state


# ---------------------------------------------------------------------------
# transition()
# ---------------------------------------------------------------------------

class TestTransition:
    r"""Tests for StateMachine.transition(). Validates allowed transitions, retry count increments, and metadata storage."""
    def test_valid_transition_created_to_planning(self, sm):
        r"""Should allow transition from created to planning."""
        sm.transition("planning")
        assert sm.current_state == "planning"

    ALL_POSSIBLE_PAIRS = [(src,dst) for src in STATES for dst in STATES]
    @pytest.mark.parametrize("src,dst", ALL_POSSIBLE_PAIRS)
    def test_state_transitions(self, sm, src, dst):
        r"""Test every single combination of states, 
        if src->dst is valid, transition should succeed, 
        else should raise ValueError."""
        sm._r.hset(sm._key, "state", src)  # Force the source state directly via Redis
        is_allowed = dst in TRANSITIONS.get(src, [])
        if is_allowed:
            sm.transition(dst)
            assert sm.current_state == dst
        else:
            with pytest.raises(ValueError, match="Invalid transition"):
                sm.transition(dst)

    def test_unknown_state_raises_value_error(self, sm):
        r"""Transitioning to a state not in STATES should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown state"):
            sm.transition("not_a_real_state")

    ALL_STATES = list(STATES)
    @pytest.mark.parametrize("terminal_state", ALL_STATES)
    def test_terminal_state_blocks_further_transitions(self, sm, state):
        r"""Once in a terminal state, any further transition should raise ValueError."""
        # Walk to done via a valid path
        sm.transition("planning")
        sm.transition("pending_contract_approval")
        sm.transition("teacher_training")
        sm.transition("training_eval_gate")
        sm.transition("pending_approval")
        sm.transition("done")
        with pytest.raises(ValueError, match="terminal state"):
            sm.transition(state)

    def test_failed_retrying_increments_retry_count(self, sm):
        r"""Every time we transition into failed_retrying, retry_count should increment by 1."""
        sm.transition("planning")
        sm.transition("pending_contract_approval")
        sm.transition("auto_labeling")
        sm.transition("failed_retrying")
        assert sm.retry_count == 1

    def test_retry_count_accumulates(self, sm):
        r"""Multiple transitions into failed_retrying should accumulate retry_count."""
        sm.transition("planning")
        sm.transition("pending_contract_approval")
        sm.transition("auto_labeling")
        sm.transition("failed_retrying")
        sm.transition("auto_labeling")
        sm.transition("failed_retrying")
        assert sm.retry_count == 2

    def test_transition_stores_error_message(self, sm):
        r"""When transitioning to a failed state, should be able to store an error message."""
        sm.transition("planning")
        sm.transition("failed_unrecoverable", error_message="OOM on epoch 3")
        data = sm.load()
        assert data["error_message"] == "OOM on epoch 3"

    def test_transition_stores_metadata(self, sm):
        r"""Should be able to store arbitrary metadata on transition, retrievable via load()."""
        sm.transition("planning", metadata={"note": "test"})
        data = sm.load()
        assert json.loads(data["metadata"]) == {"note": "test"}

    ALL_POSSIBLE_PAIRS = [(src,dst) for src in TERMINAL_STATES for dst in STATES]
    @pytest.mark.parametrize("src, dst", ALL_POSSIBLE_PAIRS)
    def test_all_terminal_states_are_blocked(self, sm, src, dst):
        """Every state in TERMINAL_STATES should refuse further transitions."""
        # for terminal in TERMINAL_STATES:
        r = fakeredis.FakeRedis(decode_responses=False)
        m = StateMachine(r, "run-" + src)
        m.initialize()
        # Force the state directly via Redis
        r.hset(m._key, "state", src)
        with pytest.raises(ValueError):
            m.transition(dst)


# ---------------------------------------------------------------------------
# Full valid pipeline walk
# ---------------------------------------------------------------------------

class TestFullPipelineWalk:
    def test_happy_path_detection_pipeline(self, redis, run_id):
        r"""Walk through a full valid pipeline with no retries, from created to done."""
        sm = StateMachine(redis, run_id + "-happy")
        sm.initialize()
        path = [
            "planning",
            "pending_contract_approval",
            "teacher_training",
            "training_eval_gate",
            "pending_approval",
            "done",
        ]
        for state in path:
            old_time = sm._r.hget(sm._key, "updated_at")
            sm.transition(state)
            assert sm.current_state == state
            assert sm._r.hget(sm._key, "updated_at") != old_time  # updated_at should change on each transition
        assert sm.current_state == "done"

    def test_happy_path_with_distillation(self, redis, run_id):
        r"""Walk through a full valid pipeline that includes the distillation loop, from created to done."""
        sm = StateMachine(redis, run_id + "-distill")
        sm.initialize()
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
            old_time = sm._r.hget(sm._key, "updated_at")
            sm.transition(state)
            assert sm.current_state == state
            assert sm._r.hget(sm._key, "updated_at") != old_time
        assert sm.current_state == "done"

    def test_retry_then_escalate(self, redis, run_id):
        r"""Walk through a pipeline that hits failed_retrying twice, then escalates."""
        sm = StateMachine(redis, run_id + "-retry")
        sm.initialize()
        sm.transition("planning")
        sm.transition("pending_contract_approval")
        sm.transition("teacher_training")
        sm.transition("failed_retrying")
        sm.transition("teacher_training")
        sm.transition("failed_unrecoverable")
        assert sm.current_state == "failed_unrecoverable"

    def test_cancel_from_pending_approval(self, redis, run_id):
        r"""Test that we can cancel from pending_approval state."""
        sm = StateMachine(redis, run_id + "-cancel")
        sm.initialize()
        sm.transition("planning")
        sm.transition("pending_contract_approval")
        sm.transition("teacher_training")
        sm.transition("training_eval_gate")
        sm.transition("pending_approval")
        sm.transition("cancelled")
        assert sm.current_state == "cancelled"


# ---------------------------------------------------------------------------
# stage_summaries
# ---------------------------------------------------------------------------

class TestStageSummaries:
    r"""Tests for the stage summaries list stored in Redis. Should be able to append summaries and retrieve the full list."""
    def test_append_and_get(self, sm):
        r"""Append a stage summary and retrieve it."""
        sm.append_stage_summary({"stage": "teacher_training", "status": "pass", "metrics": {"mAP50": 0.72}})
        summaries = sm.get_stage_summaries()
        assert len(summaries) == 1
        assert summaries[0]["stage"] == "teacher_training"

    def test_append_multiple(self, sm):
        r"""Append multiple stage summaries and retrieve the full list."""
        sm.append_stage_summary({"stage": "auto_labeling", "status": "pass"})
        sm.append_stage_summary({"stage": "teacher_training", "status": "pass"})
        assert len(sm.get_stage_summaries()) == 2

    def test_empty_initially(self, sm):
        r"""Before appending any summaries, get_stage_summaries() should return an empty list."""
        assert sm.get_stage_summaries() == []


# ---------------------------------------------------------------------------
# get_proposed_contract()
# ---------------------------------------------------------------------------

class TestGetProposedContract:
    r"""Tests for get_proposed_contract(). 
    Should return the contract dict if set, or None if not set or if JSON is invalid."""
    def test_returns_none_when_not_set(self, redis, run_id):
        r"""If no contract was set at initialize, get_proposed_contract() should return None."""
        m = StateMachine(redis, run_id + "-nocontract")
        m.initialize()
        assert m.get_proposed_contract() is None

    def test_returns_contract_when_set(self, redis, run_id, contract_dict):
        r"""If a contract dict was provided at initialize, get_proposed_contract() should return it."""
        m = StateMachine(redis, run_id + "-withcontract")
        m.initialize(contract=contract_dict)
        result = m.get_proposed_contract()
        assert result["id"] == "contract-abc"

    def test_handles_corrupt_json_gracefully(self, redis, run_id):
        r"""If the proposed_contract field contains invalid JSON, get_proposed_contract() should return None."""
        m = StateMachine(redis, run_id + "-corrupt")
        m.initialize()
        redis.hset(m._key, "proposed_contract", b"not valid json {{{")
        assert m.get_proposed_contract() is None

    def test_handles_empty_json_object(self, redis, run_id):
        r"""If the proposed_contract field is set to "{}", get_proposed_contract() should return None since {} is falsy."""
        m = StateMachine(redis, run_id + "-empty")
        m.initialize()
        redis.hset(m._key, "proposed_contract", b"{}")
        # {} is falsy after parse -- should return None
        assert m.get_proposed_contract() is None


# ---------------------------------------------------------------------------
# exists()
# ---------------------------------------------------------------------------

class TestExists:
    r"""Tests for StateMachine.exists(). Should return True if state exists for given run_id, False otherwise."""
    def test_true_after_initialize(self, redis, run_id):
        r"""After calling initialize() for a run_id, exists() should return True for that run_id."""
        m = StateMachine(redis, run_id + "-exists")
        m.initialize()
        assert StateMachine.exists(redis, run_id + "-exists") is True

    def test_false_before_initialize(self, redis):
        r"""Before calling initialize() for a run_id, exists() should return False for that run_id."""
        assert StateMachine.exists(redis, "never-initialized") is False
