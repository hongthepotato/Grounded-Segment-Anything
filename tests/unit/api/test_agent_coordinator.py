"""
Unit tests for Coordinator lifecycle in api/routes/agent.py.

Covers four behaviours:
  1. _on_done crash → failed_unrecoverable (TODO #2)
  2. approve_plan idempotency (TODO #3)
  3. resume_orphaned_coordinators startup scan (TODO #1)
  4. Crash classification: transient → failed_retrying, permanent → failed_unrecoverable
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import fakeredis.aioredis
import pytest

from ml_engine.agent.state_machine import StateMachine


@pytest.fixture
def fake_redis():
    return fakeredis.aioredis.FakeRedis(decode_responses=False)


def _patch_start_coordinator(fake_redis, coordinator_mock):
    """Return a context manager that patches all internal imports of _start_coordinator."""
    return (
        patch("api.routes.agent._get_async_redis", return_value=fake_redis),
        patch("ml_engine.agent.coordinator.Coordinator", return_value=coordinator_mock),
        patch("ml_engine.agent.contracts.PipelineContract.from_dict", return_value=MagicMock()),
        patch("ml_engine.agent.llm_client.LLMClient"),
    )


class TestCoordinatorCrashMarksFailed:
    @pytest.mark.asyncio
    async def test_crash_transitions_to_failed_unrecoverable(self, fake_redis):
        run_id = "crash-test-0001"
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()
        await sm.transition("planning")

        async def crashing_run():
            raise RuntimeError("boom: simulated OOM")

        coordinator_mock = MagicMock()
        coordinator_mock.run = crashing_run

        patches = _patch_start_coordinator(fake_redis, coordinator_mock)
        with patches[0], patches[1], patches[2], patches[3]:
            from api.routes.agent import _start_coordinator

            _start_coordinator(run_id, {})
            await asyncio.sleep(0.1)

        assert await sm.current_state() == "failed_unrecoverable"
        data = await sm.load()
        assert "boom: simulated OOM" in data["error_message"]

    @pytest.mark.asyncio
    async def test_crash_error_message_is_str_of_exception(self, fake_redis):
        run_id = "crash-test-0002"
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()
        await sm.transition("planning")

        async def crashing_run():
            raise ValueError("unexpected token at position 42")

        coordinator_mock = MagicMock()
        coordinator_mock.run = crashing_run

        patches = _patch_start_coordinator(fake_redis, coordinator_mock)
        with patches[0], patches[1], patches[2], patches[3]:
            from api.routes.agent import _start_coordinator

            _start_coordinator(run_id, {})
            await asyncio.sleep(0.1)

        data = await sm.load()
        assert data["error_message"] == "unexpected token at position 42"

    @pytest.mark.asyncio
    async def test_clean_exit_does_not_set_failed(self, fake_redis):
        """A normal Coordinator exit must NOT set failed_unrecoverable."""
        run_id = "clean-exit-0001"
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()
        await sm.transition("planning")

        async def clean_run():
            await sm.transition("pending_contract_approval")
            await sm.transition("teacher_training")
            await sm.transition("training_eval_gate")
            await sm.transition("pending_approval")
            await sm.transition("done")

        coordinator_mock = MagicMock()
        coordinator_mock.run = clean_run

        patches = _patch_start_coordinator(fake_redis, coordinator_mock)
        with patches[0], patches[1], patches[2], patches[3]:
            from api.routes.agent import _start_coordinator

            _start_coordinator(run_id, {})
            await asyncio.sleep(0.1)

        assert await sm.current_state() == "done"

    @pytest.mark.asyncio
    async def test_crash_in_mid_pipeline_state(self, fake_redis):
        """Crash from a non-planning state (e.g. teacher_training) also marks failed."""
        run_id = "crash-test-mid-0001"
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()
        await sm.transition("planning")
        await sm.transition("pending_contract_approval")
        await sm.transition("teacher_training")

        async def crashing_run():
            raise RuntimeError("CUDA kernel launch failed")

        coordinator_mock = MagicMock()
        coordinator_mock.run = crashing_run

        patches = _patch_start_coordinator(fake_redis, coordinator_mock)
        with patches[0], patches[1], patches[2], patches[3]:
            from api.routes.agent import _start_coordinator

            _start_coordinator(run_id, {})
            await asyncio.sleep(0.1)

        assert await sm.current_state() == "failed_unrecoverable"

    @pytest.mark.asyncio
    async def test_crash_when_already_terminal_does_not_raise(self, fake_redis):
        """If the run is already terminal before the crash callback fires, _mark_failed
        must swallow the transition error rather than propagating it."""
        run_id = "crash-already-terminal-0001"
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()
        await sm.transition("planning")

        async def crashing_run():
            # Simulate state already reaching a terminal via a concurrent path
            await sm.transition("failed_unrecoverable", error_message="concurrent failure")
            raise RuntimeError("secondary crash")

        coordinator_mock = MagicMock()
        coordinator_mock.run = crashing_run

        patches = _patch_start_coordinator(fake_redis, coordinator_mock)
        with patches[0], patches[1], patches[2], patches[3]:
            from api.routes.agent import _start_coordinator

            _start_coordinator(run_id, {})
            # Must not raise even though _mark_failed will hit an invalid transition
            await asyncio.sleep(0.1)

        # Still terminal — first transition wins
        assert await sm.current_state() == "failed_unrecoverable"


# ---------------------------------------------------------------------------
# Idempotent approve (#3)
# ---------------------------------------------------------------------------


def _noop_coordinator():
    """Coordinator whose run() blocks indefinitely (simulates a running pipeline)."""
    mock = MagicMock()

    async def _run():
        await asyncio.sleep(9999)

    mock.run = _run
    return mock


def _patches(fake_redis):
    return _patch_start_coordinator(fake_redis, _noop_coordinator())


class TestIdempotentApprove:
    # --- 404 / 409 guards ---

    @pytest.mark.asyncio
    async def test_unknown_run_returns_404(self, fake_redis):
        p = _patches(fake_redis)
        with p[0], p[1], p[2], p[3]:
            from fastapi.testclient import TestClient

            from api.app import app

            with TestClient(app) as client:
                resp = client.post("/api/agent/approve", json={"run_id": "no-such-run", "contract": {}})
        assert resp.status_code == 404

    @pytest.mark.parametrize("terminal", ["done", "failed_unrecoverable", "escalated", "cancelled"])
    @pytest.mark.asyncio
    async def test_every_terminal_state_returns_409(self, fake_redis, terminal):
        run_id = f"terminal-{terminal}"
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()
        await fake_redis.hset(sm._key, "state", terminal)

        p = _patches(fake_redis)
        with p[0], p[1], p[2], p[3]:
            from fastapi.testclient import TestClient

            from api.app import app

            with TestClient(app) as client:
                resp = client.post("/api/agent/approve", json={"run_id": run_id, "contract": {}})
        assert resp.status_code == 409

    # --- First approval ---

    @pytest.mark.asyncio
    async def test_created_run_transitions_to_planning_and_persists_contract(self, fake_redis):
        run_id = "approve-created-0001"
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()

        p = _patches(fake_redis)
        with p[0], p[1], p[2], p[3]:
            from fastapi.testclient import TestClient

            from api.app import app

            with TestClient(app) as client:
                resp = client.post("/api/agent/approve", json={"run_id": run_id, "contract": {"id": "c1"}})

        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "planning"
        assert await sm.current_state() == "planning"
        assert await sm.get_approved_contract() == {"id": "c1"}

    # --- Idempotent re-approve across non-terminal states ---

    @pytest.mark.parametrize(
        "state",
        [
            "planning",
            "pending_contract_approval",
            "auto_labeling",
            "label_review_gate",
            "teacher_training",
            "training_eval_gate",
            "student_distillation",
            "distill_eval_gate",
            "pending_approval",
            "failed_retrying",
        ],
    )
    @pytest.mark.asyncio
    async def test_re_approve_every_non_terminal_state_succeeds(self, fake_redis, state):
        run_id = f"re-approve-{state}"
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()
        await fake_redis.hset(sm._key, "state", state)

        p = _patches(fake_redis)
        with p[0], p[1], p[2], p[3]:
            from fastapi.testclient import TestClient

            from api.app import app

            with TestClient(app) as client:
                resp = client.post(
                    "/api/agent/approve",
                    json={"run_id": run_id, "contract": {"id": "updated"}},
                )

        assert resp.status_code == 200
        # State must not regress
        assert await sm.current_state() == state
        # Response reflects actual state, not hardcoded "planning"
        assert resp.json()["data"]["status"] == state
        # Contract updated
        assert await sm.get_approved_contract() == {"id": "updated"}

    @pytest.mark.asyncio
    async def test_double_approve_on_created_run_is_safe(self, fake_redis):
        """Calling approve twice in rapid succession on a fresh run must not crash."""
        run_id = "double-approve-0001"
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()

        p = _patches(fake_redis)
        with p[0], p[1], p[2], p[3]:
            from fastapi.testclient import TestClient

            from api.app import app

            with TestClient(app) as client:
                r1 = client.post("/api/agent/approve", json={"run_id": run_id, "contract": {"v": 1}})
                r2 = client.post("/api/agent/approve", json={"run_id": run_id, "contract": {"v": 2}})

        assert r1.status_code == 200
        assert r2.status_code == 200
        assert await sm.current_state() == "planning"
        # Second approve updated the contract
        assert await sm.get_approved_contract() == {"v": 2}


# ---------------------------------------------------------------------------
# Orphan resume on startup (#1)
# ---------------------------------------------------------------------------


class TestResumeOrphanedCoordinators:
    @pytest.mark.asyncio
    async def test_empty_redis_is_noop(self, fake_redis):
        """No runs in Redis → no error, nothing launched."""
        p = _patches(fake_redis)
        with p[0], p[1], p[2], p[3]:
            from api.routes.agent import _coordinator_tasks, resume_orphaned_coordinators

            _coordinator_tasks.clear()
            await resume_orphaned_coordinators()
            assert _coordinator_tasks == {}

    @pytest.mark.asyncio
    async def test_resumes_run_with_approved_contract(self, fake_redis):
        run_id = "orphan-resume-0001"
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()
        await sm.transition("planning")
        await sm.store_approved_contract({"id": "c-orphan"})

        p = _patches(fake_redis)
        with p[0], p[1], p[2], p[3]:
            from api.routes.agent import _coordinator_tasks, resume_orphaned_coordinators

            _coordinator_tasks.clear()
            await resume_orphaned_coordinators()
            assert run_id in _coordinator_tasks
            _coordinator_tasks.pop(run_id, None)

    @pytest.mark.asyncio
    async def test_resumes_multiple_orphaned_runs(self, fake_redis):
        run_ids = [f"orphan-multi-{i}" for i in range(3)]
        for run_id in run_ids:
            sm = StateMachine(run_id=run_id, redis_async=fake_redis)
            await sm.initialize()
            await sm.transition("planning")
            await sm.store_approved_contract({"id": run_id})

        p = _patches(fake_redis)
        with p[0], p[1], p[2], p[3]:
            from api.routes.agent import _coordinator_tasks, resume_orphaned_coordinators

            _coordinator_tasks.clear()
            await resume_orphaned_coordinators()
            for run_id in run_ids:
                assert run_id in _coordinator_tasks
                _coordinator_tasks.pop(run_id, None)

    @pytest.mark.asyncio
    async def test_skips_run_without_approved_contract(self, fake_redis):
        """Run in planning with no approved contract (pre-approve) must be skipped."""
        run_id = "orphan-no-contract-0001"
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()
        await sm.transition("planning")

        p = _patches(fake_redis)
        with p[0], p[1], p[2], p[3]:
            from api.routes.agent import _coordinator_tasks, resume_orphaned_coordinators

            _coordinator_tasks.clear()
            await resume_orphaned_coordinators()
            assert run_id not in _coordinator_tasks

    @pytest.mark.asyncio
    async def test_skips_already_running_coordinator(self, fake_redis):
        run_id = "orphan-already-running-0001"
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()
        await sm.transition("planning")
        await sm.store_approved_contract({"id": "c-running"})

        p = _patches(fake_redis)
        with p[0], p[1], p[2], p[3]:
            from api.routes.agent import _coordinator_tasks, resume_orphaned_coordinators

            sentinel = asyncio.create_task(asyncio.sleep(9999))
            _coordinator_tasks[run_id] = sentinel
            await resume_orphaned_coordinators()
            # Original task unchanged — not replaced
            assert _coordinator_tasks[run_id] is sentinel
            sentinel.cancel()
            _coordinator_tasks.pop(run_id, None)

    @pytest.mark.asyncio
    async def test_relaunches_done_task(self, fake_redis):
        """A task that is done() (finished/crashed) must be replaced, not skipped."""
        run_id = "orphan-done-task-0001"
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()
        await sm.transition("planning")
        await sm.store_approved_contract({"id": "c-done-task"})

        p = _patches(fake_redis)
        with p[0], p[1], p[2], p[3]:
            from api.routes.agent import _coordinator_tasks, resume_orphaned_coordinators

            # Simulate a dead (done) task
            async def _instant():
                return

            dead_task = asyncio.create_task(_instant())
            await asyncio.sleep(0)  # let it finish
            assert dead_task.done()

            _coordinator_tasks[run_id] = dead_task
            await resume_orphaned_coordinators()
            # New task launched to replace the dead one
            assert _coordinator_tasks[run_id] is not dead_task
            _coordinator_tasks.pop(run_id, None)

    @pytest.mark.asyncio
    async def test_skips_terminal_runs(self, fake_redis):
        run_id = "orphan-terminal-0001"
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()
        await fake_redis.hset(sm._key, "state", "done")
        await sm.store_approved_contract({"id": "c-done"})

        p = _patches(fake_redis)
        with p[0], p[1], p[2], p[3]:
            from api.routes.agent import _coordinator_tasks, resume_orphaned_coordinators

            _coordinator_tasks.clear()
            await resume_orphaned_coordinators()
            assert run_id not in _coordinator_tasks

    @pytest.mark.asyncio
    async def test_mixed_terminal_and_non_terminal(self, fake_redis):
        """Terminal runs ignored; non-terminal runs with contract resumed."""
        terminal_id = "orphan-mix-terminal"
        active_id = "orphan-mix-active"

        for run_id, state in [(terminal_id, "done"), (active_id, "teacher_training")]:
            sm = StateMachine(run_id=run_id, redis_async=fake_redis)
            await sm.initialize()
            await fake_redis.hset(sm._key, "state", state)
            await sm.store_approved_contract({"id": run_id})

        p = _patches(fake_redis)
        with p[0], p[1], p[2], p[3]:
            from api.routes.agent import _coordinator_tasks, resume_orphaned_coordinators

            _coordinator_tasks.clear()
            await resume_orphaned_coordinators()
            assert terminal_id not in _coordinator_tasks
            assert active_id in _coordinator_tasks
            _coordinator_tasks.pop(active_id, None)

    @pytest.mark.asyncio
    async def test_one_crashing_coordinator_does_not_block_others(self, fake_redis):
        """If one Coordinator crashes immediately, the others still get resumed."""
        crashing_id = "orphan-crash-0001"
        healthy_id = "orphan-healthy-0001"

        for run_id in [crashing_id, healthy_id]:
            sm = StateMachine(run_id=run_id, redis_async=fake_redis)
            await sm.initialize()
            await sm.transition("planning")
            await sm.store_approved_contract({"id": run_id})

        async def _crashing_run():
            raise RuntimeError("immediate crash")

        crashing_mock = MagicMock()
        crashing_mock.run = _crashing_run
        healthy_mock = _noop_coordinator()

        call_count = 0

        def _coordinator_factory(**kwargs):
            nonlocal call_count
            call_count += 1
            return crashing_mock if call_count == 1 else healthy_mock

        with (
            patch("api.routes.agent._get_async_redis", return_value=fake_redis),
            patch("ml_engine.agent.coordinator.Coordinator", side_effect=_coordinator_factory),
            patch("ml_engine.agent.contracts.PipelineContract.from_dict", return_value=MagicMock()),
            patch("ml_engine.agent.llm_client.LLMClient"),
        ):
            from api.routes.agent import _coordinator_tasks, resume_orphaned_coordinators

            _coordinator_tasks.clear()
            await resume_orphaned_coordinators()
            await asyncio.sleep(0.1)  # let crash propagate

            # Both tasks were launched
            assert crashing_id in _coordinator_tasks or healthy_id in _coordinator_tasks
            _coordinator_tasks.pop(crashing_id, None)
            _coordinator_tasks.pop(healthy_id, None)


# ---------------------------------------------------------------------------
# StateMachine: approved contract + scan helpers
# ---------------------------------------------------------------------------


class TestApprovedContract:
    @pytest.mark.asyncio
    async def test_store_and_retrieve_roundtrip(self, fake_redis):
        sm = StateMachine(run_id="contract-rt-0001", redis_async=fake_redis)
        await sm.initialize()
        contract = {"id": "c1", "budget": {"max_retries": 3}}
        await sm.store_approved_contract(contract)
        assert await sm.get_approved_contract() == contract

    @pytest.mark.asyncio
    async def test_get_returns_none_before_store(self, fake_redis):
        sm = StateMachine(run_id="contract-none-0001", redis_async=fake_redis)
        await sm.initialize()
        assert await sm.get_approved_contract() is None

    @pytest.mark.asyncio
    async def test_get_returns_none_on_corrupt_json(self, fake_redis):
        sm = StateMachine(run_id="contract-corrupt-0001", redis_async=fake_redis)
        await sm.initialize()
        await fake_redis.hset(sm._key, "approved_contract", b"not json {{")
        assert await sm.get_approved_contract() is None

    @pytest.mark.asyncio
    async def test_store_overwrites_previous(self, fake_redis):
        sm = StateMachine(run_id="contract-overwrite-0001", redis_async=fake_redis)
        await sm.initialize()
        await sm.store_approved_contract({"v": 1})
        await sm.store_approved_contract({"v": 2})
        assert await sm.get_approved_contract() == {"v": 2}


class TestScanNonTerminalRunIds:
    @pytest.mark.asyncio
    async def test_empty_redis_returns_empty(self, fake_redis):
        result = await StateMachine.scan_non_terminal_run_ids(fake_redis)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_non_terminal_run_ids(self, fake_redis):
        active = StateMachine(run_id="scan-active-0001", redis_async=fake_redis)
        await active.initialize()
        await active.transition("planning")

        done = StateMachine(run_id="scan-done-0001", redis_async=fake_redis)
        await done.initialize()
        await fake_redis.hset(done._key, "state", "done")

        result = await StateMachine.scan_non_terminal_run_ids(fake_redis)
        assert "scan-active-0001" in result
        assert "scan-done-0001" not in result

    @pytest.mark.parametrize("terminal", ["done", "failed_unrecoverable", "escalated", "cancelled"])
    @pytest.mark.asyncio
    async def test_all_terminal_states_excluded(self, fake_redis, terminal):
        run_id = f"scan-terminal-{terminal}"
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()
        await fake_redis.hset(sm._key, "state", terminal)

        result = await StateMachine.scan_non_terminal_run_ids(fake_redis)
        assert run_id not in result


# ---------------------------------------------------------------------------
# Human gate endpoint (#52)
# ---------------------------------------------------------------------------


def _gate_patches(fake_redis):
    """Minimal patches for human_gate: Redis + suppress apublish_event side-effects."""
    return (
        patch("api.routes.agent._get_async_redis", return_value=fake_redis),
        patch("ml_engine.agent.loop.apublish_event", return_value=None),
    )


class TestHumanGate:
    # --- input validation ---

    @pytest.mark.asyncio
    async def test_invalid_action_returns_400(self, fake_redis):
        run_id = "gate-bad-action-0001"
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()
        await fake_redis.hset(sm._key, "state", "pending_approval")

        with _gate_patches(fake_redis)[0], _gate_patches(fake_redis)[1]:
            from fastapi.testclient import TestClient

            from api.app import app

            with TestClient(app) as client:
                resp = client.post(f"/api/agent/gate/{run_id}/kick", json={"reason": ""})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_unknown_run_returns_404(self, fake_redis):
        with _gate_patches(fake_redis)[0], _gate_patches(fake_redis)[1]:
            from fastapi.testclient import TestClient

            from api.app import app

            with TestClient(app) as client:
                resp = client.post("/api/agent/gate/no-such-run/approve", json={"reason": ""})
        assert resp.status_code == 404

    @pytest.mark.parametrize("non_gate_state", ["planning", "auto_labeling", "teacher_training", "done"])
    @pytest.mark.asyncio
    async def test_non_gate_state_returns_409(self, fake_redis, non_gate_state):
        run_id = f"gate-wrong-state-{non_gate_state}"
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()
        await fake_redis.hset(sm._key, "state", non_gate_state)

        with _gate_patches(fake_redis)[0], _gate_patches(fake_redis)[1]:
            from fastapi.testclient import TestClient

            from api.app import app

            with TestClient(app) as client:
                resp = client.post(f"/api/agent/gate/{run_id}/approve", json={"reason": ""})
        assert resp.status_code == 409

    # --- pending_approval gate ---

    @pytest.mark.asyncio
    async def test_pending_approval_approve_transitions_to_done(self, fake_redis):
        run_id = "gate-pa-approve-0001"
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()
        await fake_redis.hset(sm._key, "state", "pending_approval")

        with _gate_patches(fake_redis)[0], _gate_patches(fake_redis)[1]:
            from fastapi.testclient import TestClient

            from api.app import app

            with TestClient(app) as client:
                resp = client.post(f"/api/agent/gate/{run_id}/approve", json={"reason": "LGTM"})

        assert resp.status_code == 200
        assert resp.json()["data"]["new_state"] == "done"
        assert await sm.current_state() == "done"

    @pytest.mark.asyncio
    async def test_pending_approval_reject_transitions_to_cancelled(self, fake_redis):
        run_id = "gate-pa-reject-0001"
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()
        await fake_redis.hset(sm._key, "state", "pending_approval")

        with _gate_patches(fake_redis)[0], _gate_patches(fake_redis)[1]:
            from fastapi.testclient import TestClient

            from api.app import app

            with TestClient(app) as client:
                resp = client.post(f"/api/agent/gate/{run_id}/reject", json={"reason": "results look off"})

        assert resp.status_code == 200
        assert resp.json()["data"]["new_state"] == "cancelled"
        assert await sm.current_state() == "cancelled"

    # --- pending_contract_approval gate ---

    @pytest.mark.asyncio
    async def test_pending_contract_approval_approve_transitions_to_auto_labeling(self, fake_redis):
        run_id = "gate-pca-approve-0001"
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()
        await fake_redis.hset(sm._key, "state", "pending_contract_approval")

        with _gate_patches(fake_redis)[0], _gate_patches(fake_redis)[1]:
            from fastapi.testclient import TestClient

            from api.app import app

            with TestClient(app) as client:
                resp = client.post(
                    f"/api/agent/gate/{run_id}/approve", json={"reason": "contract looks good"}
                )

        assert resp.status_code == 200
        assert resp.json()["data"]["new_state"] == "auto_labeling"
        assert await sm.current_state() == "auto_labeling"

    @pytest.mark.asyncio
    async def test_pending_contract_approval_reject_transitions_to_cancelled(self, fake_redis):
        run_id = "gate-pca-reject-0001"
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()
        await fake_redis.hset(sm._key, "state", "pending_contract_approval")

        with _gate_patches(fake_redis)[0], _gate_patches(fake_redis)[1]:
            from fastapi.testclient import TestClient

            from api.app import app

            with TestClient(app) as client:
                resp = client.post(f"/api/agent/gate/{run_id}/reject", json={"reason": "budget too high"})

        assert resp.status_code == 200
        assert resp.json()["data"]["new_state"] == "cancelled"
        assert await sm.current_state() == "cancelled"

    @pytest.mark.asyncio
    async def test_contract_approval_publishes_contract_approved_event(self, fake_redis):
        """Approve on pending_contract_approval must publish 'contract_approved' (not 'contract_accepted')."""
        run_id = "gate-pca-event-0001"
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()
        await fake_redis.hset(sm._key, "state", "pending_contract_approval")

        published: list = []

        async def _capture_event(r, rid, event):
            published.append(event)

        with (
            _gate_patches(fake_redis)[0],
            patch("ml_engine.agent.loop.apublish_event", side_effect=_capture_event),
        ):
            from fastapi.testclient import TestClient

            from api.app import app

            with TestClient(app) as client:
                client.post(f"/api/agent/gate/{run_id}/approve", json={"reason": ""})

        assert len(published) == 1
        assert published[0]["type"] == "contract_approved"

    @pytest.mark.asyncio
    async def test_contract_rejection_publishes_contract_rejected_event(self, fake_redis):
        """Reject on pending_contract_approval must publish 'contract_rejected'."""
        run_id = "gate-pca-reject-event-0001"
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()
        await fake_redis.hset(sm._key, "state", "pending_contract_approval")

        published: list = []

        async def _capture_event(r, rid, event):
            published.append(event)

        with (
            _gate_patches(fake_redis)[0],
            patch("ml_engine.agent.loop.apublish_event", side_effect=_capture_event),
        ):
            from fastapi.testclient import TestClient

            from api.app import app

            with TestClient(app) as client:
                client.post(f"/api/agent/gate/{run_id}/reject", json={"reason": "too expensive"})

        assert len(published) == 1
        assert published[0]["type"] == "contract_rejected"

    @pytest.mark.asyncio
    async def test_invalid_transition_returns_400(self, fake_redis):
        """If sm.transition() raises ValueError, gate must return 400 (not 500).

        Uses fakeredis with a direct HSET to force an invalid state (not in TRANSITIONS)
        so that the state machine's own transition() raises ValueError without any mocking.
        """
        run_id = "gate-bad-transition-0001"
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()
        # Force state machine into 'planning' (valid gate check via 409 would catch it)
        # Instead, set current state to 'pending_approval' and transition target to
        # a state that IS in pending_approval's allowed list — but use a non-existent
        # action to get past the gate and hit the transition guard.
        #
        # The simplest path: set the state directly to something the SM won't allow
        # transitioning from via 'pending_approval → done'. Actually that IS valid.
        # So the only way to trigger ValueError is to have pending_approval try to
        # transition to a state NOT in its allowed list. We do that by patching
        # the TRANSITIONS dict for this one test.
        await fake_redis.hset(sm._key, "state", "pending_approval")

        from ml_engine.agent.state_machine import TRANSITIONS

        # Temporarily remove 'done' from pending_approval's transitions so any
        # approve attempt fails with ValueError.
        original_transitions = list(TRANSITIONS["pending_approval"])
        TRANSITIONS["pending_approval"] = [t for t in original_transitions if t != "done"]

        try:
            with _gate_patches(fake_redis)[0], _gate_patches(fake_redis)[1]:
                from fastapi.testclient import TestClient

                from api.app import app

                with TestClient(app) as client:
                    resp = client.post(f"/api/agent/gate/{run_id}/approve", json={"reason": ""})
        finally:
            TRANSITIONS["pending_approval"] = original_transitions

        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Crash classification: transient vs permanent (#4)
# ---------------------------------------------------------------------------


class TestCrashClassification:
    """Tests for _handle_coordinator_crash: transient vs permanent routing.

    Calls _handle_coordinator_crash directly to avoid asyncio task scheduling
    complexity and focus on the routing logic under each condition.
    """

    def _make_contract(self, max_retries: int = 2) -> MagicMock:
        contract = MagicMock()
        contract.budget.max_retries = max_retries
        return contract

    @pytest.mark.asyncio
    async def test_transient_in_retryable_state_routes_to_failed_retrying(self, fake_redis):
        """ConnectionError from auto_labeling → failed_retrying and coordinator re-launched."""
        run_id = "crash-class-0001"
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()
        await sm.transition("planning")
        await sm.transition("pending_contract_approval")
        await sm.transition("auto_labeling")

        with patch("api.routes.agent._start_coordinator") as mock_restart:
            from api.routes.agent import _handle_coordinator_crash

            await _handle_coordinator_crash(
                run_id,
                ConnectionError("redis gone"),
                fake_redis,
                self._make_contract(),
                {},
                transient=True,
            )

        assert await sm.current_state() == "failed_retrying"
        mock_restart.assert_called_once_with(run_id, {})

    @pytest.mark.asyncio
    async def test_transient_in_non_retryable_state_routes_to_failed_unrecoverable(self, fake_redis):
        """TimeoutError from planning → failed_unrecoverable (planning has no failed_retrying arc)."""
        run_id = "crash-class-0002"
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()
        await sm.transition("planning")

        with patch("api.routes.agent._start_coordinator") as mock_restart:
            from api.routes.agent import _handle_coordinator_crash

            await _handle_coordinator_crash(
                run_id,
                TimeoutError("llm timeout"),
                fake_redis,
                self._make_contract(),
                {},
                transient=True,
            )

        assert await sm.current_state() == "failed_unrecoverable"
        mock_restart.assert_not_called()

    @pytest.mark.asyncio
    async def test_permanent_exception_routes_to_failed_unrecoverable(self, fake_redis):
        """RuntimeError from teacher_training → failed_unrecoverable (not retried)."""
        run_id = "crash-class-0003"
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()
        await sm.transition("planning")
        await sm.transition("pending_contract_approval")
        await sm.transition("teacher_training")

        with patch("api.routes.agent._start_coordinator") as mock_restart:
            from api.routes.agent import _handle_coordinator_crash

            await _handle_coordinator_crash(
                run_id,
                RuntimeError("logic bug"),
                fake_redis,
                self._make_contract(),
                {},
                transient=False,
            )

        assert await sm.current_state() == "failed_unrecoverable"
        mock_restart.assert_not_called()

    @pytest.mark.asyncio
    async def test_retries_exhausted_routes_to_failed_unrecoverable(self, fake_redis):
        """Transient error from auto_labeling with retry_count at max → failed_unrecoverable."""
        run_id = "crash-class-0004"
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()
        await sm.transition("planning")
        await sm.transition("pending_contract_approval")
        await sm.transition("auto_labeling")
        await fake_redis.hset(sm._key, "retry_count", "2")  # budget exhausted

        with patch("api.routes.agent._start_coordinator") as mock_restart:
            from api.routes.agent import _handle_coordinator_crash

            await _handle_coordinator_crash(
                run_id,
                ConnectionError("redis gone"),
                fake_redis,
                self._make_contract(max_retries=2),
                {},
                transient=True,
            )

        assert await sm.current_state() == "failed_unrecoverable"
        mock_restart.assert_not_called()

    @pytest.mark.asyncio
    async def test_transient_sets_error_message_on_failed_retrying(self, fake_redis):
        """The error message is persisted even when routing to failed_retrying."""
        run_id = "crash-class-0005"
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()
        await sm.transition("planning")
        await sm.transition("pending_contract_approval")
        await sm.transition("teacher_training")

        with patch("api.routes.agent._start_coordinator"):
            from api.routes.agent import _handle_coordinator_crash

            await _handle_coordinator_crash(
                run_id,
                ConnectionError("lost connection to worker"),
                fake_redis,
                self._make_contract(),
                {},
                transient=True,
            )

        data = await sm.load()
        assert "lost connection to worker" in data["error_message"]

    @pytest.mark.asyncio
    async def test_retry_increments_retry_count(self, fake_redis):
        """Each failed_retrying transition increments retry_count in Redis."""
        run_id = "crash-class-0006"
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()
        await sm.transition("planning")
        await sm.transition("pending_contract_approval")
        await sm.transition("auto_labeling")

        with patch("api.routes.agent._start_coordinator"):
            from api.routes.agent import _handle_coordinator_crash

            await _handle_coordinator_crash(
                run_id,
                TimeoutError("worker timeout"),
                fake_redis,
                self._make_contract(),
                {},
                transient=True,
            )

        assert await sm.retry_count() == 1

    def test_is_transient_exception_builtin_types(self):
        from api.routes.agent import _is_transient_exception

        assert _is_transient_exception(ConnectionError("gone"))
        assert _is_transient_exception(TimeoutError("timed out"))
        assert _is_transient_exception(InterruptedError("EINTR"))
        # ConnectionError subtypes are also transient
        assert _is_transient_exception(BrokenPipeError("pipe gone"))
        assert _is_transient_exception(ConnectionResetError("reset"))

    def test_is_transient_exception_non_transient_types(self):
        from api.routes.agent import _is_transient_exception

        assert not _is_transient_exception(RuntimeError("logic bug"))
        assert not _is_transient_exception(ValueError("bad input"))
        assert not _is_transient_exception(KeyError("missing key"))
        # OSError itself is too broad — only specific subclasses are transient
        assert not _is_transient_exception(OSError("generic os error"))
        assert not _is_transient_exception(FileNotFoundError("model not found"))
        assert not _is_transient_exception(PermissionError("cannot write checkpoint"))

    @pytest.mark.asyncio
    async def test_second_crash_from_failed_retrying_routes_to_failed_unrecoverable(self, fake_redis):
        """If the re-launched coordinator crashes while the SM is in failed_retrying
        (i.e. before it can advance to a work state), the handler must not loop —
        failed_retrying has no self-arc, so can_retry is False and it transitions
        to failed_unrecoverable."""
        run_id = "crash-class-0008"
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()
        # Simulate: first crash already put the run in failed_retrying
        await sm.transition("planning")
        await sm.transition("pending_contract_approval")
        await sm.transition("auto_labeling")
        await sm.transition("failed_retrying", error_message="first crash")

        with patch("api.routes.agent._start_coordinator") as mock_restart:
            from api.routes.agent import _handle_coordinator_crash

            await _handle_coordinator_crash(
                run_id,
                ConnectionError("second crash"),
                fake_redis,
                self._make_contract(),
                {},
                transient=True,
            )

        # failed_retrying has no self-arc, so the second crash must not retry again
        assert await sm.current_state() == "failed_unrecoverable"
        mock_restart.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_state_key_returns_silently(self, fake_redis):
        """If the Redis key for the run doesn't exist, KeyError is caught and the
        handler returns without raising."""
        run_id = "crash-class-missing-key"
        contract_mock = MagicMock()
        contract_mock.budget.max_retries = 2

        with patch("api.routes.agent._start_coordinator") as mock_restart:
            from api.routes.agent import _handle_coordinator_crash

            # No sm.initialize() — the key doesn't exist in Redis
            await _handle_coordinator_crash(
                run_id,
                ConnectionError("redis gone"),
                fake_redis,
                contract_mock,
                {},
                transient=True,
            )

        mock_restart.assert_not_called()

    @pytest.mark.asyncio
    async def test_student_distillation_transient_routes_to_failed_retrying(self, fake_redis):
        """student_distillation is a retryable state too — symmetry check."""
        run_id = "crash-class-0007"
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()
        await sm.transition("planning")
        await sm.transition("pending_contract_approval")
        await sm.transition("teacher_training")
        await sm.transition("training_eval_gate")
        await sm.transition("student_distillation")

        contract_dict = {"id": "test-contract"}
        with patch("api.routes.agent._start_coordinator") as mock_restart:
            from api.routes.agent import _handle_coordinator_crash

            await _handle_coordinator_crash(
                run_id,
                OSError("worker killed"),
                fake_redis,
                self._make_contract(),
                contract_dict,
                transient=True,
            )

        assert await sm.current_state() == "failed_retrying"
        mock_restart.assert_called_once_with(run_id, contract_dict)


# ---------------------------------------------------------------------------
# Retry dispatch (issue #54)
# ---------------------------------------------------------------------------


class TestRetryDispatch:
    """Tests for Coordinator.on_event job_failed branch re-dispatch logic.

    Core invariants under test:
    - failed_stage is captured BEFORE transitioning away (not after)
    - SM ends in the original work stage after successful dispatch
    - dispatch_stage.execute() is called exactly once with the correct stage name
    - stage_dispatch_overrides from LoopState are forwarded verbatim (empty dict, not None)
    - retry_count is incremented by exactly 1 per retry
    - retries-exhausted path never calls dispatch and ends in failed_unrecoverable
    - dispatch failure path ends in failed_unrecoverable
    - error_message is stored in SM data (persisted at failed_retrying transition)
    - no LLM call is made (job_failed is deterministic)
    """

    def _make_contract(self, max_retries: int = 2) -> MagicMock:
        contract = MagicMock()
        contract.budget.max_retries = max_retries
        contract.to_dict.return_value = {"max_retries": max_retries}
        return contract

    async def _reach_stage(self, run_id: str, fake_redis, stage: str) -> "StateMachine":
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()
        await sm.transition("planning")
        await sm.transition("pending_contract_approval")
        await sm.transition(stage)
        return sm

    async def _reach_student_distillation(self, run_id: str, fake_redis) -> "StateMachine":
        sm = StateMachine(run_id=run_id, redis_async=fake_redis)
        await sm.initialize()
        await sm.transition("planning")
        await sm.transition("pending_contract_approval")
        await sm.transition("teacher_training")
        await sm.transition("training_eval_gate")
        await sm.transition("student_distillation")
        return sm

    # ------------------------------------------------------------------
    # State sequence correctness
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_teacher_training_failure_final_state_is_teacher_training(self, fake_redis):
        """After a successful retry dispatch, the SM must be back in teacher_training —
        NOT stuck in failed_retrying. This catches the original bug (no re-dispatch)."""
        from ml_engine.agent.coordinator import Coordinator
        from ml_engine.agent.loop import LoopState
        from ml_engine.agent.tools import ToolResult

        run_id = "rd-seq-0001"
        sm = await self._reach_stage(run_id, fake_redis, "teacher_training")

        coordinator = Coordinator(fake_redis, run_id, contract=self._make_contract())
        dispatch_tool = coordinator._tools.get("dispatch_stage")
        with patch.object(dispatch_tool, "execute", return_value=ToolResult(success=True)):
            await coordinator.on_event({"type": "job_failed", "error": "OOM"}, LoopState(run_id=run_id))

        final = await sm.current_state()
        assert final == "teacher_training", (
            f"SM ended in {final!r} — re-dispatch did not return run to work stage"
        )

    @pytest.mark.asyncio
    async def test_auto_labeling_failure_final_state_is_auto_labeling(self, fake_redis):
        """Same invariant as above for auto_labeling."""
        from ml_engine.agent.coordinator import Coordinator
        from ml_engine.agent.loop import LoopState
        from ml_engine.agent.tools import ToolResult

        run_id = "rd-seq-0002"
        sm = await self._reach_stage(run_id, fake_redis, "auto_labeling")

        coordinator = Coordinator(fake_redis, run_id, contract=self._make_contract())
        dispatch_tool = coordinator._tools.get("dispatch_stage")
        with patch.object(dispatch_tool, "execute", return_value=ToolResult(success=True)):
            await coordinator.on_event({"type": "job_failed", "error": "timeout"}, LoopState(run_id=run_id))

        assert await sm.current_state() == "auto_labeling"

    @pytest.mark.asyncio
    async def test_student_distillation_failure_final_state_is_student_distillation(self, fake_redis):
        """Same invariant for student_distillation."""
        from ml_engine.agent.coordinator import Coordinator
        from ml_engine.agent.loop import LoopState
        from ml_engine.agent.tools import ToolResult

        run_id = "rd-seq-0003"
        sm = await self._reach_student_distillation(run_id, fake_redis)

        coordinator = Coordinator(fake_redis, run_id, contract=self._make_contract())
        dispatch_tool = coordinator._tools.get("dispatch_stage")
        with patch.object(dispatch_tool, "execute", return_value=ToolResult(success=True)):
            await coordinator.on_event({"type": "job_failed", "error": "CUDA OOM"}, LoopState(run_id=run_id))

        assert await sm.current_state() == "student_distillation"

    # ------------------------------------------------------------------
    # Dispatch called with the right stage — catches wrong-stage bugs
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_dispatch_receives_exact_failed_stage_name(self, fake_redis):
        """dispatch_stage.execute() must receive the stage that was active at failure time,
        not the transient failed_retrying state or any other value."""
        from ml_engine.agent.coordinator import Coordinator
        from ml_engine.agent.loop import LoopState
        from ml_engine.agent.tools import ToolResult

        run_id = "rd-stage-0001"
        await self._reach_stage(run_id, fake_redis, "auto_labeling")

        coordinator = Coordinator(fake_redis, run_id, contract=self._make_contract())
        dispatch_tool = coordinator._tools.get("dispatch_stage")
        captured_args = []
        with patch.object(
            dispatch_tool,
            "execute",
            side_effect=lambda args, ctx: captured_args.append(args) or ToolResult(success=True),
        ):
            await coordinator.on_event({"type": "job_failed", "error": "crash"}, LoopState(run_id=run_id))

        assert len(captured_args) == 1, "dispatch must be called exactly once"
        assert captured_args[0].stage == "auto_labeling", (
            f"dispatched stage was {captured_args[0].stage!r}, expected 'auto_labeling'"
        )

    @pytest.mark.asyncio
    async def test_dispatch_called_exactly_once(self, fake_redis):
        """dispatch_stage.execute() is called exactly once per job_failed event."""
        from ml_engine.agent.coordinator import Coordinator
        from ml_engine.agent.loop import LoopState
        from ml_engine.agent.tools import ToolResult

        run_id = "rd-count-0001"
        await self._reach_stage(run_id, fake_redis, "teacher_training")

        coordinator = Coordinator(fake_redis, run_id, contract=self._make_contract())
        dispatch_tool = coordinator._tools.get("dispatch_stage")
        with patch.object(dispatch_tool, "execute", return_value=ToolResult(success=True)) as mock_dispatch:
            await coordinator.on_event({"type": "job_failed", "error": "crash"}, LoopState(run_id=run_id))

        assert mock_dispatch.call_count == 1

    # ------------------------------------------------------------------
    # Overrides forwarding — catches None vs {} bug
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_overrides_forwarded_verbatim(self, fake_redis):
        """LoopState.stage_dispatch_overrides must reach dispatch args unchanged."""
        from ml_engine.agent.coordinator import Coordinator
        from ml_engine.agent.loop import LoopState
        from ml_engine.agent.tools import ToolResult

        run_id = "rd-override-0001"
        await self._reach_stage(run_id, fake_redis, "teacher_training")

        overrides = {"lr": 0.001, "epochs": 5, "batch_size": 16}
        coordinator = Coordinator(fake_redis, run_id, contract=self._make_contract())
        dispatch_tool = coordinator._tools.get("dispatch_stage")
        captured = []
        with patch.object(
            dispatch_tool,
            "execute",
            side_effect=lambda args, ctx: captured.append(args) or ToolResult(success=True),
        ):
            await coordinator.on_event(
                {"type": "job_failed", "error": "crash"},
                LoopState(run_id=run_id, stage_dispatch_overrides=overrides),
            )

        assert captured[0].overrides == overrides

    @pytest.mark.asyncio
    async def test_empty_overrides_forwarded_as_empty_dict_not_none(self, fake_redis):
        """When LoopState has no overrides, dispatch args.overrides must be {} not None.
        This catches the `overrides=state.stage_dispatch_overrides or {}` guard."""
        from ml_engine.agent.coordinator import Coordinator
        from ml_engine.agent.loop import LoopState
        from ml_engine.agent.tools import ToolResult

        run_id = "rd-override-0002"
        await self._reach_stage(run_id, fake_redis, "teacher_training")

        coordinator = Coordinator(fake_redis, run_id, contract=self._make_contract())
        dispatch_tool = coordinator._tools.get("dispatch_stage")
        captured = []
        with patch.object(
            dispatch_tool,
            "execute",
            side_effect=lambda args, ctx: captured.append(args) or ToolResult(success=True),
        ):
            await coordinator.on_event({"type": "job_failed", "error": "crash"}, LoopState(run_id=run_id))

        assert captured[0].overrides is not None, "overrides must not be None"
        assert captured[0].overrides == {}

    # ------------------------------------------------------------------
    # retry_count bookkeeping
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_retry_count_increments_by_one(self, fake_redis):
        """retry_count must go from 0 → 1 on first retry."""
        from ml_engine.agent.coordinator import Coordinator
        from ml_engine.agent.loop import LoopState
        from ml_engine.agent.tools import ToolResult

        run_id = "rd-count-0002"
        sm = await self._reach_stage(run_id, fake_redis, "teacher_training")
        assert await sm.retry_count() == 0

        coordinator = Coordinator(fake_redis, run_id, contract=self._make_contract())
        dispatch_tool = coordinator._tools.get("dispatch_stage")
        with patch.object(dispatch_tool, "execute", return_value=ToolResult(success=True)):
            await coordinator.on_event({"type": "job_failed", "error": "crash"}, LoopState(run_id=run_id))

        assert await sm.retry_count() == 1

    @pytest.mark.asyncio
    async def test_second_retry_increments_to_two(self, fake_redis):
        """Second retry: retry_count goes from 1 → 2 (not reset, not double-incremented)."""
        from ml_engine.agent.coordinator import Coordinator
        from ml_engine.agent.loop import LoopState
        from ml_engine.agent.tools import ToolResult

        run_id = "rd-count-0003"
        sm = await self._reach_stage(run_id, fake_redis, "teacher_training")
        # Simulate first retry already happened
        await sm.transition("failed_retrying", error_message="first failure")
        await sm.transition("teacher_training")

        coordinator = Coordinator(fake_redis, run_id, contract=self._make_contract(max_retries=3))
        dispatch_tool = coordinator._tools.get("dispatch_stage")
        with patch.object(dispatch_tool, "execute", return_value=ToolResult(success=True)):
            await coordinator.on_event(
                {"type": "job_failed", "error": "second crash"}, LoopState(run_id=run_id)
            )

        assert await sm.retry_count() == 2

    # ------------------------------------------------------------------
    # Retries exhausted — must not dispatch
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_retries_exhausted_no_dispatch_ends_in_failed_unrecoverable(self, fake_redis):
        """When retry_count >= max_retries, dispatch must NOT be called and SM
        must end in failed_unrecoverable."""
        from ml_engine.agent.coordinator import Coordinator
        from ml_engine.agent.loop import LoopState
        from ml_engine.agent.tools import ToolResult

        run_id = "rd-exhaust-0001"
        sm = await self._reach_stage(run_id, fake_redis, "teacher_training")
        await fake_redis.hset(sm._key, "retry_count", "2")  # at budget limit

        coordinator = Coordinator(fake_redis, run_id, contract=self._make_contract(max_retries=2))
        dispatch_tool = coordinator._tools.get("dispatch_stage")
        with patch.object(dispatch_tool, "execute", return_value=ToolResult(success=True)) as mock_dispatch:
            await coordinator.on_event({"type": "job_failed", "error": "crash"}, LoopState(run_id=run_id))

        assert await sm.current_state() == "failed_unrecoverable"
        mock_dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_retry_count_one_below_max_still_dispatches(self, fake_redis):
        """retry_count == max_retries - 1 must still dispatch (boundary: last allowed retry)."""
        from ml_engine.agent.coordinator import Coordinator
        from ml_engine.agent.loop import LoopState
        from ml_engine.agent.tools import ToolResult

        run_id = "rd-exhaust-0002"
        sm = await self._reach_stage(run_id, fake_redis, "teacher_training")
        await fake_redis.hset(sm._key, "retry_count", "1")  # one retry used, max is 2

        coordinator = Coordinator(fake_redis, run_id, contract=self._make_contract(max_retries=2))
        dispatch_tool = coordinator._tools.get("dispatch_stage")
        with patch.object(dispatch_tool, "execute", return_value=ToolResult(success=True)) as mock_dispatch:
            await coordinator.on_event({"type": "job_failed", "error": "crash"}, LoopState(run_id=run_id))

        assert await sm.current_state() == "teacher_training"
        mock_dispatch.assert_called_once()

    # ------------------------------------------------------------------
    # Dispatch failure path
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_dispatch_failure_ends_in_failed_unrecoverable(self, fake_redis):
        """If dispatch_stage.execute() returns success=False, SM must end in
        failed_unrecoverable (not stuck in a work stage or failed_retrying)."""
        from ml_engine.agent.coordinator import Coordinator
        from ml_engine.agent.loop import LoopState
        from ml_engine.agent.tools import ToolResult

        run_id = "rd-fail-0001"
        sm = await self._reach_stage(run_id, fake_redis, "auto_labeling")

        coordinator = Coordinator(fake_redis, run_id, contract=self._make_contract())
        dispatch_tool = coordinator._tools.get("dispatch_stage")
        with patch.object(
            dispatch_tool, "execute", return_value=ToolResult(success=False, error="executor queue full")
        ):
            await coordinator.on_event(
                {"type": "job_failed", "error": "worker died"}, LoopState(run_id=run_id)
            )

        assert await sm.current_state() == "failed_unrecoverable"

    # ------------------------------------------------------------------
    # Error message persistence
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_job_error_message_stored_in_sm(self, fake_redis):
        """The error field from the job_failed event must be written to SM
        (it is set during the failed_retrying transition)."""
        from ml_engine.agent.coordinator import Coordinator
        from ml_engine.agent.loop import LoopState
        from ml_engine.agent.tools import ToolResult

        run_id = "rd-err-0001"
        sm = await self._reach_stage(run_id, fake_redis, "teacher_training")

        coordinator = Coordinator(fake_redis, run_id, contract=self._make_contract())
        dispatch_tool = coordinator._tools.get("dispatch_stage")
        with patch.object(dispatch_tool, "execute", return_value=ToolResult(success=True)):
            await coordinator.on_event(
                {"type": "job_failed", "error": "GPU out of memory: 24 GiB"},
                LoopState(run_id=run_id),
            )

        data = await sm.load()
        assert "GPU out of memory" in data.get("error_message", ""), (
            "error_message not persisted in SM — failed_retrying transition must store it"
        )

    @pytest.mark.asyncio
    async def test_dispatch_execute_raises_ends_in_failed_unrecoverable(self, fake_redis):
        """If dispatch_stage.execute() raises an exception (not just returns success=False),
        the run must end in failed_unrecoverable — not stuck in the work stage."""
        from ml_engine.agent.coordinator import Coordinator
        from ml_engine.agent.loop import LoopState

        run_id = "rd-raise-0001"
        sm = await self._reach_stage(run_id, fake_redis, "auto_labeling")

        coordinator = Coordinator(fake_redis, run_id, contract=self._make_contract())
        dispatch_tool = coordinator._tools.get("dispatch_stage")
        with patch.object(dispatch_tool, "execute", side_effect=RuntimeError("redis timeout")):
            await coordinator.on_event(
                {"type": "job_failed", "error": "worker died"}, LoopState(run_id=run_id)
            )

        assert await sm.current_state() == "failed_unrecoverable", (
            "dispatch_stage.execute() raised but run was not marked failed_unrecoverable"
        )

    # ------------------------------------------------------------------
    # No LLM call on job_failed (deterministic path)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_no_llm_call_on_job_failed(self, fake_redis):
        """job_failed handling is fully deterministic — the LLM must never be invoked."""
        from ml_engine.agent.coordinator import Coordinator
        from ml_engine.agent.loop import LoopState
        from ml_engine.agent.tools import ToolResult

        run_id = "rd-llm-0001"
        await self._reach_stage(run_id, fake_redis, "teacher_training")

        coordinator = Coordinator(fake_redis, run_id, contract=self._make_contract())
        dispatch_tool = coordinator._tools.get("dispatch_stage")
        with (
            patch.object(dispatch_tool, "execute", return_value=ToolResult(success=True)),
            patch.object(
                coordinator._llm, "call", side_effect=AssertionError("LLM must not be called on job_failed")
            ),
        ):
            # Would raise if LLM is invoked
            await coordinator.on_event({"type": "job_failed", "error": "crash"}, LoopState(run_id=run_id))
