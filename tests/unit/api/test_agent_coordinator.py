"""
Unit tests for Coordinator lifecycle in api/routes/agent.py.

Focuses on the _on_done callback: when Coordinator.run() crashes, the run
must transition to failed_unrecoverable with the exception message stored in
Redis so GET /api/agent/status returns a clear failure state instead of a
healthy-looking response with coordinator_active=False and no error context.
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
