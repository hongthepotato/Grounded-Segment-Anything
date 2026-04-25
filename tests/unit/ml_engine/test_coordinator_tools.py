"""
Unit tests for ml_engine.agent.coordinator Tool classes.

Tests each tool's execute() paths in isolation using async fakeredis.
No LLM client needed -- tools are pure Redis + event publishing.

DispatchStageTool patches get_async_job_manager so the test does not need a
real Redis connection for the job store; the stream publish uses the shared
fakeredis via self._r.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ml_engine.agent.contracts import (
    AcceptanceCriteria,
    BudgetSpec,
    DataSpec,
    LineageSpec,
    PipelineContract,
    TargetSpec,
)
from ml_engine.agent.coordinator import (
    AdvanceGateArgs,
    AdvanceGateTool,
    DispatchStageArgs,
    DispatchStageTool,
    InspectStatusArgs,
    InspectStatusTool,
    ProposePlanArgs,
    ProposePlanTool,
    ReadMemoryArgs,
    ReadMemoryTool,
    RequestEvaluationArgs,
    RequestEvaluationTool,
)
from ml_engine.agent.memory import MemoryStore
from ml_engine.agent.state_machine import StateMachine
from ml_engine.agent.tools import RunContext
from tests.unit.ml_engine.conftest import read_stream_events

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def run_id():
    return "tool-test-run-001"


@pytest.fixture
def contract_dict():
    return PipelineContract(
        id="contract-001",
        target=TargetSpec(class_names=["defect"]),
        data=DataSpec(data_path="/data", image_paths=["img1.jpg"]),
        acceptance_criteria=AcceptanceCriteria(min_mAP50=0.5),
        budget=BudgetSpec(max_retries=2),
        lineage=LineageSpec(),
    ).to_dict()


@pytest.fixture
def context(run_id, contract_dict):
    return RunContext(
        run_id=run_id,
        redis_url="redis://localhost:6379",
        contract=contract_dict,
    )


@pytest.fixture
def context_no_contract(run_id):
    return RunContext(
        run_id=run_id,
        redis_url="redis://localhost:6379",
        contract=None,
    )


def _mock_job_manager():
    """Return a mock AsyncJobManager whose store.store_job is a no-op."""
    mock_store = MagicMock()
    mock_store.store_job = AsyncMock()
    mock_manager = MagicMock()
    mock_manager.store = mock_store
    return mock_manager


# ---------------------------------------------------------------------------
# InspectStatusTool
# ---------------------------------------------------------------------------


class TestInspectStatusTool:
    @pytest.mark.asyncio
    async def test_returns_state_dict_for_known_run(self, redis_async, run_id, context):
        sm = StateMachine(run_id=run_id, redis_async=redis_async)
        await sm.initialize()
        tool = InspectStatusTool(redis_async)
        result = await tool.execute(InspectStatusArgs(run_id=run_id), context)
        assert result.success is True
        assert result.output["state"]["state"] == "created"

    @pytest.mark.asyncio
    async def test_returns_failure_for_unknown_run(self, redis_async, context):
        tool = InspectStatusTool(redis_async)
        result = await tool.execute(InspectStatusArgs(run_id="nonexistent-run"), context)
        assert result.success is False
        assert "No state" in result.error

    @pytest.mark.asyncio
    async def test_stage_summaries_parsed_from_load(self, redis_async, run_id, context):
        sm = StateMachine(run_id=run_id, redis_async=redis_async)
        await sm.initialize()
        await sm.append_stage_summary({"stage": "teacher_training", "status": "pass"})
        tool = InspectStatusTool(redis_async)
        result = await tool.execute(InspectStatusArgs(run_id=run_id), context)
        assert result.success is True
        assert len(result.output["stage_summaries"]) == 1
        assert result.output["stage_summaries"][0]["stage"] == "teacher_training"


# ---------------------------------------------------------------------------
# ReadMemoryTool
# ---------------------------------------------------------------------------


class TestReadMemoryTool:
    @pytest.mark.asyncio
    async def test_returns_empty_for_empty_store(self, redis_async, context):
        memory = MemoryStore(redis_async=redis_async)
        tool = ReadMemoryTool(memory)
        result = await tool.execute(ReadMemoryArgs(types=["feedback"]), context)
        assert result.success is True
        assert result.output["memory"]["feedback"] == []

    @pytest.mark.asyncio
    async def test_returns_written_records(self, redis_async, context):
        memory = MemoryStore(redis_async=redis_async)
        await memory.write("feedback", "key1", {"note": "test feedback"})
        tool = ReadMemoryTool(memory)
        result = await tool.execute(ReadMemoryArgs(types=["feedback"]), context)
        assert len(result.output["memory"]["feedback"]) == 1

    @pytest.mark.asyncio
    async def test_multiple_types_returned(self, redis_async, context):
        memory = MemoryStore(redis_async=redis_async)
        await memory.write("feedback", "f1", {"x": 1})
        await memory.write("project", "p1", {"y": 2})
        tool = ReadMemoryTool(memory)
        result = await tool.execute(ReadMemoryArgs(types=["feedback", "project"]), context)
        assert "feedback" in result.output["memory"]
        assert "project" in result.output["memory"]
        assert len(result.output["memory"]["feedback"]) == 1
        assert len(result.output["memory"]["project"]) == 1


# ---------------------------------------------------------------------------
# DispatchStageTool
# ---------------------------------------------------------------------------


class TestDispatchStageTool:
    @pytest.mark.asyncio
    async def test_unknown_stage_returns_failure(self, redis_async, context):
        tool = DispatchStageTool(redis_async)
        result = await tool.execute(DispatchStageArgs(stage="unknown_stage"), context)
        assert result.success is False
        assert "Unknown stage" in result.error

    @pytest.mark.asyncio
    async def test_no_contract_returns_failure(self, redis_async, context_no_contract):
        tool = DispatchStageTool(redis_async)
        result = await tool.execute(DispatchStageArgs(stage="teacher_training"), context_no_contract)
        assert result.success is False
        assert "No contract" in result.error

    @pytest.mark.asyncio
    async def test_happy_path_publishes_dispatch_requested(self, redis_async, redis_sync, run_id, context):
        tool = DispatchStageTool(redis_async)
        with patch("ml_engine.jobs.get_async_job_manager", return_value=_mock_job_manager()):
            result = await tool.execute(DispatchStageArgs(stage="teacher_training"), context)
        assert result.success is True
        events = read_stream_events(redis_sync, run_id, "dispatch_requested")
        assert len(events) == 1
        assert events[0]["stage"] == "teacher_training"

    @pytest.mark.asyncio
    async def test_happy_path_returns_job_id(self, redis_async, run_id, context):
        tool = DispatchStageTool(redis_async)
        with patch("ml_engine.jobs.get_async_job_manager", return_value=_mock_job_manager()):
            result = await tool.execute(DispatchStageArgs(stage="teacher_training"), context)
        assert result.success is True
        assert "job_id" in result.output

    @pytest.mark.asyncio
    async def test_overrides_merged_into_job_config(self, redis_async, run_id, context):
        tool = DispatchStageTool(redis_async)
        mock_mgr = _mock_job_manager()
        with patch("ml_engine.jobs.get_async_job_manager", return_value=mock_mgr):
            await tool.execute(
                DispatchStageArgs(stage="teacher_training", overrides={"epochs": 50}),
                context,
            )
        stored_job = mock_mgr.store.store_job.call_args[0][0]
        assert stored_job.config["epochs"] == 50


# ---------------------------------------------------------------------------
# RequestEvaluationTool
# ---------------------------------------------------------------------------


class TestRequestEvaluationTool:
    @pytest.mark.asyncio
    async def test_publishes_evaluation_requested(self, redis_async, redis_sync, run_id, context):
        tool = RequestEvaluationTool(redis_async)
        result = await tool.execute(
            RequestEvaluationArgs(stage="teacher_training", job_id="job-001", output_dir="/out"),
            context,
        )
        assert result.success is True
        events = read_stream_events(redis_sync, run_id, "evaluation_requested")
        assert len(events) == 1
        assert events[0]["stage"] == "teacher_training"
        assert events[0]["job_id"] == "job-001"
        assert events[0]["output_dir"] == "/out"


# ---------------------------------------------------------------------------
# AdvanceGateTool
# ---------------------------------------------------------------------------


class TestAdvanceGateTool:
    @pytest.mark.asyncio
    async def test_valid_transition_succeeds(self, redis_async, run_id, context):
        sm = StateMachine(run_id=run_id, redis_async=redis_async)
        await sm.initialize()
        tool = AdvanceGateTool(redis_async)
        result = await tool.execute(
            AdvanceGateArgs(target_state="planning", reason="starting pipeline"),
            context,
        )
        assert result.success is True
        assert result.output["new_state"] == "planning"
        assert await sm.current_state() == "planning"

    @pytest.mark.asyncio
    async def test_invalid_transition_returns_failure(self, redis_async, run_id, context):
        sm = StateMachine(run_id=run_id, redis_async=redis_async)
        await sm.initialize()
        tool = AdvanceGateTool(redis_async)
        # created -> done is not a valid transition
        result = await tool.execute(
            AdvanceGateArgs(target_state="done", reason="skip everything"),
            context,
        )
        assert result.success is False
        assert result.error

    @pytest.mark.asyncio
    async def test_valid_transition_publishes_state_changed(self, redis_async, redis_sync, run_id, context):
        sm = StateMachine(run_id=run_id, redis_async=redis_async)
        await sm.initialize()
        tool = AdvanceGateTool(redis_async)
        await tool.execute(
            AdvanceGateArgs(target_state="planning", reason="test"),
            context,
        )
        events = read_stream_events(redis_sync, run_id, "state_changed")
        assert len(events) == 1
        assert events[0]["new_state"] == "planning"


# ---------------------------------------------------------------------------
# ProposePlanTool
# ---------------------------------------------------------------------------


class TestProposePlanTool:
    @pytest.mark.asyncio
    async def test_returns_contract_with_class_names(self, redis_async, context):
        memory = MemoryStore(redis_async=redis_async)
        tool = ProposePlanTool(memory)
        result = await tool.execute(
            ProposePlanArgs(
                intent="detect surface defects",
                data_path="/data/plant1",
                image_paths=["img1.jpg"],
                class_names=["defect", "scratch"],
            ),
            context,
        )
        assert result.success is True
        assert result.output["contract"]["target"]["class_names"] == ["defect", "scratch"]

    @pytest.mark.asyncio
    async def test_memory_applied_flag_present(self, redis_async, context):
        memory = MemoryStore(redis_async=redis_async)
        tool = ProposePlanTool(memory)
        result = await tool.execute(
            ProposePlanArgs(
                intent="test",
                data_path="/d",
                image_paths=[],
                class_names=["x"],
            ),
            context,
        )
        assert "memory_applied" in result.output

    @pytest.mark.asyncio
    async def test_contract_note_references_approval_endpoint(self, redis_async, context):
        memory = MemoryStore(redis_async=redis_async)
        tool = ProposePlanTool(memory)
        result = await tool.execute(
            ProposePlanArgs(
                intent="test",
                data_path="/d",
                image_paths=[],
                class_names=["x"],
            ),
            context,
        )
        assert "approve" in result.output["note"].lower()
