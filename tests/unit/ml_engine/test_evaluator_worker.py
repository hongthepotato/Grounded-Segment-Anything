"""
Unit tests for ml_engine.agent.workers.EvaluatorWorker.

Tests _handle_evaluation() in isolation using a tmp outcome.json file
and async fakeredis. No real asyncio streams needed -- we call
_handle_evaluation directly.

Async-only after Phase 7 (StateMachine/MemoryStore sync API removed).
Event readback uses the sync twin (``redis_sync``) which shares the same
FakeServer, keeping the helpers simple.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import pytest_asyncio

from ml_engine.agent.memory import MemoryStore
from ml_engine.agent.workers import EvaluatorWorker
from tests.unit.ml_engine.conftest import read_stream_events
from ml_engine.agent.contracts import (
    AcceptanceCriteria,
    BudgetSpec,
    DataSpec,
    LineageSpec,
    PipelineContract,
    TargetSpec,
)
from ml_engine.agent.state_machine import StateMachine


# ---------------------------------------------------------------------------
# Fixtures (redis_async, redis_sync come from conftest and share FakeServer)
# ---------------------------------------------------------------------------

@pytest.fixture
def run_id():
    return "eval-worker-test-001"


@pytest.fixture
def contract():
    return PipelineContract(
        id="contract-001",
        target=TargetSpec(class_names=["defect"]),
        data=DataSpec(data_path="/data", image_paths=[]),
        acceptance_criteria=AcceptanceCriteria(min_mAP50=0.5, min_mIoU=0.4),
        budget=BudgetSpec(max_retries=2),
        lineage=LineageSpec(),
    )


@pytest_asyncio.fixture
async def worker(redis_async, run_id, contract):
    """Initialize state machine so retry_count is readable, return worker."""
    sm = StateMachine(run_id=run_id, redis_async=redis_async)
    await sm.initialize()
    return EvaluatorWorker(redis_async, run_id, contract=contract)


def make_outcome(tmp_path: Path, metrics: dict, **kwargs) -> str:
    """Write outcome.json to tmp_path and return the dir path."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    outcome = {"metrics": metrics, "wall_time_seconds": 120.0, "artifacts": {}}
    outcome.update(kwargs)
    (out_dir / "outcome.json").write_text(json.dumps(outcome))
    return str(out_dir)


def read_gate_events(redis_sync, run_id: str) -> list:
    """Read gate_decision events. Thin wrapper around the shared conftest helper."""
    return read_stream_events(redis_sync, run_id, "gate_decision")


# ---------------------------------------------------------------------------
# _handle_evaluation -- outcome.json missing
# ---------------------------------------------------------------------------

class TestHandleEvaluationMissingOutcome:
    @pytest.mark.asyncio
    async def test_escalates_when_output_dir_missing(self, worker, redis_sync, run_id):
        event = {
            "type": "evaluation_requested",
            "stage": "teacher_training",
            "job_id": "job-001",
            "output_dir": "/nonexistent/path/1234",
        }
        await worker._handle_evaluation(event)
        gate_events = read_gate_events(redis_sync, run_id)
        assert len(gate_events) == 1
        assert gate_events[0]["verdict"] == "escalate"
        assert "outcome.json not found" in gate_events[0]["reason"]

    @pytest.mark.asyncio
    async def test_escalates_when_outcome_file_missing_but_dir_exists(self, worker, redis_sync, run_id, tmp_path):
        out_dir = tmp_path / "empty_output"
        out_dir.mkdir()
        event = {
            "type": "evaluation_requested",
            "stage": "teacher_training",
            "job_id": "job-002",
            "output_dir": str(out_dir),
        }
        await worker._handle_evaluation(event)
        gate_events = read_gate_events(redis_sync, run_id)
        assert gate_events[0]["verdict"] == "escalate"


# ---------------------------------------------------------------------------
# _handle_evaluation -- pass verdicts
# ---------------------------------------------------------------------------

class TestHandleEvaluationPass:
    @pytest.mark.asyncio
    async def test_pass_when_mAP50_above_threshold(self, worker, redis_sync, run_id, tmp_path):
        out_dir = make_outcome(tmp_path, {"mAP50": 0.72})
        event = {
            "type": "evaluation_requested",
            "stage": "teacher_training",
            "job_id": "job-pass",
            "output_dir": out_dir,
        }
        await worker._handle_evaluation(event)
        gate_events = read_gate_events(redis_sync, run_id)
        assert gate_events[0]["verdict"] == "pass"

    @pytest.mark.asyncio
    async def test_gate_event_includes_metrics(self, worker, redis_sync, run_id, tmp_path):
        out_dir = make_outcome(tmp_path, {"mAP50": 0.65, "val_loss": 0.8})
        event = {
            "type": "evaluation_requested",
            "stage": "teacher_training",
            "job_id": "job-metrics",
            "output_dir": out_dir,
        }
        await worker._handle_evaluation(event)
        gate_events = read_gate_events(redis_sync, run_id)
        assert gate_events[0]["metrics"]["mAP50"] == pytest.approx(0.65)

    @pytest.mark.asyncio
    async def test_gate_event_includes_stage(self, worker, redis_sync, run_id, tmp_path):
        out_dir = make_outcome(tmp_path, {"mAP50": 0.72})
        event = {
            "type": "evaluation_requested",
            "stage": "teacher_training",
            "job_id": "job-stage",
            "output_dir": out_dir,
        }
        await worker._handle_evaluation(event)
        gate_events = read_gate_events(redis_sync, run_id)
        assert gate_events[0]["stage"] == "teacher_training"

    @pytest.mark.asyncio
    async def test_gate_event_includes_wall_time(self, worker, redis_sync, run_id, tmp_path):
        out_dir = make_outcome(tmp_path, {"mAP50": 0.72})
        event = {
            "type": "evaluation_requested",
            "stage": "teacher_training",
            "job_id": "job-wt",
            "output_dir": out_dir,
        }
        await worker._handle_evaluation(event)
        gate_events = read_gate_events(redis_sync, run_id)
        assert gate_events[0]["wall_time_seconds"] == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# _handle_evaluation -- retry / escalate
# ---------------------------------------------------------------------------

class TestHandleEvaluationRetry:
    @pytest.mark.asyncio
    async def test_retry_when_below_threshold(self, worker, redis_sync, run_id, tmp_path):
        out_dir = make_outcome(tmp_path, {"mAP50": 0.3})
        event = {
            "type": "evaluation_requested",
            "stage": "teacher_training",
            "job_id": "job-retry",
            "output_dir": out_dir,
        }
        await worker._handle_evaluation(event)
        gate_events = read_gate_events(redis_sync, run_id)
        assert gate_events[0]["verdict"] == "retry"

    @pytest.mark.asyncio
    async def test_escalate_when_retries_exhausted(self, redis_async, redis_sync, run_id, tmp_path):
        sm = StateMachine(run_id=run_id + "-exhaust", redis_async=redis_async)
        await sm.initialize()
        # Force retry_count to max
        await redis_async.hset(sm._key, "retry_count", "2")
        contract = PipelineContract(
            id="c",
            target=TargetSpec(class_names=["x"]),
            data=DataSpec(data_path="/d", image_paths=[]),
            acceptance_criteria=AcceptanceCriteria(min_mAP50=0.5),
            budget=BudgetSpec(max_retries=2),
            lineage=LineageSpec(),
        )
        worker = EvaluatorWorker(redis_async, run_id + "-exhaust", contract=contract)
        out_dir = make_outcome(tmp_path, {"mAP50": 0.2})
        event = {
            "type": "evaluation_requested",
            "stage": "teacher_training",
            "job_id": "job-exhaust",
            "output_dir": out_dir,
        }
        await worker._handle_evaluation(event)
        gate_events = read_gate_events(redis_sync, run_id + "-exhaust")
        assert gate_events[0]["verdict"] == "escalate"


# ---------------------------------------------------------------------------
# _handle_evaluation -- no contract (pre-approval)
# ---------------------------------------------------------------------------

class TestHandleEvaluationNoContract:
    @pytest.mark.asyncio
    async def test_defaults_to_pass_when_no_contract(self, redis_async, redis_sync, run_id, tmp_path):
        sm = StateMachine(run_id=run_id + "-nocontract", redis_async=redis_async)
        await sm.initialize()
        worker = EvaluatorWorker(redis_async, run_id + "-nocontract", contract=None)
        out_dir = make_outcome(tmp_path, {"mAP50": 0.1})  # would fail with contract
        event = {
            "type": "evaluation_requested",
            "stage": "teacher_training",
            "job_id": "job-nc",
            "output_dir": out_dir,
        }
        await worker._handle_evaluation(event)
        gate_events = read_gate_events(redis_sync, run_id + "-nocontract")
        assert gate_events[0]["verdict"] == "pass"
        assert "no contract" in gate_events[0]["reason"]


# ---------------------------------------------------------------------------
# _write_feedback_memory
# ---------------------------------------------------------------------------

class TestWriteFeedbackMemory:
    @pytest.mark.asyncio
    async def test_feedback_written_to_redis(self, worker, redis_async, run_id, tmp_path):
        out_dir = make_outcome(tmp_path, {"mAP50": 0.65})
        event = {
            "type": "evaluation_requested",
            "stage": "teacher_training",
            "job_id": "job-mem",
            "output_dir": out_dir,
        }
        await worker._handle_evaluation(event)
        store = MemoryStore(redis_async=redis_async)
        records = await store.read("feedback")
        assert len(records) >= 1
        stages = [r["content"]["stage"] for r in records]
        assert "teacher_training" in stages

    @pytest.mark.asyncio
    async def test_feedback_includes_verdict(self, worker, redis_async, run_id, tmp_path):
        out_dir = make_outcome(tmp_path, {"mAP50": 0.65})
        event = {
            "type": "evaluation_requested",
            "stage": "teacher_training",
            "job_id": "job-verdict",
            "output_dir": out_dir,
        }
        await worker._handle_evaluation(event)
        store = MemoryStore(redis_async=redis_async)
        records = await store.read("feedback")
        # StageSummary uses "status" for the gate verdict (pass/retry/escalate).
        statuses = [r["content"]["status"] for r in records]
        assert "pass" in statuses

    @pytest.mark.asyncio
    async def test_experiment_result_included_when_present(self, worker, redis_async, run_id, tmp_path):
        out_dir = make_outcome(
            tmp_path,
            {"mAP50": 0.65},
            experiment_result={"trials_completed": 10, "best_metric": 0.65},
        )
        event = {
            "type": "evaluation_requested",
            "stage": "experiment_loop",
            "job_id": "job-exp",
            "output_dir": out_dir,
        }
        await worker._handle_evaluation(event)
        store = MemoryStore(redis_async=redis_async)
        records = await store.read("feedback")
        exp_records = [r for r in records if r["content"].get("stage") == "experiment_loop"]
        assert len(exp_records) >= 1
        assert "experiment_result" in exp_records[0]["content"]
        assert "note" in exp_records[0]["content"]

    @pytest.mark.asyncio
    async def test_memory_write_failure_does_not_crash(self, redis_async, redis_sync, run_id, tmp_path):
        """MemoryStore failure must be swallowed, not propagate."""
        sm = StateMachine(run_id=run_id + "-memfail", redis_async=redis_async)
        await sm.initialize()
        worker = EvaluatorWorker(redis_async, run_id + "-memfail")

        from unittest.mock import patch
        out_dir = make_outcome(tmp_path, {"mAP50": 0.7})
        event = {
            "type": "evaluation_requested",
            "stage": "teacher_training",
            "job_id": "job-mf",
            "output_dir": out_dir,
        }
        with patch("ml_engine.agent.memory.MemoryStore.write", side_effect=RuntimeError("redis down")):
            # Should not raise
            await worker._handle_evaluation(event)

        # Gate decision still published
        gate_events = read_gate_events(redis_sync, run_id + "-memfail")
        assert len(gate_events) == 1


# ---------------------------------------------------------------------------
# set_contract()
# ---------------------------------------------------------------------------

class TestSetContract:
    @pytest.mark.asyncio
    async def test_set_contract_updates_contract(self, worker, contract):
        new_contract = PipelineContract(
            id="new",
            target=TargetSpec(class_names=["crack"]),
            data=DataSpec(data_path="/new", image_paths=[]),
            acceptance_criteria=AcceptanceCriteria(min_mAP50=0.8),
            budget=BudgetSpec(max_retries=1),
            lineage=LineageSpec(),
        )
        worker.set_contract(new_contract)
        assert worker._contract.acceptance_criteria.min_mAP50 == 0.8


# ---------------------------------------------------------------------------
# Distillation gate path (mIoU metric)
# ---------------------------------------------------------------------------

class TestHandleEvaluationDistillation:
    @pytest.mark.asyncio
    async def test_distillation_pass_when_mIoU_above_threshold(
        self, redis_async, redis_sync, run_id, tmp_path
    ):
        sm = StateMachine(run_id=run_id + "-distill-pass", redis_async=redis_async)
        await sm.initialize()
        contract = PipelineContract(
            id="c",
            target=TargetSpec(class_names=["defect"]),
            data=DataSpec(data_path="/d", image_paths=[]),
            acceptance_criteria=AcceptanceCriteria(min_mAP50=0.5, min_mIoU=0.4),
            budget=BudgetSpec(max_retries=2),
            lineage=LineageSpec(),
        )
        worker = EvaluatorWorker(redis_async, run_id + "-distill-pass", contract=contract)
        out_dir = make_outcome(tmp_path, {"mIoU": 0.6})
        event = {
            "type": "evaluation_requested",
            "stage": "student_distillation",
            "job_id": "job-distill-pass",
            "output_dir": out_dir,
        }
        await worker._handle_evaluation(event)
        gate_events = read_gate_events(redis_sync, run_id + "-distill-pass")
        assert gate_events[0]["verdict"] == "pass"
        assert gate_events[0]["stage"] == "student_distillation"

    @pytest.mark.asyncio
    async def test_distillation_retry_when_mIoU_below_threshold(
        self, redis_async, redis_sync, run_id, tmp_path
    ):
        sm = StateMachine(run_id=run_id + "-distill-retry", redis_async=redis_async)
        await sm.initialize()
        contract = PipelineContract(
            id="c2",
            target=TargetSpec(class_names=["defect"]),
            data=DataSpec(data_path="/d", image_paths=[]),
            acceptance_criteria=AcceptanceCriteria(min_mAP50=0.5, min_mIoU=0.4),
            budget=BudgetSpec(max_retries=2),
            lineage=LineageSpec(),
        )
        worker = EvaluatorWorker(redis_async, run_id + "-distill-retry", contract=contract)
        out_dir = make_outcome(tmp_path, {"mIoU": 0.2})
        event = {
            "type": "evaluation_requested",
            "stage": "student_distillation",
            "job_id": "job-distill-retry",
            "output_dir": out_dir,
        }
        await worker._handle_evaluation(event)
        gate_events = read_gate_events(redis_sync, run_id + "-distill-retry")
        assert gate_events[0]["verdict"] == "retry"


# ---------------------------------------------------------------------------
# Malformed outcome.json (JSON parse error)
# ---------------------------------------------------------------------------

class TestHandleEvaluationMalformedOutcome:
    @pytest.mark.asyncio
    async def test_escalates_when_outcome_json_is_corrupt(
        self, redis_async, redis_sync, run_id, tmp_path
    ):
        """
        Malformed JSON causes json.loads to raise inside _handle_evaluation.
        StreamConsumer._dispatch catches it, calls on_event_error, which
        publishes gate_decision(escalate). Pipeline should not stall silently.
        """
        sm = StateMachine(run_id=run_id + "-corrupt", redis_async=redis_async)
        await sm.initialize()
        worker = EvaluatorWorker(redis_async, run_id + "-corrupt")

        out_dir = tmp_path / "corrupt_output"
        out_dir.mkdir()
        (out_dir / "outcome.json").write_text("{this is not valid json{{{{")

        event = {
            "type": "evaluation_requested",
            "stage": "teacher_training",
            "job_id": "job-corrupt",
            "output_dir": str(out_dir),
        }
        # on_event_error is called by _dispatch, not _handle_evaluation directly.
        # Call via on_event_error to simulate the base class dispatch flow.
        try:
            await worker._handle_evaluation(event)
        except Exception as exc:
            await worker.on_event_error(event, "test-entry-id", exc)

        gate_events = read_gate_events(redis_sync, run_id + "-corrupt")
        assert len(gate_events) == 1
        assert gate_events[0]["verdict"] == "escalate"
