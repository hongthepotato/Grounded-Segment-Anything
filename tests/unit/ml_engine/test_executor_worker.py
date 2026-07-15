"""
Unit tests for ml_engine.agent.workers.ExecutorWorker.

Focuses on _handle_dispatch() in isolation:
  - Normal dispatch path: job found, contract allows, enqueue succeeds
  - Idempotency: same entry_id_str re-delivers stage_dispatched without re-enqueuing
  - Dispatch rejected: missing job_id, retry budget exhausted
  - Contract absent: dispatch allowed (pre-approval planning phase)

Uses async fakeredis + AsyncRedisJobStore (real async store, no mocking of Redis).
The sync twin (redis_sync) shares the same FakeServer and is used to read
published stream events without async wiring.

Design note: ExecutorWorker trusts that dispatch_requested events carry valid
job IDs created by the Coordinator. An unknown job_id results in a sparse Redis
hash being created by update_job() before enqueue_by_id() is attempted. This is
a Coordinator-enforced invariant, not a worker-level guard.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from ml_engine.agent.contracts import (
    AcceptanceCriteria,
    BudgetSpec,
    DataSpec,
    LineageSpec,
    PipelineContract,
    TargetSpec,
)
from ml_engine.agent.state_machine import StateMachine
from ml_engine.agent.workers import ExecutorWorker
from ml_engine.jobs.async_redis_store import AsyncRedisJobStore
from ml_engine.jobs.models import Job, JobType
from tests.unit.ml_engine.conftest import read_stream_events

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def run_id():
    return "exec-worker-test-001"


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
async def store(redis_async):
    return AsyncRedisJobStore(redis_client=redis_async)


@pytest_asyncio.fixture
async def worker(redis_async, run_id, store, contract):
    """Executor worker with state machine initialized."""
    sm = StateMachine(run_id=run_id, redis_async=redis_async)
    await sm.initialize()
    return ExecutorWorker(redis_async, run_id, store=store, contract=contract)


@pytest_asyncio.fixture
async def job(store):
    """A pending job in the store, ready to be dispatched."""
    j = Job(
        type=JobType.TEACHER_TRAINING.value,
        config={"epochs": 10},
    )
    await store.store_job(j)
    return j


def read_events(redis_sync, run_id: str, event_type: str) -> list:
    """Read events of a given type from the stream. Delegates to shared conftest helper."""
    return read_stream_events(redis_sync, run_id, event_type)


async def queue_length(store: AsyncRedisJobStore) -> int:
    """Return the current length of the job queue."""
    return await store.get_queue_length()


# ---------------------------------------------------------------------------
# Normal dispatch path
# ---------------------------------------------------------------------------


class TestNormalDispatch:
    @pytest.mark.asyncio
    async def test_enqueues_job_and_publishes_stage_dispatched(self, worker, store, job, redis_sync, run_id):
        event = {
            "type": "dispatch_requested",
            "job_id": job.id,
            "stage": "teacher_training",
            "run_id": run_id,
        }
        await worker._handle_dispatch(event, entry_id_str="1000-0")

        dispatched = read_events(redis_sync, run_id, "stage_dispatched")
        assert len(dispatched) == 1
        assert dispatched[0]["job_id"] == job.id
        assert dispatched[0]["stage"] == "teacher_training"

    @pytest.mark.asyncio
    async def test_stamps_dispatch_event_id_on_job(self, worker, store, job, run_id):
        event = {
            "type": "dispatch_requested",
            "job_id": job.id,
            "stage": "teacher_training",
            "run_id": run_id,
        }
        await worker._handle_dispatch(event, entry_id_str="1234-0")

        updated = await store.get_job(job.id)
        assert updated.dispatch_event_id == "1234-0"

    @pytest.mark.asyncio
    async def test_job_lands_in_queue(self, worker, store, job, run_id):
        assert await queue_length(store) == 0
        event = {
            "type": "dispatch_requested",
            "job_id": job.id,
            "stage": "teacher_training",
            "run_id": run_id,
        }
        await worker._handle_dispatch(event, entry_id_str="1000-0")
        assert await queue_length(store) == 1

    @pytest.mark.asyncio
    async def test_no_rejected_event_on_success(self, worker, store, job, redis_sync, run_id):
        event = {
            "type": "dispatch_requested",
            "job_id": job.id,
            "stage": "teacher_training",
            "run_id": run_id,
        }
        await worker._handle_dispatch(event, entry_id_str="1000-0")

        rejected = read_events(redis_sync, run_id, "dispatch_rejected")
        assert rejected == []


# ---------------------------------------------------------------------------
# Idempotency: PEL re-delivery with same entry_id_str
# ---------------------------------------------------------------------------


class TestIdempotency:
    @pytest.mark.asyncio
    async def test_second_call_same_entry_id_does_not_reenqueue(self, worker, store, job, run_id):
        """When the same entry_id_str is seen again, the queue must not grow."""
        event = {
            "type": "dispatch_requested",
            "job_id": job.id,
            "stage": "teacher_training",
            "run_id": run_id,
        }
        await worker._handle_dispatch(event, entry_id_str="5000-0")
        assert await queue_length(store) == 1

        # Second call: same entry_id_str (PEL re-delivery)
        await worker._handle_dispatch(event, entry_id_str="5000-0")

        # Queue must still be length 1 -- job was NOT re-enqueued
        assert await queue_length(store) == 1

    @pytest.mark.asyncio
    async def test_idempotency_hit_publishes_stage_dispatched(self, worker, store, job, redis_sync, run_id):
        """Idempotency hit must re-publish stage_dispatched so Coordinator unblocks."""
        event = {
            "type": "dispatch_requested",
            "job_id": job.id,
            "stage": "teacher_training",
            "run_id": run_id,
        }
        await worker._handle_dispatch(event, entry_id_str="5000-0")
        await worker._handle_dispatch(event, entry_id_str="5000-0")

        dispatched = read_events(redis_sync, run_id, "stage_dispatched")
        # Two stage_dispatched events: one from first call, one from idempotency hit
        assert len(dispatched) == 2
        assert all(d["job_id"] == job.id for d in dispatched)

    @pytest.mark.asyncio
    async def test_idempotency_hit_emits_no_rejected_event(self, worker, store, job, redis_sync, run_id):
        event = {
            "type": "dispatch_requested",
            "job_id": job.id,
            "stage": "teacher_training",
            "run_id": run_id,
        }
        await worker._handle_dispatch(event, entry_id_str="5000-0")
        await worker._handle_dispatch(event, entry_id_str="5000-0")

        rejected = read_events(redis_sync, run_id, "dispatch_rejected")
        assert rejected == []

    @pytest.mark.asyncio
    async def test_different_entry_id_stamps_new_id(self, worker, store, job, run_id):
        """A new dispatch (different entry_id_str) should update dispatch_event_id."""
        event = {
            "type": "dispatch_requested",
            "job_id": job.id,
            "stage": "teacher_training",
            "run_id": run_id,
        }
        await worker._handle_dispatch(event, entry_id_str="5000-0")
        await worker._handle_dispatch(event, entry_id_str="6000-0")

        updated = await store.get_job(job.id)
        assert updated.dispatch_event_id == "6000-0"


# ---------------------------------------------------------------------------
# Dispatch rejected paths
# ---------------------------------------------------------------------------


class TestDispatchRejected:
    @pytest.mark.asyncio
    async def test_missing_job_id_skips_silently(self, worker, redis_sync, run_id):
        event = {
            "type": "dispatch_requested",
            "stage": "teacher_training",
            # no job_id
        }
        # Should not raise
        await worker._handle_dispatch(event, entry_id_str="1000-0")

        # No dispatched or rejected events
        assert read_events(redis_sync, run_id, "stage_dispatched") == []
        assert read_events(redis_sync, run_id, "dispatch_rejected") == []

    @pytest.mark.asyncio
    async def test_retry_budget_exhausted_publishes_dispatch_rejected(
        self, redis_async, run_id, store, job, redis_sync
    ):
        """When retry_count >= max_retries, dispatch must be rejected."""
        sm = StateMachine(run_id=run_id, redis_async=redis_async)
        await sm.initialize()
        # Directly write retry_count=2 into the state hash to exhaust the budget.
        # StateMachine only increments via failed_retrying transitions; we bypass
        # the transition guard here because we're testing the ExecutorWorker, not
        # the state machine.
        await redis_async.hset(sm._key, "retry_count", "2")

        contract = PipelineContract(
            id="contract-budget",
            target=TargetSpec(class_names=["defect"]),
            data=DataSpec(data_path="/data", image_paths=[]),
            acceptance_criteria=AcceptanceCriteria(min_mAP50=0.5, min_mIoU=0.4),
            budget=BudgetSpec(max_retries=2),
            lineage=LineageSpec(),
        )
        worker = ExecutorWorker(redis_async, run_id, store=store, contract=contract)

        event = {
            "type": "dispatch_requested",
            "job_id": job.id,
            "stage": "teacher_training",
            "run_id": run_id,
        }
        await worker._handle_dispatch(event, entry_id_str="1000-0")

        rejected = read_events(redis_sync, run_id, "dispatch_rejected")
        assert len(rejected) == 1
        assert "Retry budget exhausted" in rejected[0]["errors"][0]

    @pytest.mark.asyncio
    async def test_retry_budget_exhausted_does_not_enqueue(self, redis_async, run_id, store, job, redis_sync):
        sm = StateMachine(run_id=run_id, redis_async=redis_async)
        await sm.initialize()
        await redis_async.hset(sm._key, "retry_count", "2")

        contract = PipelineContract(
            id="contract-budget2",
            target=TargetSpec(class_names=["defect"]),
            data=DataSpec(data_path="/data", image_paths=[]),
            acceptance_criteria=AcceptanceCriteria(min_mAP50=0.5, min_mIoU=0.4),
            budget=BudgetSpec(max_retries=2),
            lineage=LineageSpec(),
        )
        worker = ExecutorWorker(redis_async, run_id, store=store, contract=contract)
        event = {
            "type": "dispatch_requested",
            "job_id": job.id,
            "stage": "teacher_training",
            "run_id": run_id,
        }
        await worker._handle_dispatch(event, entry_id_str="1000-0")

        assert await queue_length(store) == 0


# ---------------------------------------------------------------------------
# Contract absent: pre-approval planning phase
# ---------------------------------------------------------------------------


class TestNoContract:
    @pytest.mark.asyncio
    async def test_dispatch_allowed_without_contract(self, redis_async, run_id, store, job, redis_sync):
        """No contract = pre-approval phase. All dispatches are allowed."""
        sm = StateMachine(run_id=run_id, redis_async=redis_async)
        await sm.initialize()
        worker = ExecutorWorker(redis_async, run_id, store=store, contract=None)

        event = {
            "type": "dispatch_requested",
            "job_id": job.id,
            "stage": "teacher_training",
            "run_id": run_id,
        }
        await worker._handle_dispatch(event, entry_id_str="1000-0")

        dispatched = read_events(redis_sync, run_id, "stage_dispatched")
        assert len(dispatched) == 1

    @pytest.mark.asyncio
    async def test_set_contract_activates_validation(
        self, redis_async, run_id, store, job, redis_sync, contract
    ):
        """set_contract() must take effect on the next dispatch."""
        sm = StateMachine(run_id=run_id, redis_async=redis_async)
        await sm.initialize()
        # Exhaust budget (see comment in TestDispatchRejected for why direct write)
        await redis_async.hset(sm._key, "retry_count", "2")

        worker = ExecutorWorker(redis_async, run_id, store=store, contract=None)
        # Before set_contract: no constraint, dispatch allowed
        event = {
            "type": "dispatch_requested",
            "job_id": job.id,
            "stage": "teacher_training",
            "run_id": run_id,
        }
        await worker._handle_dispatch(event, entry_id_str="1000-0")
        assert len(read_events(redis_sync, run_id, "stage_dispatched")) == 1

        # Now set contract with exhausted budget -- next dispatch should be rejected
        worker.set_contract(contract)

        # Create a fresh job for the second dispatch
        job2 = Job(type=JobType.TEACHER_TRAINING.value, config={})
        await store.store_job(job2)
        event2 = {
            "type": "dispatch_requested",
            "job_id": job2.id,
            "stage": "teacher_training",
            "run_id": run_id,
        }
        await worker._handle_dispatch(event2, entry_id_str="2000-0")

        rejected = read_events(redis_sync, run_id, "dispatch_rejected")
        assert len(rejected) == 1
        assert "Retry budget exhausted" in rejected[0]["errors"][0]
