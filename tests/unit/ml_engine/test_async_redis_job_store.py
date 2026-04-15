"""
Unit tests for AsyncRedisJobStore + AsyncJobManager (Phase 8).

Uses the shared ``redis_async`` fixture from conftest so the async store
sees a real async-style Redis (fakeredis.aioredis). No real Redis needed.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest
import pytest_asyncio

from ml_engine.jobs.async_redis_store import AsyncRedisJobStore
from ml_engine.jobs.async_manager import AsyncJobManager
from ml_engine.jobs.models import Job, JobStatus, JobProgress, JobType, WorkerInfo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def store(redis_async):
    """AsyncRedisJobStore backed by shared fakeredis.aioredis client."""
    return AsyncRedisJobStore(redis_client=redis_async)


@pytest_asyncio.fixture
async def manager(redis_async):
    return AsyncJobManager(redis_client=redis_async)


# ---------------------------------------------------------------------------
# enqueue_job
# ---------------------------------------------------------------------------

class TestEnqueueJob:
    @pytest.mark.asyncio
    async def test_enqueue_stores_job(self, store):
        job = Job(type=JobType.TEACHER_TRAINING.value, config={"batch_size": 8})
        await store.enqueue_job(job)
        stored = await store.get_job(job.id)
        assert stored is not None
        assert stored.id == job.id

    @pytest.mark.asyncio
    async def test_enqueue_adds_to_queue(self, store):
        await store.enqueue_job(Job())
        assert await store.get_queue_length() == 1

    @pytest.mark.asyncio
    async def test_config_preserved(self, store):
        job = Job(config={"lr": 0.001, "epochs": 10})
        await store.enqueue_job(job)
        stored = await store.get_job(job.id)
        assert stored.config["lr"] == pytest.approx(0.001)
        assert stored.config["epochs"] == 10


# ---------------------------------------------------------------------------
# store_job / enqueue_by_id
# ---------------------------------------------------------------------------

class TestStoreJobEnqueueById:
    @pytest.mark.asyncio
    async def test_store_job_does_not_add_to_queue(self, store):
        await store.store_job(Job())
        assert await store.get_queue_length() == 0

    @pytest.mark.asyncio
    async def test_store_job_persists_to_redis(self, store):
        job = Job(config={"x": 1})
        await store.store_job(job)
        stored = await store.get_job(job.id)
        assert stored is not None
        assert stored.config["x"] == 1

    @pytest.mark.asyncio
    async def test_enqueue_by_id_adds_to_queue(self, store):
        job = Job()
        await store.store_job(job)
        assert await store.enqueue_by_id(job.id) is True
        assert await store.get_queue_length() == 1

    @pytest.mark.asyncio
    async def test_enqueue_by_id_unknown_returns_false(self, store):
        assert await store.enqueue_by_id("nonexistent-id") is False


# ---------------------------------------------------------------------------
# get_job
# ---------------------------------------------------------------------------

class TestGetJob:
    @pytest.mark.asyncio
    async def test_returns_none_for_unknown(self, store):
        assert await store.get_job("no-such-job") is None

    @pytest.mark.asyncio
    async def test_returns_job_after_enqueue(self, store):
        job = Job(type=JobType.EXPERIMENT_LOOP.value)
        await store.enqueue_job(job)
        stored = await store.get_job(job.id)
        assert stored.type == JobType.EXPERIMENT_LOOP.value

    @pytest.mark.asyncio
    async def test_status_deserialized_correctly(self, store):
        job = Job(status=JobStatus.PENDING)
        await store.enqueue_job(job)
        stored = await store.get_job(job.id)
        assert stored.status == JobStatus.PENDING


# ---------------------------------------------------------------------------
# update_job (and status-index maintenance)
# ---------------------------------------------------------------------------

class TestUpdateJob:
    @pytest.mark.asyncio
    async def test_update_status(self, store):
        job = Job()
        await store.enqueue_job(job)
        await store.update_job(job.id, status=JobStatus.RUNNING)
        stored = await store.get_job(job.id)
        assert stored.status == JobStatus.RUNNING

    @pytest.mark.asyncio
    async def test_update_progress(self, store):
        job = Job()
        await store.enqueue_job(job)
        await store.update_job(job.id, progress=JobProgress(current_epoch=6, total_epochs=10))
        stored = await store.get_job(job.id)
        assert stored.progress is not None
        assert stored.progress.current_epoch == 6

    @pytest.mark.asyncio
    async def test_update_error_message(self, store):
        job = Job()
        await store.enqueue_job(job)
        await store.update_job(job.id, status=JobStatus.FAILED, error_message="OOM")
        stored = await store.get_job(job.id)
        assert stored.error_message == "OOM"

    @pytest.mark.asyncio
    async def test_update_none_value_stored_as_empty(self, store):
        job = Job()
        await store.enqueue_job(job)
        await store.update_job(job.id, worker_id=None)
        stored = await store.get_job(job.id)
        assert stored.worker_id is None

    @pytest.mark.asyncio
    async def test_status_change_moves_index(self, store):
        """Status change should move job id between status-index sets."""
        job = Job(status=JobStatus.PENDING)
        await store.enqueue_job(job)
        assert await store.count_jobs(JobStatus.PENDING) == 1
        assert await store.count_jobs(JobStatus.RUNNING) == 0
        await store.update_job(job.id, status=JobStatus.RUNNING)
        assert await store.count_jobs(JobStatus.PENDING) == 0
        assert await store.count_jobs(JobStatus.RUNNING) == 1


# ---------------------------------------------------------------------------
# count_jobs (O(1) via SCARD -- TODO-6 fix)
# ---------------------------------------------------------------------------

class TestCountJobs:
    @pytest.mark.asyncio
    async def test_count_by_status_uses_index(self, store):
        for _ in range(3):
            await store.enqueue_job(Job(status=JobStatus.PENDING))
        for _ in range(2):
            j = Job(status=JobStatus.PENDING)
            await store.enqueue_job(j)
            await store.update_job(j.id, status=JobStatus.COMPLETED)
        assert await store.count_jobs(JobStatus.PENDING) == 3
        assert await store.count_jobs(JobStatus.COMPLETED) == 2

    @pytest.mark.asyncio
    async def test_delete_removes_from_index(self, store):
        job = Job(status=JobStatus.PENDING)
        await store.enqueue_job(job)
        await store.update_job(job.id, status=JobStatus.COMPLETED)
        assert await store.count_jobs(JobStatus.COMPLETED) == 1
        await store.delete_job(job.id)
        assert await store.count_jobs(JobStatus.COMPLETED) == 0


# ---------------------------------------------------------------------------
# AsyncJobManager high-level API
# ---------------------------------------------------------------------------

class TestAsyncJobManager:
    @pytest.mark.asyncio
    async def test_submit_job_returns_job(self, manager):
        job = await manager.submit_job(
            job_type=JobType.TEACHER_TRAINING.value,
            config={"data_path": "/x", "image_paths": ["a.jpg"]},
        )
        assert job.id
        assert job.status == JobStatus.PENDING

    @pytest.mark.asyncio
    async def test_submit_and_query(self, manager):
        job = await manager.submit_job(
            job_type=JobType.TEACHER_TRAINING.value,
            config={},
        )
        fetched = await manager.get_job(job.id)
        assert fetched.id == job.id

    @pytest.mark.asyncio
    async def test_get_job_count_by_status(self, manager):
        for _ in range(4):
            await manager.submit_job(job_type=JobType.TEACHER_TRAINING.value, config={})
        assert await manager.get_job_count("pending") == 4

    @pytest.mark.asyncio
    async def test_cancel_pending_job(self, manager):
        job = await manager.submit_job(job_type=JobType.TEACHER_TRAINING.value, config={})
        assert await manager.cancel_job(job.id) is True
        fetched = await manager.get_job(job.id)
        assert fetched.status == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_missing_returns_false(self, manager):
        assert await manager.cancel_job("no-such-job") is False

    @pytest.mark.asyncio
    async def test_delete_non_terminal_returns_false(self, manager):
        job = await manager.submit_job(job_type=JobType.TEACHER_TRAINING.value, config={})
        assert await manager.delete_job(job.id) is False

    @pytest.mark.asyncio
    async def test_delete_terminal_returns_true(self, manager):
        job = await manager.submit_job(job_type=JobType.TEACHER_TRAINING.value, config={})
        await manager.cancel_job(job.id)  # -> CANCELLED (terminal)
        assert await manager.delete_job(job.id) is True

    @pytest.mark.asyncio
    async def test_invalid_job_type_raises(self, manager):
        with pytest.raises(ValueError):
            await manager.submit_job(job_type="not_a_real_type", config={})


# ---------------------------------------------------------------------------
# dequeue_job
# ---------------------------------------------------------------------------

class TestQueueOps:
    @pytest.mark.asyncio
    async def test_queue_length_zero_when_empty(self, store):
        assert await store.get_queue_length() == 0

    @pytest.mark.asyncio
    async def test_queue_length_increments(self, store):
        for _ in range(5):
            await store.enqueue_job(Job())
        assert await store.get_queue_length() == 5


# ---------------------------------------------------------------------------
# list_jobs with status filter uses index (O(matching) not O(all))
# ---------------------------------------------------------------------------

class TestListJobs:
    @pytest.mark.asyncio
    async def test_status_filter_returns_only_matching(self, store):
        for _ in range(3):
            await store.enqueue_job(Job(status=JobStatus.PENDING))
        for _ in range(2):
            j = Job(status=JobStatus.PENDING)
            await store.enqueue_job(j)
            await store.update_job(j.id, status=JobStatus.RUNNING)
        running = await store.list_jobs(status=JobStatus.RUNNING)
        assert len(running) == 2
        assert all(j.status == JobStatus.RUNNING for j in running)

    @pytest.mark.asyncio
    async def test_no_filter_returns_all(self, store):
        for _ in range(4):
            await store.enqueue_job(Job())
        all_jobs = await store.list_jobs()
        assert len(all_jobs) == 4

    @pytest.mark.asyncio
    async def test_status_and_type_filter(self, store):
        j1 = Job(type=JobType.TEACHER_TRAINING.value, status=JobStatus.PENDING)
        j2 = Job(type=JobType.EXPERIMENT_LOOP.value, status=JobStatus.PENDING)
        await store.enqueue_job(j1)
        await store.enqueue_job(j2)
        result = await store.list_jobs(
            status=JobStatus.PENDING, job_type=JobType.TEACHER_TRAINING.value
        )
        assert len(result) == 1
        assert result[0].type == JobType.TEACHER_TRAINING.value

    @pytest.mark.asyncio
    async def test_limit_applied(self, store):
        for _ in range(10):
            await store.enqueue_job(Job())
        result = await store.list_jobs(limit=3)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_empty_status_returns_empty(self, store):
        await store.enqueue_job(Job(status=JobStatus.PENDING))
        result = await store.list_jobs(status=JobStatus.RUNNING)
        assert result == []


# ---------------------------------------------------------------------------
# list_workers
# ---------------------------------------------------------------------------

class TestListWorkers:
    @pytest.mark.asyncio
    async def test_no_workers_returns_empty(self, store):
        assert await store.list_workers() == []

    @pytest.mark.asyncio
    async def test_returns_registered_workers(self, store):
        await store.register_worker(WorkerInfo(id="w1"))
        await store.register_worker(WorkerInfo(id="w2"))
        workers = await store.list_workers()
        assert len(workers) == 2
        assert {w.id for w in workers} == {"w1", "w2"}

    @pytest.mark.asyncio
    async def test_status_filter(self, store):
        await store.register_worker(WorkerInfo(id="idle-1", status="idle"))
        await store.register_worker(WorkerInfo(id="busy-1", status="busy"))
        idle = await store.list_workers(status="idle")
        assert len(idle) == 1
        assert idle[0].id == "idle-1"


# ---------------------------------------------------------------------------
# cleanup_stale_workers
# ---------------------------------------------------------------------------

class TestCleanupStaleWorkers:
    @pytest.mark.asyncio
    async def test_removes_stale_worker(self, store):
        old_hb = datetime.now(timezone.utc) - timedelta(seconds=200)
        await store.register_worker(WorkerInfo(id="stale-w", last_heartbeat=old_hb))
        removed = await store.cleanup_stale_workers(timeout_seconds=60)
        assert removed == 1
        assert await store.get_worker("stale-w") is None

    @pytest.mark.asyncio
    async def test_fresh_worker_not_removed(self, store):
        await store.register_worker(WorkerInfo(id="fresh-w"))
        removed = await store.cleanup_stale_workers(timeout_seconds=60)
        assert removed == 0
        assert await store.get_worker("fresh-w") is not None

    @pytest.mark.asyncio
    async def test_stale_worker_with_job_requeues_job(self, store):
        job = Job(status=JobStatus.RUNNING)
        await store.enqueue_job(job)
        # Simulate job popped from queue by worker
        await store.update_job(job.id, status=JobStatus.RUNNING, worker_id="stale-w")
        old_hb = datetime.now(timezone.utc) - timedelta(seconds=200)
        await store.register_worker(
            WorkerInfo(id="stale-w", status="busy", current_job_id=job.id, last_heartbeat=old_hb)
        )
        removed = await store.cleanup_stale_workers(timeout_seconds=60)
        assert removed == 1
        # Job requeued: queue length is 2 (original enqueue + requeue)
        assert await store.get_queue_length() == 2


# ---------------------------------------------------------------------------
# AsyncJobManager: cancel_job CANCELLING and get_queue_status
# ---------------------------------------------------------------------------

class TestAsyncJobManagerExtended:
    @pytest.mark.asyncio
    async def test_cancel_cancelling_job_returns_true(self, manager):
        """cancel_job on an already-CANCELLING job: idempotent, returns True."""
        job = await manager.submit_job(job_type=JobType.TEACHER_TRAINING.value, config={})
        await manager.store.update_job(job.id, status=JobStatus.CANCELLING)
        result = await manager.cancel_job(job.id)
        assert result is True
        # Status must NOT have changed back or forward
        fetched = await manager.get_job(job.id)
        assert fetched.status == JobStatus.CANCELLING

    @pytest.mark.asyncio
    async def test_get_queue_status_returns_counts(self, manager):
        for _ in range(3):
            await manager.submit_job(job_type=JobType.TEACHER_TRAINING.value, config={})
        status = await manager.get_queue_status()
        assert status["queue_length"] == 3
        assert status["job_counts"]["pending"] == 3
        assert status["job_counts"]["running"] == 0
        assert "workers" in status

    @pytest.mark.asyncio
    async def test_count_jobs_total_no_status(self, manager):
        """count_jobs() with no filter returns sum across all statuses."""
        j1 = await manager.submit_job(job_type=JobType.TEACHER_TRAINING.value, config={})
        j2 = await manager.submit_job(job_type=JobType.TEACHER_TRAINING.value, config={})
        await manager.cancel_job(j1.id)
        total = await manager.store.count_jobs()
        # j1 cancelled, j2 pending -- both counted
        assert total == 2
