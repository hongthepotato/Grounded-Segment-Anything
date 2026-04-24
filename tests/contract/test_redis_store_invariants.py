"""
Contract tests for ml_engine.jobs.redis_store.RedisJobStore.

Named regression tests for the 2 QA issues the QA skill caught in prod
(ISSUE-001, ISSUE-002) plus concurrent-write invariants that were deferred
as a TODO but moved in-scope during /plan-eng-review (amendment #16).

These use fakeredis the same way the existing
tests/unit/ml_engine/test_redis_job_store.py does — fakeredis.FakeRedis is
drop-in for redis.Redis, patched into redis_store.py at construction time.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import fakeredis
import pytest

from ml_engine.jobs.models import Job, JobStatus, JobType
from ml_engine.jobs.redis_store import RedisJobStore

# ============================================================================
# Fixtures — each test gets its own FakeServer so concurrent-update tests
# don't bleed state across tests. Within a single test, sync Redis operations
# go through one server, matching production semantics.
# ============================================================================


@pytest.fixture
def fake_server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


@pytest.fixture
def fake_redis(fake_server: fakeredis.FakeServer) -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis(server=fake_server, decode_responses=False)


@pytest.fixture
def store(fake_redis: fakeredis.FakeRedis) -> RedisJobStore:
    """RedisJobStore wired to a FakeRedis instance (bypasses real Redis)."""
    with (
        patch("ml_engine.jobs.redis_store.redis.ConnectionPool") as mock_pool_cls,
        patch("ml_engine.jobs.redis_store.redis.Redis") as mock_redis_cls,
    ):
        mock_pool = MagicMock()
        mock_pool_cls.from_url.return_value = mock_pool
        mock_redis_cls.return_value = fake_redis
        return RedisJobStore("redis://localhost:6379")


# ============================================================================
# ISSUE-001 regression: sync update_job must maintain jobs:by_status:* index.
# Pre-fix symptom: GET /api/jobs?status=running returned empty and
# job_counts.running reported 0 even with active jobs.
# ============================================================================


class TestIssue001StatusIndexMaintainedOnUpdate:
    """Regression for commit fcfbcbd."""

    def test_status_index_moves_on_status_change(self, store: RedisJobStore, fake_redis) -> None:
        job = Job(type=JobType.TEACHER_TRAINING.value, config={"x": 1})
        store.enqueue_job(job)

        # Before transition: nothing in status indices.
        pending_set = fake_redis.smembers("jobs:by_status:pending")
        running_set = fake_redis.smembers("jobs:by_status:running")
        assert pending_set == set()
        assert running_set == set()

        # Transition to RUNNING via update_job (the code path the sync worker uses).
        store.update_job(job.id, status=JobStatus.RUNNING)

        running_set = fake_redis.smembers("jobs:by_status:running")
        assert job.id.encode() in running_set, (
            "update_job did not add job to jobs:by_status:running. "
            "ISSUE-001 regression — filtered job queries will silently return empty."
        )

    def test_status_index_removes_from_old_on_transition(self, store: RedisJobStore, fake_redis) -> None:
        job = Job(type=JobType.TEACHER_TRAINING.value, config={"x": 1})
        store.enqueue_job(job)

        store.update_job(job.id, status=JobStatus.RUNNING)
        store.update_job(job.id, status=JobStatus.COMPLETED)

        running_set = fake_redis.smembers("jobs:by_status:running")
        completed_set = fake_redis.smembers("jobs:by_status:completed")

        assert job.id.encode() not in running_set, (
            "update_job to COMPLETED left job in jobs:by_status:running. "
            "ISSUE-001 regression — stale entries accumulate in status indices."
        )
        assert job.id.encode() in completed_set

    def test_update_without_status_change_leaves_index_alone(self, store: RedisJobStore, fake_redis) -> None:
        job = Job(type=JobType.TEACHER_TRAINING.value, config={"x": 1})
        store.enqueue_job(job)
        store.update_job(job.id, status=JobStatus.RUNNING)

        # Update a non-status field — status indices must stay intact.
        store.update_job(job.id, error_message="nothing bad yet")

        running_set = fake_redis.smembers("jobs:by_status:running")
        assert job.id.encode() in running_set


# ============================================================================
# ISSUE-002 regression: cancelled pending jobs must be removed from the
# job_queue LIST, not just marked cancelled in the hash.
# Pre-fix symptom: queue_length included cancelled jobs and the worker
# could dequeue and attempt to start one.
# ============================================================================


class TestIssue002CancelledJobRemovedFromQueue:
    """Regression for commit 57cd8b0."""

    def test_remove_from_queue_removes_job_id(self, store: RedisJobStore, fake_redis) -> None:
        job = Job(type=JobType.TEACHER_TRAINING.value, config={"x": 1})
        store.enqueue_job(job)
        assert store.get_queue_length() == 1

        removed = store.remove_from_queue(job.id)

        assert removed is True
        assert store.get_queue_length() == 0
        # Concretely: the LIST must not contain the id.
        queue = fake_redis.lrange("job_queue", 0, -1)
        assert job.id.encode() not in queue

    def test_remove_from_queue_is_noop_for_unknown_id(self, store: RedisJobStore) -> None:
        # Nothing to remove — method must not crash.
        removed = store.remove_from_queue("nonexistent-job-id")
        assert removed is False

    def test_remove_from_queue_leaves_other_jobs(self, store: RedisJobStore) -> None:
        j1 = Job(type=JobType.TEACHER_TRAINING.value, config={"x": 1})
        j2 = Job(type=JobType.TEACHER_TRAINING.value, config={"x": 2})
        store.enqueue_job(j1)
        store.enqueue_job(j2)

        store.remove_from_queue(j1.id)

        assert store.get_queue_length() == 1
        next_id = store.dequeue_job(timeout=0)
        assert next_id == j2.id


# ============================================================================
# Concurrent update invariants — amendment #16 (moved in-scope from TODOs).
# Uses threading since the sync RedisJobStore is documented as thread-safe.
# ============================================================================


class TestConcurrentUpdatesPreserveStatusIndex:
    """Two threads updating the same job's status concurrently: the final
    status index state must be consistent (one of the two terminal statuses,
    not a stuck intermediate)."""

    def test_two_concurrent_updates_settle_to_one_state(self, store: RedisJobStore, fake_redis) -> None:
        job = Job(type=JobType.TEACHER_TRAINING.value, config={"x": 1})
        store.enqueue_job(job)
        store.update_job(job.id, status=JobStatus.RUNNING)  # known start state

        final_statuses = {"completed": JobStatus.COMPLETED, "failed": JobStatus.FAILED}

        def _update(status: JobStatus) -> None:
            store.update_job(job.id, status=status)

        threads = [
            threading.Thread(target=_update, args=(final_statuses["completed"],)),
            threading.Thread(target=_update, args=(final_statuses["failed"],)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # Invariant: the job must be in exactly ONE status index after the dust settles.
        running_set = fake_redis.smembers("jobs:by_status:running")
        completed_set = fake_redis.smembers("jobs:by_status:completed")
        failed_set = fake_redis.smembers("jobs:by_status:failed")

        found_in = []
        if job.id.encode() in running_set:
            found_in.append("running")
        if job.id.encode() in completed_set:
            found_in.append("completed")
        if job.id.encode() in failed_set:
            found_in.append("failed")

        # Must be in exactly one terminal state, not in running (the start),
        # and not duplicated across multiple indices.
        assert len(found_in) == 1, (
            f"Concurrent updates left job in multiple status indices: {found_in}. "
            f"This is the status-index-stale bug that ISSUE-001 was supposed to prevent."
        )
        assert found_in[0] in ("completed", "failed"), (
            f"Concurrent updates did not reach a final state: job is in {found_in[0]!r}"
        )


class TestPipelinedTransactionFailureDoesNotCorruptState:
    """If a pipelined transaction fails mid-execution, the store must not leak
    partial writes that cause downstream queries to see an inconsistent state.

    fakeredis supports pipelines, so we can inject a failure by transitioning
    through an invalid state name and verify the store's exception handling
    doesn't leave the job in two status sets at once.
    """

    def test_updatejob_failure_does_not_leave_duplicate_status_index_entry(
        self, store: RedisJobStore, fake_redis
    ) -> None:
        job = Job(type=JobType.TEACHER_TRAINING.value, config={"x": 1})
        store.enqueue_job(job)
        store.update_job(job.id, status=JobStatus.RUNNING)

        # Baseline: exactly one entry in "running" index.
        assert job.id.encode() in fake_redis.smembers("jobs:by_status:running")

        # Simulate a mid-pipeline redis failure by patching execute() to raise
        # on the next call.
        real_pipeline = fake_redis.pipeline

        class _FailingPipeline:
            def __init__(self, real):
                self._real = real

            def __getattr__(self, name):
                return getattr(self._real, name)

            def execute(self):
                raise RuntimeError("Simulated Redis failure mid-pipeline")

        def _patched_pipeline(*args, **kwargs):
            return _FailingPipeline(real_pipeline(*args, **kwargs))

        with patch.object(fake_redis, "pipeline", _patched_pipeline):
            with pytest.raises(Exception):
                store.update_job(job.id, status=JobStatus.COMPLETED)

        # After the failure: the job should NOT simultaneously be in running AND
        # completed. Either state is defensible — duplication is not.
        running_set = fake_redis.smembers("jobs:by_status:running")
        completed_set = fake_redis.smembers("jobs:by_status:completed")

        in_running = job.id.encode() in running_set
        in_completed = job.id.encode() in completed_set

        assert not (in_running and in_completed), (
            "Pipeline failure left the job in BOTH running and completed status indices. "
            "Downstream status-filtered queries will see duplicates or inconsistent counts."
        )
