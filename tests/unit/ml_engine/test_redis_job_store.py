"""
Unit tests for ml_engine.jobs.redis_store.RedisJobStore.

Uses fakeredis -- no real Redis needed.
The RedisJobStore constructor calls redis.ping(), so we patch the connection
to use a FakeRedis instance.
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import fakeredis
import pytest

from ml_engine.jobs.models import Job, JobStatus, JobProgress, JobType
from ml_engine.jobs.redis_store import RedisJobStore


# ---------------------------------------------------------------------------
# Fixture: store backed by fakeredis
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_redis():
    r"""eturns a fakeredis instance for testing (bypasses real Redis connection)."""
    return fakeredis.FakeRedis(decode_responses=False)


@pytest.fixture
def store(fake_redis):
    """RedisJobStore wired to a FakeRedis instance (bypasses real connection)."""
    with patch("ml_engine.jobs.redis_store.redis.ConnectionPool") as mock_pool_cls, \
         patch("ml_engine.jobs.redis_store.redis.Redis") as mock_redis_cls:

        mock_pool = MagicMock()
        mock_pool_cls.from_url.return_value = mock_pool
        mock_redis_cls.return_value = fake_redis

        s = RedisJobStore("redis://localhost:6379")
    return s


# ---------------------------------------------------------------------------
# enqueue_job
# ---------------------------------------------------------------------------

class TestEnqueueJob:
    r"""Tests for enqueue_job (one-phase dispatch)."""
    def test_enqueue_stores_job(self, store, fake_redis):
        r"""Test that enqueue_job persists the job to Redis (not just the ID in the queue)."""
        job = Job(type=JobType.TEACHER_TRAINING.value, config={"batch_size": 8})
        store.enqueue_job(job)
        stored = store.get_job(job.id)
        assert stored is not None
        assert stored.id == job.id

    def test_enqueue_adds_to_queue(self, store, fake_redis):
        r"""Test that enqueue_job adds the job ID to the queue."""
        job = Job()
        store.enqueue_job(job)
        assert store.get_queue_length() == 1

    def test_enqueue_multiple_fifo_order(self, store, fake_redis):
        r"""Test that multiple enqueued jobs come out in FIFO order (if same priority)."""
        j1 = Job()
        j2 = Job()
        store.enqueue_job(j1)
        store.enqueue_job(j2)
        # FIFO: j1 should be at front
        first_id = store.dequeue_job(timeout=0)
        assert first_id == j1.id

    def test_high_priority_job_goes_to_front(self, store, fake_redis):
        r"""Test that a higher priority job jumps ahead of lower priority ones in the queue."""
        j_normal = Job(priority=0)
        j_high = Job(priority=5)
        store.enqueue_job(j_normal)
        store.enqueue_job(j_high)
        first_id = store.dequeue_job(timeout=0)
        assert first_id == j_high.id

    def test_config_preserved(self, store, fake_redis):
        r"""Test that the job config is correctly stored and retrieved (not just the ID)."""
        job = Job(config={"lr": 0.001, "epochs": 10})
        store.enqueue_job(job)
        stored = store.get_job(job.id)
        assert stored.config["lr"] == pytest.approx(0.001)
        assert stored.config["epochs"] == 10


# ---------------------------------------------------------------------------
# store_job / enqueue_by_id  (two-phase dispatch)
# ---------------------------------------------------------------------------

class TestStoreJobEnqueueById:
    r"""Tests for store_job + enqueue_by_id (two-phase dispatch)."""
    def test_store_job_does_not_add_to_queue(self, store, fake_redis):
        r"""Test that store_job alone does not add the job ID to the queue."""
        job = Job()
        store.store_job(job)
        assert store.get_queue_length() == 0

    def test_store_job_persists_to_redis(self, store, fake_redis):
        r"""Test that store_job persists the job to Redis."""
        job = Job(config={"x": 1})
        store.store_job(job)
        stored = store.get_job(job.id)
        assert stored is not None
        assert stored.config["x"] == 1

    def test_enqueue_by_id_adds_to_queue(self, store, fake_redis):
        r"""Test that enqueue_by_id adds the job ID to the queue."""
        job = Job()
        store.store_job(job)
        result = store.enqueue_by_id(job.id)
        assert result is True
        assert store.get_queue_length() == 1

    def test_enqueue_by_id_unknown_returns_false(self, store, fake_redis):
        r"""Test that enqueue_by_id returns False if the job ID is not found in Redis."""
        result = store.enqueue_by_id("nonexistent-id")
        assert result is False

    def test_enqueue_by_id_high_priority_goes_to_front(self, store, fake_redis):
        r"""Test that enqueue_by_id respects job priority when adding to the queue."""
        j_normal = Job(priority=0)
        j_high = Job(priority=5)
        store.store_job(j_normal)
        store.store_job(j_high)
        store.enqueue_by_id(j_normal.id)
        store.enqueue_by_id(j_high.id)
        first_id = store.dequeue_job(timeout=0)
        assert first_id == j_high.id


# ---------------------------------------------------------------------------
# get_job
# ---------------------------------------------------------------------------

class TestGetJob:
    r"""Tests for get_job."""
    def test_returns_none_for_unknown(self, store, fake_redis):
        r"""Test that get_job returns None if the job ID is not found in Redis."""
        assert store.get_job("no-such-job") is None

    def test_returns_job_after_enqueue(self, store, fake_redis):
        r"""Test that get_job returns the correct Job object after enqueueing."""
        job = Job(type=JobType.EXPERIMENT_LOOP.value)
        store.enqueue_job(job)
        stored = store.get_job(job.id)
        assert stored.type == JobType.EXPERIMENT_LOOP.value

    def test_status_deserialized_correctly(self, store, fake_redis):
        r"""Test that the JobStatus enum is correctly deserialized from Redis."""
        job = Job(status=JobStatus.PENDING)
        store.enqueue_job(job)
        stored = store.get_job(job.id)
        assert stored.status == JobStatus.PENDING


# ---------------------------------------------------------------------------
# update_job
# ---------------------------------------------------------------------------

class TestUpdateJob:
    r"""Tests for update_job."""
    def test_update_status(self, store, fake_redis):
        r"""Test that update_job can update the job status and it is correctly stored in Redis."""
        job = Job()
        store.enqueue_job(job)
        store.update_job(job.id, status=JobStatus.RUNNING)
        stored = store.get_job(job.id)
        assert stored.status == JobStatus.RUNNING

    def test_update_progress(self, store, fake_redis):
        r"""Test that update_job can update the job progress and it is correctly stored in Redis."""
        job = Job()
        store.enqueue_job(job)
        progress = JobProgress(current_epoch=3, total_epochs=10)
        updated_progress = JobProgress(current_epoch=6, total_epochs=10)
        store.update_job(job.id, progress=updated_progress)
        stored = store.get_job(job.id)
        assert stored.progress is not None
        assert stored.progress.current_epoch == 6

    def test_update_error_message(self, store, fake_redis):
        r"""Test that update_job can update the error_message and it is correctly stored in Redis."""
        job = Job()
        store.enqueue_job(job)
        store.update_job(job.id, status=JobStatus.FAILED, error_message="OOM")
        stored = store.get_job(job.id)
        assert stored.error_message == "OOM"

    def test_update_none_value_stored_as_empty(self, store, fake_redis):
        r"""Test that updating a field to None results in it being stored as empty."""
        job = Job()
        store.enqueue_job(job)
        store.update_job(job.id, worker_id=None)
        stored = store.get_job(job.id)
        assert stored.worker_id is None


# ---------------------------------------------------------------------------
# dequeue_job
# ---------------------------------------------------------------------------

class TestDequeueJob:
    r"""Tests for dequeue_job."""
    def test_dequeue_returns_none_when_empty(self, store, fake_redis):
        r"""Test that dequeue_job returns None if the queue is empty."""
        result = store.dequeue_job(timeout=1)
        assert result is None

    def test_dequeue_returns_job_id(self, store, fake_redis):
        r"""Test that dequeue_job returns the correct job ID when a job is enqueued."""
        job = Job()
        store.enqueue_job(job)
        job_id = store.dequeue_job(timeout=0)
        assert job_id == job.id

    def test_dequeue_removes_from_queue(self, store, fake_redis):
        r"""Test that dequeue_job removes the job ID from the queue."""
        job = Job()
        store.enqueue_job(job)
        store.dequeue_job(timeout=0)
        assert store.get_queue_length() == 0


# ---------------------------------------------------------------------------
# get_queue_length
# ---------------------------------------------------------------------------

class TestGetQueueLength:
    r"""Tests for get_queue_length."""
    def test_zero_when_empty(self, store, fake_redis):
        r"""Test that get_queue_length returns 0 when the queue is empty."""
        assert store.get_queue_length() == 0

    def test_increments_on_enqueue(self, store, fake_redis):
        r"""Test that get_queue_length increments correctly when jobs are enqueued."""
        for _ in range(5):
            store.enqueue_job(Job())
        assert store.get_queue_length() == 5

    def test_decrements_on_dequeue(self, store, fake_redis):
        r"""Test that get_queue_length decrements correctly when jobs are dequeued."""
        store.enqueue_job(Job())
        store.enqueue_job(Job())
        store.dequeue_job(timeout=0)
        assert store.get_queue_length() == 1
