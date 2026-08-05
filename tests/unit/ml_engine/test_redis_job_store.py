"""
Unit tests for ml_engine.jobs.redis_store.RedisJobStore.

Uses fakeredis -- no real Redis needed.
The RedisJobStore constructor calls redis.ping(), so we patch the connection
to use a FakeRedis instance.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import fakeredis
import pytest
from redis.exceptions import RedisError

from ml_engine.jobs.models import Job, JobProgress, JobStatus, JobType
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
    with (
        patch("ml_engine.jobs.redis_store.redis.ConnectionPool") as mock_pool_cls,
        patch("ml_engine.jobs.redis_store.redis.Redis") as mock_redis_cls,
    ):
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


# ---------------------------------------------------------------------------
# count_jobs
# ---------------------------------------------------------------------------


class TestCountJobs:
    r"""Tests for count_jobs (uncapped counting without materializing Jobs)."""

    def test_zero_when_empty(self, store):
        r"""Test that count_jobs returns 0 against an empty store."""
        assert store.count_jobs() == 0

    def test_counts_all_jobs_without_filter(self, store):
        r"""Test that an unfiltered count returns every stored job."""
        for _ in range(7):
            store.enqueue_job(Job())
        assert store.count_jobs() == 7

    def test_status_filter(self, store):
        r"""Test that a status filter counts only jobs in that status."""
        for _ in range(3):
            store.enqueue_job(Job())
        running = Job()
        store.enqueue_job(running)
        store.update_job(running.id, status=JobStatus.RUNNING)

        assert store.count_jobs(status=JobStatus.RUNNING) == 1
        assert store.count_jobs(status=JobStatus.PENDING) == 3
        assert store.count_jobs(status=JobStatus.FAILED) == 0

    def test_job_with_no_status_field_is_not_counted(self, store, fake_redis):
        r"""
        Test that a job key whose hash lacks a status field is skipped.

        Every Job.to_dict() writes a status, so such a key is corrupt (or was
        partially written). Counting it would inflate the total with something
        that can never match a status filter.
        """
        store.enqueue_job(Job())
        fake_redis.hset("job:corrupt", "type", "auto_label")  # no status field

        assert store.count_jobs() == 1

    def test_ignores_non_job_keys(self, store, fake_redis):
        r"""Test that queue/worker/index keys are not miscounted as jobs."""
        store.enqueue_job(Job())  # also writes the job_queue LIST
        fake_redis.hset("worker:w1", "status", "idle")
        fake_redis.sadd("jobs:by_status:pending", "not-a-job")
        assert store.count_jobs() == 1

    def test_not_capped_by_a_page_limit(self, store):
        r"""
        Test that count_jobs reports the true total past a page boundary.

        TRAP for the bug this method replaces: JobManager.get_job_count was
        len(list_jobs(limit=10000)), so any status above the limit reported the
        limit. Here list_jobs(limit=100) undercounts by construction while
        count_jobs must not.
        """
        for _ in range(150):
            store.enqueue_job(Job())

        assert len(store.list_jobs(limit=100)) == 100  # the capped, wrong way
        assert store.count_jobs() == 150  # the uncapped, right way

    def test_reads_are_batched_not_one_round_trip_per_job(self, store, fake_redis):
        r"""
        Test that field reads are pipelined per SCAN batch.

        This is the perf property the method exists for, asserted as a call
        pattern rather than as wall-clock (fakeredis is in-process, so timing
        here would measure nothing and flake on CI). A regression to a per-key
        ``hmget`` would issue one round-trip per job on real Redis.
        """
        for _ in range(150):
            store.enqueue_job(Job())

        spy = MagicMock(wraps=fake_redis)
        store.redis = spy
        assert store.count_jobs() == 150

        # No un-batched per-key reads at all...
        assert spy.hmget.call_count == 0
        # ...and far fewer batches than jobs (150 jobs must not cost 150 trips).
        assert spy.pipeline.call_count <= 10

    def test_duplicate_scan_results_do_not_inflate_the_count(self, store, fake_redis):
        r"""
        Test that a key returned twice by SCAN is counted once.

        Redis SCAN guarantees at-least-once, not exactly-once: a key present for
        the whole iteration is returned at least once, but rehashing mid-scan can
        return it again. Without de-duplication the count silently inflates -- a
        counter that reads high is worse than one that is merely slow, because
        nothing downstream can detect it. Here SCAN is forced to repeat its
        batch, so an un-deduped implementation reports 4 instead of 2.
        """
        j1, j2 = Job(), Job()
        store.enqueue_job(j1)
        store.enqueue_job(j2)
        keys = [f"job:{j1.id}".encode(), f"job:{j2.id}".encode()]

        spy = MagicMock(wraps=fake_redis)
        # First call yields the batch with a live cursor, second repeats it and ends.
        spy.scan.side_effect = [(1, keys), (0, keys)]
        store.redis = spy

        assert store.count_jobs() == 2

    def test_returns_zero_on_redis_error(self, store):
        r"""Test that count_jobs degrades to 0 rather than propagating a RedisError."""
        spy = MagicMock()
        spy.scan.side_effect = RedisError("connection lost")
        store.redis = spy

        assert store.count_jobs() == 0


# ---------------------------------------------------------------------------
# status index maintenance (cross-store invariant)
# ---------------------------------------------------------------------------


class TestStatusIndexMaintenance:
    r"""
    Tests that the sync store maintains jobs:by_status:* on every write.

    The index is not decoration: AsyncRedisJobStore READS it as ground truth --
    count_jobs uses SCARD and list_jobs(status=...) uses SMEMBERS, with no
    fallback scan. Both stores are pointed at the same Redis, and
    AsyncRedisJobStore documents "same Redis layout, same keys, same semantics",
    so a sync write that skips the index makes the async store answer wrongly
    about a job that is really there (or really gone).
    """

    def test_enqueue_indexes_the_job(self, store, fake_redis):
        r"""Test that enqueue_job adds the job to its status index set."""
        job = Job()
        store.enqueue_job(job)
        assert fake_redis.sismember(f"jobs:by_status:{job.status.value}", job.id)

    def test_store_job_indexes_the_job(self, store, fake_redis):
        r"""Test that store_job (queue-less Coordinator path) also indexes."""
        job = Job()
        store.store_job(job)
        assert fake_redis.sismember(f"jobs:by_status:{job.status.value}", job.id)

    def test_delete_removes_the_index_entry(self, store, fake_redis):
        r"""
        Test that delete_job removes the ID from the status index.

        Without this the entry outlives the hash, so an async SCARD counts a job
        that no longer exists -- and unlike a stale list entry (which get_job
        filters out by returning None) an inflated count never self-corrects.
        """
        job = Job()
        store.enqueue_job(job)
        store.delete_job(job.id)
        assert not fake_redis.sismember(f"jobs:by_status:{job.status.value}", job.id)
        assert fake_redis.scard(f"jobs:by_status:{job.status.value}") == 0

    def test_index_scard_agrees_with_count_jobs(self, store, fake_redis):
        r"""
        Test that the index SCARD matches the SCAN-based count after churn.

        This is the invariant that lets the async store answer with SCARD and the
        sync store answer with a SCAN and get the same number. Exercises all
        three writers -- enqueue, update (status transition), delete.
        """
        jobs = [Job() for _ in range(5)]
        for j in jobs:
            store.enqueue_job(j)
        store.update_job(jobs[0].id, status=JobStatus.RUNNING)
        store.update_job(jobs[1].id, status=JobStatus.RUNNING)
        store.delete_job(jobs[2].id)

        for status in (JobStatus.PENDING, JobStatus.RUNNING):
            assert fake_redis.scard(f"jobs:by_status:{status.value}") == store.count_jobs(status=status)
        assert store.count_jobs(status=JobStatus.RUNNING) == 2
        assert store.count_jobs(status=JobStatus.PENDING) == 2
