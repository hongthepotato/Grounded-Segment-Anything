"""
Parity contract between RedisJobStore (sync) and AsyncRedisJobStore (async).

WHY THIS FILE EXISTS
--------------------
The two stores are not independent systems. They are two doors into ONE Redis
database: ml_engine/jobs/worker.py (a standalone sync process) writes job state
that the FastAPI routes (async) read back from the same keyspace, often within
milliseconds. So a write path that exists in one store and not the other does
not merely make that store incomplete -- it corrupts what the OTHER store
reports about data it can plainly see.

That failure mode is not hypothetical. Two real divergences reached main:

  1. count_jobs existed only on the async store. The sync manager emulated it
     with len(list_jobs(limit=10000)), which reported exactly 10000 above the
     cap. The async docstring even records the fix as "TODO-6" -- it was found
     once, fixed on one side, and never mirrored.
  2. The jobs:by_status:* index was maintained by the async store on enqueue,
     update AND delete, but by the sync store only on update. The async store
     reads that index as ground truth (SCARD / SMEMBERS, no fallback scan), so
     a job created by the sync store was invisible to async status queries and
     a job deleted by it leaked its index entry forever.

Both are the same bug class: one protocol, two implementations, silent drift.
Fixing instances does not stop the next one. These tests do, by asserting the
contract against BOTH implementations at once.

WHAT IS AND IS NOT THE CONTRACT
-------------------------------
The contract is observable behavior plus resulting Redis state. It is NOT the
code. The two stores are free to differ -- and do -- wherever that is invisible
to a reader of the database:

  free to differ : counting strategy (async SCARD is O(1), sync SCANs),
                   concurrency (asyncio.gather vs a serial loop), whether
                   list_jobs takes a SMEMBERS fast path, logging
  must match     : key names, value serialization, which writes maintain which
                   index, queue ordering semantics

So "sync and async count_jobs run completely different algorithms" is fine and
expected; "they return different numbers for the same data" is the bug.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import fakeredis
import fakeredis.aioredis
import pytest

from ml_engine.jobs.async_redis_store import AsyncRedisJobStore
from ml_engine.jobs.models import Job, JobStatus, JobType
from ml_engine.jobs.redis_store import RedisJobStore

# ---------------------------------------------------------------------------
# Fixtures -- both stores share ONE FakeServer, mirroring production where the
# worker process and the API talk to the same Redis.
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


@pytest.fixture
def raw_sync(fake_server) -> fakeredis.FakeRedis:
    """Bare client for asserting on Redis state directly, bypassing both stores."""
    return fakeredis.FakeRedis(server=fake_server, decode_responses=False)


@pytest.fixture
def sync_store(fake_server) -> RedisJobStore:
    client = fakeredis.FakeRedis(server=fake_server, decode_responses=False)
    with (
        patch("ml_engine.jobs.redis_store.redis.ConnectionPool") as pool_cls,
        patch("ml_engine.jobs.redis_store.redis.Redis") as redis_cls,
    ):
        pool_cls.from_url.return_value = MagicMock()
        redis_cls.return_value = client
        return RedisJobStore("redis://localhost:6379")


@pytest.fixture
def async_store(fake_server) -> AsyncRedisJobStore:
    client = fakeredis.aioredis.FakeRedis(server=fake_server, decode_responses=False)
    return AsyncRedisJobStore(redis_client=client)


@pytest.fixture(params=["sync", "async"])
def store(request, sync_store, async_store):
    """
    The same test body, run once per implementation.

    Parametrizing is the whole point: a test that passes here has been proven
    against BOTH stores, so an invariant cannot be quietly satisfied by only one.
    """
    return sync_store if request.param == "sync" else async_store


async def call(store, method: str, *args, **kwargs):
    """
    Invoke a store method regardless of whether it is sync or async.

    Lets one test body drive both implementations without duplicating it --
    duplicated test bodies drift exactly the way the code under test did.
    """
    result = getattr(store, method)(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def status_key(status: JobStatus) -> str:
    return f"jobs:by_status:{status.value}"


# ---------------------------------------------------------------------------
# Contract 1: every write path maintains the status index
# ---------------------------------------------------------------------------


class TestStatusIndexContract:
    """
    The jobs:by_status:* SETs are read as ground truth by AsyncRedisJobStore
    (SCARD for count_jobs, SMEMBERS for the list_jobs fast path) with no
    fallback scan. Any write path that skips them makes those reads lie.
    """

    @pytest.mark.asyncio
    async def test_enqueue_indexes_the_job(self, store, raw_sync):
        job = Job(type=JobType.AUTO_LABEL.value)
        await call(store, "enqueue_job", job)
        assert raw_sync.sismember(status_key(job.status), job.id)

    @pytest.mark.asyncio
    async def test_store_job_indexes_the_job(self, store, raw_sync):
        """store_job is the queue-less Coordinator path; it persists a job, so it indexes."""
        job = Job(type=JobType.AUTO_LABEL.value)
        await call(store, "store_job", job)
        assert raw_sync.sismember(status_key(job.status), job.id)

    @pytest.mark.asyncio
    async def test_status_transition_moves_the_entry(self, store, raw_sync):
        job = Job(type=JobType.AUTO_LABEL.value)
        await call(store, "enqueue_job", job)
        await call(store, "update_job", job.id, status=JobStatus.RUNNING)

        assert raw_sync.sismember(status_key(JobStatus.RUNNING), job.id)
        assert not raw_sync.sismember(status_key(JobStatus.PENDING), job.id), (
            "stale entry left in the old status set -- counts for that status read high forever"
        )

    @pytest.mark.asyncio
    async def test_delete_removes_the_entry(self, store, raw_sync):
        job = Job(type=JobType.AUTO_LABEL.value)
        await call(store, "enqueue_job", job)
        await call(store, "delete_job", job.id)

        assert not raw_sync.sismember(status_key(job.status), job.id), (
            "index entry outlived the job hash. A stale list entry self-heals "
            "(get_job returns None and list_jobs drops it) but a stale SCARD "
            "never does -- the count is permanently wrong."
        )

    @pytest.mark.asyncio
    async def test_index_is_exact_after_churn(self, store, raw_sync):
        """The index must equal reality after a realistic mix of writes."""
        jobs = [Job(type=JobType.AUTO_LABEL.value) for _ in range(5)]
        for j in jobs:
            await call(store, "enqueue_job", j)
        await call(store, "update_job", jobs[0].id, status=JobStatus.RUNNING)
        await call(store, "update_job", jobs[1].id, status=JobStatus.COMPLETED)
        await call(store, "delete_job", jobs[2].id)

        assert raw_sync.scard(status_key(JobStatus.PENDING)) == 2
        assert raw_sync.scard(status_key(JobStatus.RUNNING)) == 1
        assert raw_sync.scard(status_key(JobStatus.COMPLETED)) == 1


# ---------------------------------------------------------------------------
# Contract 0: the two stores expose the same callable surface
# ---------------------------------------------------------------------------


class TestSignatureParity:
    """
    Behavioural tests cannot catch a method that one store simply does not have,
    or a parameter only one store accepts -- a test calling the common subset
    passes on both while callers written against one store crash on the other.

    This caught a real regression while it was still staged: count_jobs had
    grown a ``job_type`` filter on the sync store only, so
    ``count_jobs(status=..., job_type=...)`` raised TypeError on the async one.
    """

    # Methods both stores must expose identically.
    #
    # NOT listed, deliberately: dequeue_job. It is a blocking BLPOP pull and a
    # WORKER concern -- only ml_engine/jobs/worker.py (a sync process) consumes
    # the queue, while the async side only ever submits. Its absence from the
    # async store is a design decision, not drift, so demanding parity here
    # would be forcing an unused blocking-pull API onto the API-side store.
    SHARED_METHODS = [
        "enqueue_job",
        "store_job",
        "enqueue_by_id",
        "requeue_job",
        "remove_from_queue",
        "get_job",
        "update_job",
        "list_jobs",
        "count_jobs",
        "delete_job",
        "get_queue_length",
        "publish_event",
        "register_worker",
        "unregister_worker",
        "get_worker",
        "list_workers",
        "cleanup_stale_workers",
    ]

    @pytest.mark.parametrize("method", SHARED_METHODS)
    def test_method_exists_on_both(self, method, sync_store, async_store):
        assert hasattr(sync_store, method), f"sync store is missing {method}"
        assert hasattr(async_store, method), f"async store is missing {method}"

    @pytest.mark.parametrize("method", SHARED_METHODS)
    def test_parameter_names_match(self, method, sync_store, async_store):
        """
        Same parameter NAMES in the same order. Types and defaults may differ
        (async count_jobs accepts ``JobStatus | str``, sync takes ``JobStatus``)
        -- that is an implementation detail. A caller's keyword arguments are not.
        """

        def params(store) -> list:
            sig = inspect.signature(getattr(store, method))
            return [
                name for name, p in sig.parameters.items() if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            ]

        assert params(sync_store) == params(async_store), (
            f"{method} takes different arguments on each store; code written "
            f"against one will fail against the other"
        )


# ---------------------------------------------------------------------------
# Contract 2: counting agrees, however it is implemented
# ---------------------------------------------------------------------------


class TestCountContract:
    """
    Both stores must return the same number for the same data. They are NOT
    required to compute it the same way -- async answers with an O(1) SCARD,
    sync SCANs the keyspace. Only the answer is contractual.
    """

    @pytest.mark.asyncio
    async def test_count_matches_reality(self, store):
        for _ in range(3):
            await call(store, "enqueue_job", Job(type=JobType.AUTO_LABEL.value))
        running = Job(type=JobType.AUTO_LABEL.value)
        await call(store, "enqueue_job", running)
        await call(store, "update_job", running.id, status=JobStatus.RUNNING)

        assert await call(store, "count_jobs", JobStatus.PENDING) == 3
        assert await call(store, "count_jobs", JobStatus.RUNNING) == 1
        assert await call(store, "count_jobs", JobStatus.FAILED) == 0

    @pytest.mark.asyncio
    async def test_count_is_not_capped_by_a_page_limit(self, store):
        """
        TRAP for the bug this suite was written after: get_job_count was
        len(list_jobs(limit=10000)). Any cap-based count is wrong; 150 jobs with
        a default page size of 100 is enough to expose one without needing
        10000 fixtures.
        """
        for _ in range(150):
            await call(store, "enqueue_job", Job(type=JobType.AUTO_LABEL.value))

        assert len(await call(store, "list_jobs", JobStatus.PENDING)) == 100  # paged
        assert await call(store, "count_jobs", JobStatus.PENDING) == 150  # counted

    @pytest.mark.asyncio
    async def test_count_survives_delete(self, store):
        jobs = [Job(type=JobType.AUTO_LABEL.value) for _ in range(4)]
        for j in jobs:
            await call(store, "enqueue_job", j)
        await call(store, "delete_job", jobs[0].id)

        assert await call(store, "count_jobs", JobStatus.PENDING) == 3


# ---------------------------------------------------------------------------
# Contract 3: cross-store -- one store's writes are visible to the other
# ---------------------------------------------------------------------------


class TestCrossStoreVisibility:
    """
    The tests that actually model production: the sync worker process and the
    async API sharing one Redis. Everything above checks each store against
    itself; these check them against EACH OTHER, which is where drift bites.
    """

    @pytest.mark.asyncio
    async def test_async_counts_jobs_created_by_sync(self, sync_store, async_store):
        """
        Async count_jobs answers from the status index (SCARD). If the sync
        store creates a job without indexing it, the API reports a backlog it
        cannot see -- jobs plainly present in Redis, counted as zero.
        """
        for _ in range(3):
            sync_store.enqueue_job(Job(type=JobType.AUTO_LABEL.value))

        assert await async_store.count_jobs(JobStatus.PENDING) == 3

    @pytest.mark.asyncio
    async def test_async_lists_jobs_created_by_sync(self, sync_store, async_store):
        """Async list_jobs(status=...) uses the SMEMBERS fast path, same exposure."""
        sync_store.enqueue_job(Job(type=JobType.AUTO_LABEL.value))
        created = Job(type=JobType.TEACHER_TRAINING.value)
        sync_store.enqueue_job(created)

        listed = await async_store.list_jobs(status=JobStatus.PENDING)
        assert created.id in {j.id for j in listed}

    @pytest.mark.asyncio
    async def test_async_count_drops_jobs_deleted_by_sync(self, sync_store, async_store):
        job = Job(type=JobType.AUTO_LABEL.value)
        sync_store.enqueue_job(job)
        sync_store.delete_job(job.id)

        assert await async_store.count_jobs(JobStatus.PENDING) == 0, (
            "sync delete left the ID in the status index, so the async SCARD "
            "still counts a job that no longer exists"
        )

    @pytest.mark.asyncio
    async def test_sync_counts_jobs_created_by_async(self, sync_store, async_store):
        for _ in range(3):
            await async_store.enqueue_job(Job(type=JobType.AUTO_LABEL.value))

        assert sync_store.count_jobs(status=JobStatus.PENDING) == 3

    @pytest.mark.asyncio
    async def test_sync_and_async_agree_after_interleaved_writes(self, sync_store, async_store):
        """
        The end-to-end shape: API creates, worker transitions, API reads back.
        Both stores must report the same numbers throughout.
        """
        jobs = [Job(type=JobType.AUTO_LABEL.value) for _ in range(4)]
        for j in jobs[:2]:
            await async_store.enqueue_job(j)  # API submits
        for j in jobs[2:]:
            sync_store.enqueue_job(j)  # worker-side creation

        sync_store.update_job(jobs[0].id, status=JobStatus.RUNNING)  # worker picks up
        await async_store.update_job(jobs[3].id, status=JobStatus.COMPLETED)

        for status in (JobStatus.PENDING, JobStatus.RUNNING, JobStatus.COMPLETED):
            assert sync_store.count_jobs(status=status) == await async_store.count_jobs(status), (
                f"sync and async disagree on the number of {status.value} jobs"
            )

    @pytest.mark.asyncio
    async def test_serialization_round_trips_across_stores(self, sync_store, async_store):
        """A job written by one store must read back identically through the other."""
        job = Job(type=JobType.TEACHER_TRAINING.value, config={"lr": 0.001, "epochs": 10}, priority=3)
        sync_store.enqueue_job(job)

        via_async = await async_store.get_job(job.id)
        assert via_async is not None
        assert via_async.id == job.id
        assert via_async.type == job.type
        assert via_async.status == job.status
        assert via_async.priority == job.priority
        assert via_async.config == job.config

    @pytest.mark.asyncio
    async def test_both_stores_write_the_same_key_shape(self, sync_store, async_store, raw_sync):
        """
        Key LAYOUT is contractual -- it is the shared schema. If one store
        invented a different key name, every cross-store read would silently
        miss, which no single-store test could detect.
        """
        sync_job = Job(type=JobType.AUTO_LABEL.value)
        async_job = Job(type=JobType.AUTO_LABEL.value)
        sync_store.enqueue_job(sync_job)
        await async_store.enqueue_job(async_job)

        def shape(job_id: str) -> set:
            return {
                k.decode().replace(job_id, "{id}")
                for k in raw_sync.keys("*")
                if job_id in k.decode() or not k.decode().startswith("job:")
            }

        assert shape(sync_job.id) == shape(async_job.id)
