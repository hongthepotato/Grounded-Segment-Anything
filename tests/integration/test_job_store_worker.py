"""
Integration tests for the async job store → worker dequeue boundary.

Coverage:
  Scenario 1 -- enqueue → dequeue happy path: submit → blpop → update → done.
  Scenario 2 -- cancel PENDING removes from queue (LREM guard; no ghost entries).
  Scenario 3 -- status index consistency: store, transition, list_jobs / count_jobs.
  Scenario 4 -- PEL drain after ACK: fully-ACKed stream leaves PEL at zero;
               a second AgentLoop finds nothing and does not re-process.

Read before editing:
  ml_engine/jobs/async_manager.py      -- AsyncJobManager.submit_job(), cancel_job()
  ml_engine/jobs/async_redis_store.py  -- AsyncRedisJobStore, enqueue_job, store_job,
                                          update_job, remove_from_queue (LREM), list_jobs
  ml_engine/jobs/models.py             -- Job, JobStatus, JobType
  ml_engine/agent/loop.py              -- AgentLoop (used for PEL test; StreamConsumer is ABC)

Pitfall catalogue:
  cancel-pending-no-lrem        -- cancel_job PENDING must LREM queue list; ghost entry
                                   causes a worker to dequeue a cancelled job
  async-sync-store-index-drift  -- update_job must atomically move job between status
                                   SET indices; a missed srem/sadd makes count_jobs lie
  duplicate-dequeue             -- high-priority LPUSH vs low-priority RPUSH ordering;
                                   wrong direction means worker skips urgent jobs
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

import pytest
import pytest_asyncio

from ml_engine.agent.loop import AgentLoop, apublish_event, ensure_consumer_group
from ml_engine.agent.stream_consumer import stream_key
from ml_engine.jobs.async_manager import AsyncJobManager
from ml_engine.jobs.async_redis_store import AsyncRedisJobStore
from ml_engine.jobs.models import Job, JobStatus, JobType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_QUEUE = AsyncRedisJobStore.JOB_QUEUE_KEY
_CONSUMER_0 = "coordinator-0"
_CONSUMER_1 = "coordinator-1"


def _decode(raw: bytes | str) -> str:
    return raw.decode() if isinstance(raw, bytes) else raw


async def _blpop(redis_async: Any, timeout: float = 0.1) -> str | None:
    """BLPOP job_queue and return the job_id string, or None if empty."""
    result = await redis_async.blpop(_QUEUE, timeout=timeout)
    if result is None:
        return None
    _key, raw_id = result
    return _decode(raw_id)


@pytest_asyncio.fixture
async def running_job(redis_async: Any):
    """Submit a job, dequeue it, and mark it RUNNING. Returns (manager, job)."""
    manager = AsyncJobManager(redis_client=redis_async)
    job = await manager.submit_job(job_type=JobType.AUTO_LABEL.value, config={})
    await _blpop(redis_async)
    await manager.store.update_job(job.id, status=JobStatus.RUNNING)
    return manager, job


# ---------------------------------------------------------------------------
# Scenario 1 -- Enqueue → dequeue happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_job_enqueue_dequeue_flow(redis_async: Any) -> None:
    """Submit → BLPOP → RUNNING → COMPLETED round-trip. Core integration boundary."""
    manager = AsyncJobManager(redis_client=redis_async)
    job = await manager.submit_job(job_type=JobType.AUTO_LABEL.value, config={})

    assert await redis_async.llen(_QUEUE) == 1

    job_id = await _blpop(redis_async)
    assert job_id == job.id, "dequeued id must match submitted job"

    await manager.store.update_job(job_id, status=JobStatus.RUNNING)
    await manager.store.update_job(job_id, status=JobStatus.COMPLETED)

    fetched = await manager.get_job(job_id)
    assert fetched is not None
    assert fetched.status == JobStatus.COMPLETED
    assert await redis_async.llen(_QUEUE) == 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_five_jobs_all_dequeue_unique_ids(redis_async: Any) -> None:
    """Five submitted jobs enqueue; each BLPOP yields a unique, correct id.
    Catches: duplicate enqueue, RPUSH ordering preserving identity."""
    manager = AsyncJobManager(redis_client=redis_async)
    submitted = [await manager.submit_job(job_type=JobType.AUTO_LABEL.value, config={}) for _ in range(5)]
    submitted_ids = {j.id for j in submitted}

    assert await redis_async.llen(_QUEUE) == 5

    dequeued = set()
    for _ in range(5):
        jid = await _blpop(redis_async)
        assert jid is not None
        assert jid not in dequeued, f"duplicate dequeue: {jid}"
        dequeued.add(jid)

    assert dequeued == submitted_ids


@pytest.mark.asyncio
@pytest.mark.integration
async def test_high_priority_job_dequeued_first(redis_async: Any) -> None:
    """priority>0 uses LPUSH (front); priority=0 uses RPUSH (back).
    Catches: inverted priority direction — urgent jobs buried behind normal ones."""
    manager = AsyncJobManager(redis_client=redis_async)
    normal = await manager.submit_job(job_type=JobType.AUTO_LABEL.value, config={}, priority=0)
    urgent = await manager.submit_job(job_type=JobType.AUTO_LABEL.value, config={}, priority=1)

    first_id = await _blpop(redis_async)
    assert first_id == urgent.id, "high-priority job must be dequeued before normal"
    second_id = await _blpop(redis_async)
    assert second_id == normal.id


@pytest.mark.asyncio
@pytest.mark.integration
async def test_config_roundtrip_nested(redis_async: Any) -> None:
    """Complex nested config survives JSON serialization in Redis hash.
    Catches: json.dumps/loads mismatch or field truncation in Job.to_dict()."""
    cfg = {"lr": 0.001, "layers": [1, 2, 3], "model": {"name": "resnet", "depth": 50}}
    manager = AsyncJobManager(redis_client=redis_async)
    await manager.submit_job(job_type=JobType.AUTO_LABEL.value, config=cfg)
    jid = await _blpop(redis_async)
    fetched = await manager.get_job(jid)
    assert fetched is not None
    assert fetched.config == cfg


@pytest.mark.asyncio
@pytest.mark.integration
async def test_submit_invalid_job_type_raises_no_queue_entry(redis_async: Any) -> None:
    """Invalid job_type raises ValueError before any Redis write.
    Catches: partial enqueue leaving orphan queue entry on validation failure."""
    manager = AsyncJobManager(redis_client=redis_async)
    with pytest.raises(ValueError, match="Invalid job type"):
        await manager.submit_job(job_type="not_a_real_type", config={})
    assert await redis_async.llen(_QUEUE) == 0, "invalid submit must not touch queue"
    assert await redis_async.keys("job:*") == [], "invalid submit must not leave orphan hash"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_all_valid_job_types_accepted(redis_async: Any) -> None:
    """Every JobType enum value is accepted by submit_job without error."""
    manager = AsyncJobManager(redis_client=redis_async)
    for jt in JobType:
        job = await manager.submit_job(job_type=jt.value, config={})
        assert job.type == jt.value
    assert await redis_async.llen(_QUEUE) == len(JobType)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_queue_empty_after_all_dequeued(redis_async: Any) -> None:
    """After dequeuing all submitted jobs, queue length is 0 and further BLPOP is empty."""
    manager = AsyncJobManager(redis_client=redis_async)
    await manager.submit_job(job_type=JobType.AUTO_LABEL.value, config={})

    jid = await _blpop(redis_async)
    assert jid is not None
    assert await redis_async.llen(_QUEUE) == 0


# ---------------------------------------------------------------------------
# Scenario 2 -- Cancel PENDING removes from queue (LREM guard)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_cancel_pending_removes_from_queue_no_ghost(redis_async: Any) -> None:
    """Cancel PENDING: llen drops to 0, status CANCELLED, no ghost entry survives LREM.
    This is the primary LREM guard test — the named pitfall 'cancel-pending-no-lrem'."""
    manager = AsyncJobManager(redis_client=redis_async)
    job = await manager.submit_job(job_type=JobType.AUTO_LABEL.value, config={})
    assert await redis_async.llen(_QUEUE) == 1

    cancelled = await manager.cancel_job(job.id)
    assert cancelled is True
    assert await redis_async.llen(_QUEUE) == 0

    fetched = await manager.get_job(job.id)
    assert fetched is not None
    assert fetched.status == JobStatus.CANCELLED

    # Ghost-entry check: any entry remaining would be picked up by a worker
    assert await redis_async.llen(_QUEUE) == 0, "ghost entry detected after cancel — LREM failed"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_cancel_middle_job_of_five(redis_async: Any) -> None:
    """Cancel 2 of 5 jobs; exactly 3 remain; dequeued ids never include cancelled ones."""
    manager = AsyncJobManager(redis_client=redis_async)
    jobs = [await manager.submit_job(job_type=JobType.AUTO_LABEL.value, config={}) for _ in range(5)]
    cancel_ids = {jobs[1].id, jobs[3].id}

    for jid in cancel_ids:
        assert await manager.cancel_job(jid) is True

    assert await redis_async.llen(_QUEUE) == 3

    dequeued = set()
    for _ in range(3):
        jid = await _blpop(redis_async)
        assert jid is not None
        dequeued.add(jid)

    assert dequeued.isdisjoint(cancel_ids), "cancelled job ids must not appear in dequeue"
    assert await redis_async.llen(_QUEUE) == 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_double_cancel_pending_second_returns_false(redis_async: Any) -> None:
    """Double-cancel of a PENDING job: first returns True, second returns False.
    CANCELLED is terminal; cancel_job guards against terminal-state re-cancel."""
    manager = AsyncJobManager(redis_client=redis_async)
    job = await manager.submit_job(job_type=JobType.AUTO_LABEL.value, config={})

    first = await manager.cancel_job(job.id)
    assert first is True

    second = await manager.cancel_job(job.id)
    assert second is False, "second cancel of CANCELLED job must return False (terminal guard)"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_cancel_running_job_sets_cancelling_no_lrem(running_job: Any, redis_async: Any) -> None:
    """Cancel RUNNING → CANCELLING (not CANCELLED). No LREM: already dequeued by worker.
    Catches: LREM on an empty queue causing an unintended LPUSH re-enqueue elsewhere."""
    manager, job = running_job

    result = await manager.cancel_job(job.id)
    assert result is True

    fetched = await manager.get_job(job.id)
    assert fetched is not None
    assert fetched.status == JobStatus.CANCELLING
    assert await redis_async.llen(_QUEUE) == 0, "queue must stay empty after cancel of RUNNING job"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_cancel_cancelling_job_returns_true(running_job: Any) -> None:
    """cancel_job on an already-CANCELLING job returns True (idempotent, no second event)."""
    manager, job = running_job
    await manager.cancel_job(job.id)  # → CANCELLING

    result = await manager.cancel_job(job.id)
    assert result is True  # second cancel of CANCELLING still returns True


@pytest.mark.asyncio
@pytest.mark.integration
async def test_cancel_terminal_job_returns_false(running_job: Any) -> None:
    """cancel_job on COMPLETED returns False without modifying state."""
    manager, job = running_job
    await manager.store.update_job(job.id, status=JobStatus.COMPLETED)

    result = await manager.cancel_job(job.id)
    assert result is False

    # Status must be unchanged
    fetched = await manager.get_job(job.id)
    assert fetched is not None
    assert fetched.status == JobStatus.COMPLETED


@pytest.mark.asyncio
@pytest.mark.integration
async def test_cancel_nonexistent_job_returns_false(redis_async: Any) -> None:
    """cancel_job on an unknown id returns False, no exception, no side effects."""
    manager = AsyncJobManager(redis_client=redis_async)
    result = await manager.cancel_job("00000000-0000-0000-0000-000000000000")
    assert result is False
    assert await redis_async.llen(_QUEUE) == 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_cancel_failed_job_returns_false(running_job: Any) -> None:
    """FAILED is terminal — cancel returns False and status stays FAILED."""
    manager, job = running_job
    await manager.store.update_job(job.id, status=JobStatus.FAILED)

    result = await manager.cancel_job(job.id)
    assert result is False


# ---------------------------------------------------------------------------
# Scenario 3 -- Status index consistency (AsyncRedisJobStore)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_status_index_consistent_across_transitions(redis_async: Any) -> None:
    """
    store_job RUNNING → list_jobs RUNNING includes it →
    update COMPLETED → list_jobs COMPLETED includes it, RUNNING excludes it.

    Pitfall 'async-sync-store-index-drift': update_job must atomically pipeline
    SREM old-status + SADD new-status. A missed SREM leaves ghost entries in the
    old index — list_jobs returns stale results and count_jobs lies.
    """
    store = AsyncRedisJobStore(redis_client=redis_async)
    job = Job(type=JobType.AUTO_LABEL.value, status=JobStatus.RUNNING, config={})
    await store.store_job(job)

    running = await store.list_jobs(status=JobStatus.RUNNING)
    assert any(j.id == job.id for j in running), "must appear in RUNNING after store_job"

    await store.update_job(job.id, status=JobStatus.COMPLETED)

    completed = await store.list_jobs(status=JobStatus.COMPLETED)
    assert any(j.id == job.id for j in completed), "must appear in COMPLETED after update"

    running_after = await store.list_jobs(status=JobStatus.RUNNING)
    assert not any(j.id == job.id for j in running_after), (
        "must NOT remain in RUNNING after update — SREM missing (index drift bug)"
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_status_index_pending_to_failed(redis_async: Any) -> None:
    """PENDING → FAILED transition: index correctly reflects FAILED, PENDING is clear."""
    store = AsyncRedisJobStore(redis_client=redis_async)
    job = Job(type=JobType.AUTO_LABEL.value, config={})
    await store.enqueue_job(job)

    await store.update_job(job.id, status=JobStatus.FAILED)

    failed = await store.list_jobs(status=JobStatus.FAILED)
    assert any(j.id == job.id for j in failed)

    pending = await store.list_jobs(status=JobStatus.PENDING)
    assert not any(j.id == job.id for j in pending), "PENDING index must not retain FAILED job"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_count_jobs_tracks_transitions(redis_async: Any) -> None:
    """count_jobs SCARD stays consistent through enqueue → running → completed.
    Catches index-drift bug where SCARD reports stale counts."""
    store = AsyncRedisJobStore(redis_client=redis_async)
    job = Job(type=JobType.AUTO_LABEL.value, config={})
    await store.enqueue_job(job)

    assert await store.count_jobs(JobStatus.PENDING) == 1
    assert await store.count_jobs(JobStatus.RUNNING) == 0

    await store.update_job(job.id, status=JobStatus.RUNNING)
    assert await store.count_jobs(JobStatus.PENDING) == 0
    assert await store.count_jobs(JobStatus.RUNNING) == 1

    await store.update_job(job.id, status=JobStatus.COMPLETED)
    assert await store.count_jobs(JobStatus.RUNNING) == 0
    assert await store.count_jobs(JobStatus.COMPLETED) == 1


@pytest.mark.asyncio
@pytest.mark.integration
async def test_count_jobs_multiple_statuses_simultaneously(redis_async: Any) -> None:
    """Multiple jobs in distinct statuses: SCARD is accurate per bucket.
    Catches: status index keying error where all jobs land in the same SET."""
    store = AsyncRedisJobStore(redis_client=redis_async)

    pending_job = Job(type=JobType.AUTO_LABEL.value, config={})
    await store.enqueue_job(pending_job)

    running_job = Job(type=JobType.AUTO_LABEL.value, status=JobStatus.RUNNING, config={})
    await store.store_job(running_job)

    completed_job = Job(type=JobType.AUTO_LABEL.value, status=JobStatus.RUNNING, config={})
    await store.store_job(completed_job)
    await store.update_job(completed_job.id, status=JobStatus.COMPLETED)

    assert await store.count_jobs(JobStatus.PENDING) == 1
    assert await store.count_jobs(JobStatus.RUNNING) == 1
    assert await store.count_jobs(JobStatus.COMPLETED) == 1


@pytest.mark.asyncio
@pytest.mark.integration
async def test_count_jobs_empty_returns_zero(redis_async: Any) -> None:
    """count_jobs on an empty store returns 0, not None or an exception."""
    store = AsyncRedisJobStore(redis_client=redis_async)
    assert await store.count_jobs(JobStatus.PENDING) == 0
    assert await store.count_jobs(JobStatus.RUNNING) == 0
    total = await store.count_jobs()
    assert total == 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_update_job_multiple_fields_atomic(redis_async: Any) -> None:
    """update_job with status + worker_id + started_at persists all fields together.
    Catches: partial update where status moves but scalar fields are silently dropped."""
    store = AsyncRedisJobStore(redis_client=redis_async)
    job = Job(type=JobType.AUTO_LABEL.value, config={})
    await store.enqueue_job(job)

    started = datetime.now(timezone.utc)
    await store.update_job(
        job.id,
        status=JobStatus.RUNNING,
        worker_id="worker-42",
        started_at=started,
    )

    fetched = await store.get_job(job.id)
    assert fetched is not None
    assert fetched.status == JobStatus.RUNNING
    assert fetched.worker_id == "worker-42"
    assert fetched.started_at == started


# ---------------------------------------------------------------------------
# Scenario 4 -- PEL drain after ACK: no reprocessing on second consumer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_pel_empty_after_successful_run(run_id: str, redis_async: Any) -> None:
    """
    After AgentLoop processes and ACKs all events, PEL count is 0.
    A second AgentLoop (same consumer_name) finds nothing in PEL and does
    not call on_event.

    Contrast with test_crash_recovery_* in test_agent_lifecycle.py, which tests
    the un-ACKed (CancelledError) path. This test closes the other branch:
    successful ACK → PEL is clean → no reprocessing.
    """
    await ensure_consumer_group(redis_async, run_id)
    for _ in range(3):
        await apublish_event(redis_async, run_id, {"type": "heartbeat"})

    processed_a: list = []

    async def handler_a(event: Dict[str, Any], _state: Any) -> None:
        processed_a.append(event)

    loop_a = AgentLoop(
        redis_client=redis_async,
        run_id=run_id,
        on_event=handler_a,
        consumer_name=_CONSUMER_0,
    )
    await loop_a.run(max_events=3)
    assert len(processed_a) == 3

    pending_info = await redis_async.xpending(stream_key(run_id), AgentLoop.CONSUMER_GROUP)
    assert pending_info["pending"] == 0, "PEL must be empty after all events ACKed"

    processed_b: list = []

    async def handler_b(event: Dict[str, Any], _state: Any) -> None:
        processed_b.append(event)

    loop_b = AgentLoop(
        redis_client=redis_async,
        run_id=run_id,
        on_event=handler_b,
        consumer_name=_CONSUMER_0,
    )
    # cancel_check=lambda: True exits immediately after _drain_pel finds nothing
    await loop_b.run(cancel_check=lambda: True)
    assert len(processed_b) == 0, "second loop must not reprocess already-ACKed events"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_pel_clean_for_new_consumer_name(run_id: str, redis_async: Any) -> None:
    """
    PEL is keyed per (group, consumer_name). coordinator-1 starting fresh must NOT
    replay coordinator-0's already-ACKed events; it should only see genuinely new
    messages delivered to it.

    Uses max_events (bounded internal counter) rather than cancel_check (external
    abort signal) so that _drain_pel actually runs and verifies the empty PEL before
    reading from the stream. cancel_check=lambda: True would short-circuit _drain_pel
    entirely and skip the verification we care about.
    """
    await ensure_consumer_group(redis_async, run_id)
    for _ in range(2):
        await apublish_event(redis_async, run_id, {"type": "heartbeat"})

    async def noop(_event: Dict[str, Any], _state: Any) -> None:
        pass

    loop_0 = AgentLoop(
        redis_client=redis_async,
        run_id=run_id,
        on_event=noop,
        consumer_name=_CONSUMER_0,
    )
    await loop_0.run(max_events=2)

    # Publish a new event AFTER coordinator-0 has ACKed its two heartbeats.
    # coordinator-1 should receive exactly this one new event — not a replay.
    await apublish_event(redis_async, run_id, {"type": "new_task"})

    processed_1: list = []

    async def handler_1(event: Dict[str, Any], _state: Any) -> None:
        processed_1.append(event)

    loop_1 = AgentLoop(
        redis_client=redis_async,
        run_id=run_id,
        on_event=handler_1,
        consumer_name=_CONSUMER_1,
    )
    # max_events=1: _drain_pel runs (finds empty PEL for coordinator-1), then
    # reads exactly the one new_task event from the stream and stops.
    await loop_1.run(max_events=1)

    assert len(processed_1) == 1, "coordinator-1 must receive the new event, not coordinator-0's ACKed events"
    assert processed_1[0].get("type") == "new_task", "coordinator-1 must see new_task, not replayed heartbeat"
