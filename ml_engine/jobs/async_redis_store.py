"""
Async twin of :class:`RedisJobStore`.

Same Redis layout, same keys, same semantics -- just using
``redis.asyncio.Redis`` so callers in async contexts (FastAPI routes,
Coordinator, agent workers) stop blocking the event loop.

The sync :class:`RedisJobStore` is kept for ``ml_engine/jobs/worker.py``, which
runs as a separate sync process; there is no async benefit there.

Bundled in Phase 8: a status index SET (``jobs:by_status:{status}``) maintained
by :meth:`enqueue_job`, :meth:`update_job`, and :meth:`delete_job`. This lets
:meth:`count_jobs` answer via ``SCARD`` in O(1) instead of the old
``list_jobs(limit=10000)`` scan (TODO-6).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import redis.asyncio as _aredis
from redis.exceptions import RedisError

from ml_engine.jobs.models import Job, JobProgress, JobStatus, WorkerInfo

logger = logging.getLogger(__name__)


class AsyncRedisJobStore:
    """Async counterpart of :class:`RedisJobStore`. See that class for layout."""

    JOB_QUEUE_KEY = "job_queue"
    JOB_PREFIX = "job:"
    WORKERS_KEY = "workers"
    WORKER_PREFIX = "worker:"
    STATUS_INDEX_PREFIX = "jobs:by_status:"

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        db: int = 0,
        redis_client: Optional[_aredis.Redis] = None,
    ):
        self.redis_url = redis_url
        self.db = db
        # `self.redis: Any` instead of _aredis.Redis: redis-py's stubs
        # declare async client methods as `Awaitable[X] | X` (an artifact
        # of the same class inheriting both sync and async command mixins
        # — see redis.asyncio.Redis.__mro__). That makes every
        # `await self.redis.method()` below fail mypy with
        # "Awaitable[X] | X is not Awaitable[Any]". Trade-off: lose
        # method-name typo detection on the redis client; gain a clean
        # baseline. Same workaround in the sync sibling RedisJobStore.
        self.redis: Any
        if redis_client is not None:
            # Test path: caller passes a pre-built async client (e.g. fakeredis).
            self.redis = redis_client
            self._owns_client = False
        else:
            self.redis = _aredis.Redis.from_url(
                redis_url,
                db=db,
                decode_responses=False,
                max_connections=20,
            )
            self._owns_client = True

    async def ping(self) -> None:
        """Verify connectivity. Raises RedisError on failure."""
        await self.redis.ping()

    async def close(self) -> None:
        if self._owns_client:
            await self.redis.aclose()

    # =========================================================================
    # Queue Operations
    # =========================================================================

    def _status_key(self, status: JobStatus | str) -> str:
        value = status.value if isinstance(status, JobStatus) else status
        return f"{self.STATUS_INDEX_PREFIX}{value}"

    async def enqueue_job(self, job: Job) -> None:
        job_key = f"{self.JOB_PREFIX}{job.id}"
        pipe = self.redis.pipeline()
        try:
            pipe.hset(job_key, mapping=job.to_dict())
            pipe.sadd(self._status_key(job.status), job.id)
            if job.priority > 0:
                pipe.lpush(self.JOB_QUEUE_KEY, job.id)
            else:
                pipe.rpush(self.JOB_QUEUE_KEY, job.id)
            await pipe.execute()
            logger.info("Enqueued job %s (priority=%d)", job.id[:8], job.priority)
            await self.publish_event(
                job.id,
                {
                    "type": "job_enqueued",
                    "job_id": job.id,
                    "status": job.status.value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        except RedisError as e:
            logger.error("Failed to enqueue job %s: %s", job.id[:8], e)
            raise

    async def store_job(self, job: Job) -> None:
        """Persist a job WITHOUT adding it to the work queue (Coordinator path)."""
        job_key = f"{self.JOB_PREFIX}{job.id}"
        try:
            pipe = self.redis.pipeline()
            pipe.hset(job_key, mapping=job.to_dict())
            pipe.sadd(self._status_key(job.status), job.id)
            await pipe.execute()
            logger.debug("Stored job %s (not yet queued)", job.id[:8])
        except RedisError as e:
            logger.error("Failed to store job %s: %s", job.id[:8], e)
            raise

    async def enqueue_by_id(self, job_id: str) -> bool:
        job = await self.get_job(job_id)
        if job is None:
            logger.warning("enqueue_by_id: job %s not found", job_id[:8])
            return False
        try:
            if job.priority > 0:
                await self.redis.lpush(self.JOB_QUEUE_KEY, job_id)
            else:
                await self.redis.rpush(self.JOB_QUEUE_KEY, job_id)
            logger.info("Queued job %s (priority=%d)", job_id[:8], job.priority)
            await self.publish_event(
                job_id,
                {
                    "type": "job_enqueued",
                    "job_id": job_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            return True
        except RedisError as e:
            logger.error("Failed to queue job %s: %s", job_id[:8], e)
            return False

    async def requeue_job(self, job_id: str, to_front: bool = True) -> bool:
        try:
            if to_front:
                await self.redis.lpush(self.JOB_QUEUE_KEY, job_id)
            else:
                await self.redis.rpush(self.JOB_QUEUE_KEY, job_id)
            await self.update_job(job_id, status=JobStatus.PENDING, worker_id=None)
            logger.info("Requeued job %s (front=%s)", job_id[:8], to_front)
            return True
        except RedisError as e:
            logger.error("Failed to requeue job %s: %s", job_id[:8], e)
            return False

    async def get_queue_length(self) -> int:
        try:
            return await self.redis.llen(self.JOB_QUEUE_KEY)
        except RedisError as e:
            logger.warning("Failed to get queue length: %s", e)
            return 0

    async def remove_from_queue(self, job_id: str) -> bool:
        """Remove a job ID from the pending queue LIST (LREM). Used when cancelling a pending job."""
        try:
            removed = await self.redis.lrem(self.JOB_QUEUE_KEY, 0, job_id)
            if removed:
                logger.info("Removed job %s from queue list", job_id[:8])
            return bool(removed)
        except RedisError as e:
            logger.warning("Failed to remove job %s from queue: %s", job_id[:8], e)
            return False

    # =========================================================================
    # Job State Operations
    # =========================================================================

    async def get_job(self, job_id: str) -> Optional[Job]:
        job_key = f"{self.JOB_PREFIX}{job_id}"
        try:
            data = await self.redis.hgetall(job_key)
            if not data:
                return None
            return Job.from_dict(data)
        except RedisError as e:
            logger.error("Failed to get job %s: %s", job_id[:8], e)
            return None

    async def update_job(self, job_id: str, **updates) -> bool:
        job_key = f"{self.JOB_PREFIX}{job_id}"

        # Status change needs to move job between status-index sets.
        old_status: Optional[str] = None
        new_status: Optional[str] = None
        if "status" in updates:
            raw = await self.redis.hget(job_key, "status")
            if raw is not None:
                old_status = raw.decode() if isinstance(raw, bytes) else raw
            v = updates["status"]
            new_status = v.value if isinstance(v, JobStatus) else str(v)

        try:
            redis_updates: Dict[str, Any] = {}
            for key, value in updates.items():
                if key == "status" and isinstance(value, JobStatus):
                    redis_updates[key] = value.value
                elif key == "progress" and isinstance(value, JobProgress):
                    redis_updates[key] = json.dumps(value.to_dict())
                elif isinstance(value, datetime):
                    redis_updates[key] = value.isoformat()
                elif isinstance(value, (dict, list)):
                    redis_updates[key] = json.dumps(value)
                elif value is None:
                    redis_updates[key] = ""
                else:
                    redis_updates[key] = str(value)

            if redis_updates:
                pipe = self.redis.pipeline()
                pipe.hset(job_key, mapping=redis_updates)
                if new_status is not None and new_status != old_status:
                    if old_status:
                        pipe.srem(self._status_key(old_status), job_id)
                    pipe.sadd(self._status_key(new_status), job_id)
                await pipe.execute()

                await self.publish_event(
                    job_id,
                    {
                        "type": "job_updated",
                        "job_id": job_id,
                        "updates": {k: str(v) for k, v in updates.items()},
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )

                logger.debug("Updated job %s: %s", job_id[:8], list(updates.keys()))
            return True
        except RedisError as e:
            logger.error("Failed to update job %s: %s", job_id[:8], e)
            return False

    async def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        job_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Job]:
        try:
            if status is not None:
                # Fast path: use status index (SMEMBERS) instead of SCAN.
                # O(matching) instead of O(all jobs). The index is maintained
                # by enqueue_job, update_job, and delete_job.
                raw_ids = await self.redis.smembers(self._status_key(status))
                job_ids = [jid.decode() if isinstance(jid, bytes) else jid for jid in raw_ids]
                jobs_or_none = await asyncio.gather(*[self.get_job(jid) for jid in job_ids])
                jobs: list[Job] = [j for j in jobs_or_none if j is not None]
                if job_type:
                    jobs = [j for j in jobs if j.type == job_type]
                jobs.sort(key=lambda j: j.created_at or datetime.min, reverse=True)
                return jobs[offset : offset + limit]

            # Full SCAN path: no status filter (list all jobs).
            job_keys: list = []
            cursor = 0
            while True:
                cursor, keys = await self.redis.scan(
                    cursor=cursor,
                    match=f"{self.JOB_PREFIX}*",
                    count=100,
                )
                job_keys.extend(keys)
                if cursor == 0:
                    break

            all_jobs: list[Job] = []
            for key in job_keys:
                data = await self.redis.hgetall(key)
                if not data:
                    continue
                job = Job.from_dict(data)
                if job_type and job.type != job_type:
                    continue
                all_jobs.append(job)

            all_jobs.sort(key=lambda j: j.created_at or datetime.min, reverse=True)
            return all_jobs[offset : offset + limit]
        except RedisError as e:
            logger.error("Failed to list jobs: %s", e)
            return []

    async def count_jobs(self, status: Optional[JobStatus | str] = None) -> int:
        """
        Count jobs by status in O(1) via a status-index SET (SCARD).

        When no status is given, pipelines all SCARDs in one round-trip
        (total count still requires reading every status set, but that's one
        network round-trip instead of len(JobStatus)).
        """
        try:
            if status is not None:
                return int(await self.redis.scard(self._status_key(status)))
            # Total count: pipeline all SCARDs to avoid N serial round-trips.
            pipe = self.redis.pipeline()
            for s in JobStatus:
                pipe.scard(self._status_key(s))
            results = await pipe.execute()
            return sum(int(r) for r in results)
        except RedisError as e:
            logger.warning("Failed to count jobs: %s", e)
            return 0

    async def delete_job(self, job_id: str) -> bool:
        job_key = f"{self.JOB_PREFIX}{job_id}"
        try:
            raw = await self.redis.hget(job_key, "status")
            status_val = raw.decode() if isinstance(raw, bytes) else raw
            pipe = self.redis.pipeline()
            pipe.delete(job_key)
            if status_val:
                pipe.srem(self._status_key(status_val), job_id)
            results = await pipe.execute()
            deleted = bool(results and results[0])
            if deleted:
                logger.info("Deleted job %s", job_id[:8])
            return deleted
        except RedisError as e:
            logger.error("Failed to delete job %s: %s", job_id[:8], e)
            return False

    # =========================================================================
    # Pub/Sub
    # =========================================================================

    async def publish_event(self, job_id: str, event: Dict[str, Any]) -> int:
        channel = f"{self.JOB_PREFIX}{job_id}:events"
        try:
            message = json.dumps(event)
            return int(await self.redis.publish(channel, message))
        except RedisError as e:
            logger.error("Failed to publish event for job %s: %s", job_id[:8], e)
            return 0

    # =========================================================================
    # Worker Registry
    # =========================================================================

    async def register_worker(self, worker: WorkerInfo) -> bool:
        worker_key = f"{self.WORKER_PREFIX}{worker.id}"
        try:
            pipe = self.redis.pipeline()
            pipe.hset(worker_key, mapping=worker.to_dict())
            pipe.hset(self.WORKERS_KEY, worker.id, worker_key)
            await pipe.execute()
            logger.info("Registered worker %s (GPU %d)", worker.id, worker.gpu_id)
            return True
        except RedisError as e:
            logger.error("Failed to register worker %s: %s", worker.id, e)
            return False

    async def unregister_worker(self, worker_id: str) -> bool:
        worker_key = f"{self.WORKER_PREFIX}{worker_id}"
        try:
            pipe = self.redis.pipeline()
            pipe.delete(worker_key)
            pipe.hdel(self.WORKERS_KEY, worker_id)
            await pipe.execute()
            logger.info("Unregistered worker %s", worker_id)
            return True
        except RedisError as e:
            logger.error("Failed to unregister worker %s: %s", worker_id, e)
            return False

    async def update_worker_heartbeat(self, worker_id: str) -> bool:
        worker_key = f"{self.WORKER_PREFIX}{worker_id}"
        try:
            await self.redis.hset(worker_key, "last_heartbeat", datetime.now(timezone.utc).isoformat())
            return True
        except RedisError:
            return False

    async def update_worker_status(
        self,
        worker_id: str,
        status: str,
        current_job_id: Optional[str] = None,
    ) -> bool:
        worker_key = f"{self.WORKER_PREFIX}{worker_id}"
        try:
            updates = {
                "status": status,
                "current_job_id": current_job_id or "",
                "last_heartbeat": datetime.now(timezone.utc).isoformat(),
            }
            await self.redis.hset(worker_key, mapping=updates)
            return True
        except RedisError as e:
            logger.error("Failed to update worker %s status: %s", worker_id, e)
            return False

    async def get_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        worker_key = f"{self.WORKER_PREFIX}{worker_id}"
        try:
            data = await self.redis.hgetall(worker_key)
            if not data:
                return None
            return WorkerInfo.from_dict(data)
        except RedisError:
            return None

    async def list_workers(self, status: Optional[str] = None) -> List[WorkerInfo]:
        try:
            raw_ids = await self.redis.hkeys(self.WORKERS_KEY)
            worker_ids = [wid.decode() if isinstance(wid, bytes) else wid for wid in raw_ids]
            workers_or_none = await asyncio.gather(*[self.get_worker(wid) for wid in worker_ids])
            return [w for w in workers_or_none if w is not None and (status is None or w.status == status)]
        except RedisError as e:
            logger.error("Failed to list workers: %s", e)
            return []

    async def cleanup_stale_workers(self, timeout_seconds: int = 60) -> int:
        workers = await self.list_workers()
        now = datetime.now(timezone.utc)
        removed = 0
        for worker in workers:
            if worker.last_heartbeat:
                age = (now - worker.last_heartbeat).total_seconds()
                if age > timeout_seconds:
                    if worker.current_job_id:
                        await self.requeue_job(worker.current_job_id, to_front=True)
                    await self.unregister_worker(worker.id)
                    removed += 1
                    logger.warning(
                        "Removed stale worker %s (last heartbeat: %ds ago)",
                        worker.id,
                        int(age),
                    )
        return removed
