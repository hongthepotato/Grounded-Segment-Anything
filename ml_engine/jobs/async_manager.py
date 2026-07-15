"""
Async twin of :class:`JobManager`.

Same API as :class:`ml_engine.jobs.manager.JobManager`, but every method that
touches Redis is ``async`` and awaits :class:`AsyncRedisJobStore` calls.

Why a twin instead of dual-API on one class:
- The sync :class:`JobManager` is still used by ``ml_engine/jobs/worker.py``,
  which runs as its own sync process. There is no transitional benefit in
  bundling both APIs into one class; twins keep each call site explicit about
  which world it lives in.

Uses the same per-URL singleton pattern as :func:`get_job_manager` so the
connection pool is shared across FastAPI routes, Coordinator, and workers.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import redis.asyncio as _aredis

from ml_engine.jobs.async_redis_store import AsyncRedisJobStore
from ml_engine.jobs.models import Job, JobStatus, JobType, WorkerInfo

logger = logging.getLogger(__name__)


class AsyncJobManager:
    """High-level async job management API. See :class:`JobManager` for docs."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        redis_client: Optional[_aredis.Redis] = None,
    ):
        self.redis_url = redis_url
        self.store = AsyncRedisJobStore(redis_url, redis_client=redis_client)
        logger.info("AsyncJobManager initialized with Redis at %s", redis_url)

    async def close(self) -> None:
        await self.store.close()

    # =========================================================================
    # Job Submission
    # =========================================================================

    async def submit_job(
        self,
        job_type: str,
        config: Dict[str, Any],
        priority: int = 0,
        output_dir: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Job:
        try:
            JobType(job_type)
        except ValueError as e:
            valid_types = [t.value for t in JobType]
            raise ValueError(f"Invalid job type: {job_type}. Must be one of: {valid_types}") from e

        job = Job(
            type=job_type,
            status=JobStatus.PENDING,
            config=config,
            priority=priority,
            output_dir=output_dir,
            tags=tags or [],
        )
        logger.info(
            "Submitting job %s (type=%s, priority=%d)",
            job.id[:8],
            job_type,
            priority,
        )
        await self.store.enqueue_job(job)
        return job

    # =========================================================================
    # Cancellation
    # =========================================================================

    async def cancel_job(self, job_id: str) -> bool:
        job = await self.store.get_job(job_id)
        if job is None:
            logger.warning("Cannot cancel job %s: not found", job_id[:8])
            return False
        if job.is_terminal:
            logger.info(
                "Cannot cancel job %s: already in terminal state %s",
                job_id[:8],
                job.status.value,
            )
            return False

        if job.status == JobStatus.PENDING:
            await self.store.update_job(
                job_id,
                status=JobStatus.CANCELLED,
                finished_at=datetime.now(timezone.utc),
            )
            await self.store.remove_from_queue(job_id)
            logger.info("Cancelled pending job %s", job_id[:8])
        elif job.status == JobStatus.RUNNING:
            await self.store.update_job(job_id, status=JobStatus.CANCELLING)
            logger.info("Requested cancellation for running job %s", job_id[:8])
        elif job.status == JobStatus.CANCELLING:
            logger.info("Job %s is already cancelling", job_id[:8])
            return True  # Already in progress -- no second cancel_requested event

        await self.store.publish_event(
            job_id,
            {
                "type": "cancel_requested",
                "job_id": job_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        return True

    # =========================================================================
    # Queries
    # =========================================================================

    async def get_job(self, job_id: str) -> Optional[Job]:
        return await self.store.get_job(job_id)

    async def list_jobs(
        self,
        status: Optional[str] = None,
        job_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Job]:
        status_enum: Optional[JobStatus] = None
        if status:
            try:
                status_enum = JobStatus(status)
            except ValueError:
                logger.warning("Invalid status filter: %s", status)
        return await self.store.list_jobs(
            status=status_enum,
            job_type=job_type,
            limit=limit,
            offset=offset,
        )

    async def get_job_count(self, status: Optional[str] = None) -> int:
        """O(1) via SCARD on the status index (TODO-6 fix bundled in Phase 8)."""
        return await self.store.count_jobs(status)

    async def delete_job(self, job_id: str) -> bool:
        job = await self.store.get_job(job_id)
        if job is None:
            return False
        if not job.is_terminal:
            logger.warning(
                "Cannot delete non-terminal job %s (status=%s)",
                job_id[:8],
                job.status.value,
            )
            return False
        return await self.store.delete_job(job_id)

    # =========================================================================
    # Queue info
    # =========================================================================

    async def get_queue_length(self) -> int:
        return await self.store.get_queue_length()

    async def get_queue_status(self) -> Dict[str, Any]:
        # Pulled list_workers() out of the gather so its concrete return
        # type (List[WorkerInfo]) survives — asyncio.gather over heterogeneous
        # awaitables collapses to list[Any] and the worker-iteration below
        # then fails type-checking. The 6 remaining calls all return int, so
        # gather's typed overloads handle them. Trade-off: list_workers runs
        # sequentially first instead of in parallel with the SCARDs (~1-5ms
        # extra wall time on a status endpoint, not in any hot path).
        workers = await self.store.list_workers()
        queue_length, pending, running, completed, failed, cancelled = await asyncio.gather(
            self.store.get_queue_length(),
            self.get_job_count("pending"),
            self.get_job_count("running"),
            self.get_job_count("completed"),
            self.get_job_count("failed"),
            self.get_job_count("cancelled"),
        )
        return {
            "queue_length": queue_length,
            "workers": [w.to_dict() for w in workers],
            "job_counts": {
                "pending": pending,
                "running": running,
                "completed": completed,
                "failed": failed,
                "cancelled": cancelled,
            },
        }

    # =========================================================================
    # Workers
    # =========================================================================

    async def list_workers(self, status: Optional[str] = None) -> List[WorkerInfo]:
        return await self.store.list_workers(status=status)

    async def cleanup_stale_workers(self, timeout_seconds: int = 60) -> int:
        return await self.store.cleanup_stale_workers(timeout_seconds)


# ---------------------------------------------------------------------------
# Singleton factory (one manager per redis_url, process-wide)
# ---------------------------------------------------------------------------

_default_managers: Dict[str, AsyncJobManager] = {}


def get_async_job_manager(
    redis_url: str = "redis://localhost:6379",
) -> AsyncJobManager:
    """Return the cached async job manager for ``redis_url``."""
    m = _default_managers.get(redis_url)
    if m is not None:
        return m
    m = AsyncJobManager(redis_url)
    _default_managers[redis_url] = m
    return m


async def close_async_job_managers() -> None:
    """Close every cached manager. Call at process shutdown."""
    while _default_managers:
        _url, m = _default_managers.popitem()
        try:
            await m.close()
        except Exception as e:
            logger.warning("Error closing AsyncJobManager for %s: %s", _url, e)
