"""
Redis-based job store for distributed training job management.

This module provides:
- Job queue operations (enqueue, dequeue)
- Job state persistence (get, update, list)
- Pub/sub for real-time updates
- Worker registry for tracking active workers

Redis Data Structures:
- job_queue: LIST - pending job IDs (FIFO)
- job:{id}: HASH - job state
- workers: HASH - active workers
- job:{id}:events: CHANNEL - pub/sub for job updates
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import redis
from redis.exceptions import RedisError

from ml_engine.jobs.models import Job, JobProgress, JobStatus, WorkerInfo

logger = logging.getLogger(__name__)


class RedisJobStore:
    """
    Redis-based storage for training jobs.

    Thread-safe operations for:
    - Job queue management (FIFO queue via Redis LIST)
    - Job state persistence
    - Real-time event pub/sub
    - Worker registration

    Example:
        >>> store = RedisJobStore("redis://localhost:6379")
        >>> job = Job(type="teacher_training", config={...})
        >>> store.enqueue_job(job)
        >>>
        >>> # Worker picks up job
        >>> job_id = store.dequeue_job(timeout=5)
        >>> job = store.get_job(job_id)
        >>> store.update_job(job_id, status=JobStatus.RUNNING)
    """

    # Redis key prefixes
    JOB_QUEUE_KEY = "job_queue"
    JOB_PREFIX = "job:"
    WORKERS_KEY = "workers"
    WORKER_PREFIX = "worker:"
    STATUS_INDEX_PREFIX = "jobs:by_status:"

    def __init__(self, redis_url: str = "redis://localhost:6379", db: int = 0):
        """
        Initialize Redis connection.

        Args:
            redis_url: Redis connection URL (e.g., redis://localhost:6379)
            db: Redis database number
        """
        self.redis_url = redis_url
        self.db = db

        # Create connection pool for thread safety
        self.pool = redis.ConnectionPool.from_url(
            redis_url,
            db=db,
            decode_responses=False,  # We handle decoding ourselves
            max_connections=20,
        )
        # `Any` workaround: redis-py stubs declare client methods as
        # `Awaitable[X] | X` (sync/async overload artifact) which trips
        # mypy across this file. See AsyncRedisJobStore.__init__ for the
        # full rationale. Same trade-off applies here.
        self.redis: Any = redis.Redis(connection_pool=self.pool)

        # Test connection
        try:
            self.redis.ping()
            logger.info("Connected to Redis at %s", redis_url)
        except RedisError as e:
            logger.error("Failed to connect to Redis: %s", e)
            raise

    def close(self):
        """Close Redis connections."""
        self.pool.disconnect()
        logger.info("Redis connections closed")

    def _status_key(self, status) -> str:
        value = status.value if isinstance(status, JobStatus) else status
        return f"{self.STATUS_INDEX_PREFIX}{value}"

    # =========================================================================
    # Queue Operations
    # =========================================================================

    def enqueue_job(self, job: Job) -> None:
        """
        Add job to queue and store job state.

        Uses Redis transaction (MULTI/EXEC) to ensure atomicity:
        1. Store job state in hash
        2. Add job ID to queue

        Args:
            job: Job to enqueue
        """
        job_key = f"{self.JOB_PREFIX}{job.id}"

        # Use pipeline for atomic operation
        pipe = self.redis.pipeline()
        try:
            # Store job state (to_dict() already returns Redis-compatible strings)
            # Build job_key -> job.to_dict() (str -> str) mapping, redis HASH
            pipe.hset(job_key, mapping=job.to_dict())
            pipe.sadd(self._status_key(job.status), job.id)

            if job.priority > 0:
                pipe.lpush(self.JOB_QUEUE_KEY, job.id)
            else:
                pipe.rpush(self.JOB_QUEUE_KEY, job.id)

            pipe.execute()
            logger.info("Enqueued job %s (priority=%d)", job.id[:8], job.priority)

            # Publish enqueue event
            self.publish_event(
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

    def dequeue_job(self, timeout: int = 1) -> Optional[str]:
        """
        Dequeue next job from queue (blocking).

        Uses BLPOP for blocking dequeue with timeout.

        Args:
            timeout: Seconds to wait for job (0 = block forever)

        Returns:
            Job ID or None if timeout
        """
        try:
            result = self.redis.blpop(self.JOB_QUEUE_KEY, timeout=timeout)
            if result:
                _, job_id = result
                job_id_str = job_id.decode() if isinstance(job_id, bytes) else job_id
                logger.debug("Dequeued job %s", job_id_str[:8])
                return job_id_str
            return None
        except RedisError as e:
            logger.error("Failed to dequeue job: %s", e)
            return None

    def store_job(self, job: Job) -> None:
        """
        Persist job state to Redis WITHOUT adding it to the work queue.

        Used by the Coordinator's DispatchStageTool so that the ExecutorWorker
        can validate contract constraints before the job enters the queue.

        Args:
            job: Job to persist
        """
        job_key = f"{self.JOB_PREFIX}{job.id}"
        try:
            pipe = self.redis.pipeline()
            pipe.hset(job_key, mapping=job.to_dict())
            pipe.sadd(self._status_key(job.status), job.id)
            pipe.execute()
            logger.debug("Stored job %s (not yet queued)", job.id[:8])
        except RedisError as e:
            logger.error("Failed to store job %s: %s", job.id[:8], e)
            raise

    def enqueue_by_id(self, job_id: str) -> bool:
        """
        Move an already-stored job into the work queue.

        Used by ExecutorWorker after contract validation passes.

        Args:
            job_id: ID of a job previously saved via store_job()

        Returns:
            True if queued, False if job_id not found
        """
        job = self.get_job(job_id)
        if job is None:
            logger.warning("enqueue_by_id: job %s not found", job_id[:8])
            return False
        try:
            if job.priority > 0:
                self.redis.lpush(self.JOB_QUEUE_KEY, job_id)
            else:
                self.redis.rpush(self.JOB_QUEUE_KEY, job_id)
            logger.info("Queued job %s (priority=%d)", job_id[:8], job.priority)
            self.publish_event(
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

    def requeue_job(self, job_id: str, to_front: bool = True) -> bool:
        """
        Put job back in queue (e.g., after worker failure).

        Args:
            job_id: Job ID to requeue
            to_front: If True, add to front of queue (high priority)

        Returns:
            True if successful
        """
        try:
            if to_front:
                self.redis.lpush(self.JOB_QUEUE_KEY, job_id)
            else:
                self.redis.rpush(self.JOB_QUEUE_KEY, job_id)
            # Update job status back to pending
            self.update_job(job_id, status=JobStatus.PENDING, worker_id=None)
            logger.info("Requeued job %s (front=%s)", job_id[:8], to_front)
            return True
        except RedisError as e:
            logger.error("Failed to requeue job %s: %s", job_id[:8], e)
            return False

    def get_queue_length(self) -> int:
        """Get number of jobs in queue."""
        try:
            return self.redis.llen(self.JOB_QUEUE_KEY)
        except RedisError as e:
            logger.warning("Failed to get queue length: %s", e)
            return 0

    def remove_from_queue(self, job_id: str) -> bool:
        """Remove a job ID from the pending queue LIST (LREM)."""
        try:
            removed = self.redis.lrem(self.JOB_QUEUE_KEY, 0, job_id)
            if removed:
                logger.info("Removed job %s from queue list", job_id[:8])
            return bool(removed)
        except RedisError as e:
            logger.warning("Failed to remove job %s from queue: %s", job_id[:8], e)
            return False

    # =========================================================================
    # Job State Operations
    # =========================================================================

    def get_job(self, job_id: str) -> Optional[Job]:
        """
        Get job by ID.

        Args:
            job_id: Job ID

        Returns:
            Job object or None if not found
        """
        job_key = f"{self.JOB_PREFIX}{job_id}"
        try:
            data = self.redis.hgetall(job_key)
            if not data:
                return None
            return Job.from_dict(data)
        except RedisError as e:
            logger.error("Failed to get job %s: %s", job_id[:8], e)
            return None

    def update_job(self, job_id: str, **updates) -> bool:
        """
        Update job fields.

        Args:
            job_id: Job ID
            **updates: Fields to update (status, progress, error_message, etc.)

        Returns:
            True if successful
        """
        job_key = f"{self.JOB_PREFIX}{job_id}"

        old_status: Optional[str] = None
        new_status: Optional[str] = None
        if "status" in updates:
            raw = self.redis.hget(job_key, "status")
            if raw is not None:
                old_status = raw.decode() if isinstance(raw, bytes) else raw
            v = updates["status"]
            new_status = v.value if isinstance(v, JobStatus) else str(v)

        try:
            # Convert updates to Redis-compatible format
            redis_updates = {}
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
                pipe.execute()

                # Publish update event
                self.publish_event(
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

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        job_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Job]:
        """
        List jobs with optional filtering.

        Args:
            status: Filter by status
            job_type: Filter by job type
            limit: Maximum jobs to return
            offset: Pagination offset

        Returns:
            List of Job objects
        """
        try:
            # Get all job keys
            # Note: SCAN is more efficient for large datasets
            job_keys = []
            cursor = 0
            while True:
                cursor, keys = self.redis.scan(cursor=cursor, match=f"{self.JOB_PREFIX}*", count=100)
                job_keys.extend(keys)
                if cursor == 0:
                    break

            # Fetch and filter jobs
            jobs = []
            for key in job_keys:
                data = self.redis.hgetall(key)
                if not data:
                    continue

                job = Job.from_dict(data)

                # Apply filters
                if status and job.status != status:
                    continue
                if job_type and job.type != job_type:
                    continue

                jobs.append(job)

            # Sort by created_at (newest first)
            jobs.sort(key=lambda j: j.created_at or datetime.min, reverse=True)

            # Apply pagination
            return jobs[offset : offset + limit]

        except RedisError as e:
            logger.error("Failed to list jobs: %s", e)
            return []

    def count_jobs(self, status: Optional[JobStatus] = None) -> int:
        """
        Count jobs, optionally filtered by status.

        Signature deliberately mirrors AsyncRedisJobStore.count_jobs. An earlier
        draft also took a ``job_type`` filter, which no caller used and which the
        async store does not accept — that is the exact sync/async drift this
        module has already been bitten by twice, so the unused parameter is gone.
        If counting by type is ever needed, add it to BOTH stores.

        Only the ``status`` field is read per key, and those reads are PIPELINED
        per SCAN batch — roughly one round-trip per batch instead of one per job.

        Memory is O(number of job keys) because SCAN only guarantees
        at-least-once delivery (it may return the same key twice when the
        keyspace is rehashed mid-iteration), so keys must be de-duplicated to
        avoid over-counting. That is still far cheaper than list_jobs, which
        holds a fully constructed Job per key.

        Args:
            status: Optional status filter

        Returns:
            Number of matching jobs (0 on Redis error)
        """
        want_status = status.value if isinstance(status, JobStatus) else status

        try:
            count = 0
            seen: set = set()
            cursor = 0
            while True:
                cursor, keys = self.redis.scan(cursor=cursor, match=f"{self.JOB_PREFIX}*", count=100)
                # The key IS the identity (job:{id}), so de-duping needs no extra read.
                fresh = [k for k in keys if k not in seen]
                seen.update(fresh)
                if fresh:
                    # transaction=False: plain pipelining. These are independent
                    # reads with no atomicity requirement, so MULTI/EXEC would
                    # only add two commands per batch.
                    pipe = self.redis.pipeline(transaction=False)
                    for key in fresh:
                        pipe.hget(key, "status")
                    for raw_status in pipe.execute():
                        # None => the key was deleted between SCAN and read, or the
                        # hash has no status field (corrupt: every Job.to_dict writes
                        # one). Neither is a countable job.
                        if raw_status is None:
                            continue
                        if want_status is not None:
                            value = raw_status.decode() if isinstance(raw_status, bytes) else raw_status
                            if value != want_status:
                                continue
                        count += 1
                if cursor == 0:
                    break
            return count

        except RedisError as e:
            logger.error("Failed to count jobs: %s", e)
            return 0

    def delete_job(self, job_id: str) -> bool:
        """
        Delete job from store.

        Args:
            job_id: Job ID to delete

        Returns:
            True if deleted
        """
        job_key = f"{self.JOB_PREFIX}{job_id}"
        try:
            # Drop the status-index entry too, or the ID outlives the job it
            # points at and the index over-counts forever (SCARD has no way to
            # notice the hash is gone). Mirrors async delete_job.
            raw = self.redis.hget(job_key, "status")
            status_val = raw.decode() if isinstance(raw, bytes) else raw
            pipe = self.redis.pipeline()
            pipe.delete(job_key)
            if status_val:
                pipe.srem(self._status_key(status_val), job_id)
            results = pipe.execute()
            result = results[0] if results else 0
            if result:
                logger.info("Deleted job %s", job_id[:8])
            return result > 0
        except RedisError as e:
            logger.error("Failed to delete job %s: %s", job_id[:8], e)
            return False

    # =========================================================================
    # Pub/Sub Operations
    # =========================================================================

    def publish_event(self, job_id: str, event: Dict[str, Any]) -> int:
        """
        Publish event for a job.

        Args:
            job_id: Job ID
            event: Event data (will be JSON serialized)

        Returns:
            Number of subscribers that received the message
        """
        channel = f"{self.JOB_PREFIX}{job_id}:events"
        try:
            message = json.dumps(event)
            return self.redis.publish(channel, message)
        except RedisError as e:
            logger.error("Failed to publish event for job %s: %s", job_id[:8], e)
            return 0

    # =========================================================================
    # Worker Registry Operations
    # =========================================================================

    def register_worker(self, worker: WorkerInfo) -> bool:
        """
        Register a worker.

        Args:
            worker: Worker info

        Returns:
            True if successful
        """
        worker_key = f"{self.WORKER_PREFIX}{worker.id}"
        try:
            pipe = self.redis.pipeline()
            pipe.hset(worker_key, mapping=worker.to_dict())
            pipe.hset(self.WORKERS_KEY, worker.id, worker_key)
            pipe.execute()
            logger.info("Registered worker %s (GPU %d)", worker.id, worker.gpu_id)
            return True
        except RedisError as e:
            logger.error("Failed to register worker %s: %s", worker.id, e)
            return False

    def unregister_worker(self, worker_id: str) -> bool:
        """
        Unregister a worker.

        Args:
            worker_id: Worker ID

        Returns:
            True if successful
        """
        worker_key = f"{self.WORKER_PREFIX}{worker_id}"
        try:
            pipe = self.redis.pipeline()
            pipe.delete(worker_key)
            pipe.hdel(self.WORKERS_KEY, worker_id)
            pipe.execute()
            logger.info("Unregistered worker %s", worker_id)
            return True
        except RedisError as e:
            logger.error("Failed to unregister worker %s: %s", worker_id, e)
            return False

    def update_worker_heartbeat(self, worker_id: str) -> bool:
        """
        Update worker heartbeat timestamp.

        Args:
            worker_id: Worker ID

        Returns:
            True if successful
        """
        worker_key = f"{self.WORKER_PREFIX}{worker_id}"
        try:
            self.redis.hset(worker_key, "last_heartbeat", datetime.now(timezone.utc).isoformat())
            return True
        except RedisError:
            return False

    def update_worker_status(self, worker_id: str, status: str, current_job_id: Optional[str] = None) -> bool:
        """
        Update worker status and current job.

        Args:
            worker_id: Worker ID
            status: New status (idle, busy, offline)
            current_job_id: Current job ID (if busy)

        Returns:
            True if successful
        """
        worker_key = f"{self.WORKER_PREFIX}{worker_id}"
        try:
            updates = {
                "status": status,
                "current_job_id": current_job_id or "",
                "last_heartbeat": datetime.now(timezone.utc).isoformat(),
            }
            self.redis.hset(worker_key, mapping=updates)
            return True
        except RedisError as e:
            logger.error("Failed to update worker %s status: %s", worker_id, e)
            return False

    def get_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        """
        Get worker info.

        Args:
            worker_id: Worker ID

        Returns:
            WorkerInfo or None
        """
        worker_key = f"{self.WORKER_PREFIX}{worker_id}"
        try:
            data = self.redis.hgetall(worker_key)
            if not data:
                return None
            return WorkerInfo.from_dict(data)
        except RedisError:
            return None

    def list_workers(self, status: Optional[str] = None) -> List[WorkerInfo]:
        """
        List all registered workers.

        Args:
            status: Optional status filter

        Returns:
            List of WorkerInfo
        """
        try:
            worker_ids = self.redis.hkeys(self.WORKERS_KEY)
            workers = []

            for worker_id in worker_ids:
                worker_id_str = worker_id.decode() if isinstance(worker_id, bytes) else worker_id
                worker = self.get_worker(worker_id_str)
                if worker:
                    if status is None or worker.status == status:
                        workers.append(worker)

            return workers
        except RedisError as e:
            logger.error("Failed to list workers: %s", e)
            return []

    def cleanup_stale_workers(self, timeout_seconds: int = 60) -> int:
        """
        Remove workers that haven't sent heartbeat.

        Args:
            timeout_seconds: Seconds since last heartbeat to consider stale

        Returns:
            Number of workers removed
        """
        workers = self.list_workers()
        now = datetime.now(timezone.utc)
        removed = 0

        for worker in workers:
            if worker.last_heartbeat:
                age = (now - worker.last_heartbeat).total_seconds()
                if age > timeout_seconds:
                    # Requeue any job the worker was running
                    if worker.current_job_id:
                        self.requeue_job(worker.current_job_id, to_front=True)
                    self.unregister_worker(worker.id)
                    removed += 1
                    logger.warning("Removed stale worker %s (last heartbeat: %ds ago)", worker.id, int(age))

        return removed
