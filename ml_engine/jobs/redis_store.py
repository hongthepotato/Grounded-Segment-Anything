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
from datetime import datetime
from typing import Dict, Any, Optional, List, Iterator, Callable
import threading

import redis
from redis.exceptions import RedisError

from ml_engine.jobs.models import Job, JobStatus, JobProgress, WorkerInfo

logger = logging.getLogger(__name__)


class RedisJobStore:
    """
    Redis-based storage for training jobs.
    
    Thread-safe operations for:
    - Job queue management (priority queue via sorted sets)
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
            max_connections=20
        )
        self.redis = redis.Redis(connection_pool=self.pool)
        
        # Pub/sub client (separate connection)
        self._pubsub_lock = threading.Lock()
        self._pubsub: Optional[redis.client.PubSub] = None
        
        # Test connection
        try:
            self.redis.ping()
            logger.info("Connected to Redis at %s", redis_url)
        except RedisError as e:
            logger.error("Failed to connect to Redis: %s", e)
            raise
    
    def close(self):
        """Close Redis connections."""
        if self._pubsub:
            self._pubsub.close()
        self.pool.disconnect()
        logger.info("Redis connections closed")
    
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
            # Store job state
            job_data = job.to_dict()
            # Convert all values to strings for Redis HSET
            job_data_str = {k: str(v) if not isinstance(v, str) else v 
                          for k, v in job_data.items()}
            pipe.hset(job_key, mapping=job_data_str)
            
            # Add to queue (RPUSH for FIFO)
            # Use priority score: higher priority = earlier in queue
            # We use LPUSH for high priority, RPUSH for normal
            if job.priority > 0:
                pipe.lpush(self.JOB_QUEUE_KEY, job.id)
            else:
                pipe.rpush(self.JOB_QUEUE_KEY, job.id)
            
            pipe.execute()
            logger.info("Enqueued job %s (priority=%d)", job.id[:8], job.priority)
            
            # Publish enqueue event
            self.publish_event(job.id, {
                "type": "job_enqueued",
                "job_id": job.id,
                "status": job.status.value,
                "timestamp": datetime.utcnow().isoformat()
            })
            
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
        except RedisError:
            return 0
    
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
                elif isinstance(value, dict):
                    redis_updates[key] = json.dumps(value)
                elif value is None:
                    redis_updates[key] = ""
                else:
                    redis_updates[key] = str(value)
            
            if redis_updates:
                self.redis.hset(job_key, mapping=redis_updates)
                
                # Publish update event
                self.publish_event(job_id, {
                    "type": "job_updated",
                    "job_id": job_id,
                    "updates": {k: str(v) for k, v in updates.items()},
                    "timestamp": datetime.utcnow().isoformat()
                })
                
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
        offset: int = 0
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
                cursor, keys = self.redis.scan(
                    cursor=cursor, 
                    match=f"{self.JOB_PREFIX}*",
                    count=100
                )
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
            return jobs[offset:offset + limit]
            
        except RedisError as e:
            logger.error("Failed to list jobs: %s", e)
            return []
    
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
            result = self.redis.delete(job_key)
            if result:
                logger.info("Deleted job %s", job_id[:8])
            return result > 0
        except RedisError as e:
            logger.error("Failed to delete job %s: %s", job_id[:8], e)
            return False
    
    def job_exists(self, job_id: str) -> bool:
        """Check if job exists."""
        job_key = f"{self.JOB_PREFIX}{job_id}"
        try:
            return self.redis.exists(job_key) > 0
        except RedisError:
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
    
    def subscribe_to_job(self, job_id: str) -> Iterator[Dict[str, Any]]:
        """
        Subscribe to job events (blocking iterator).
        
        Yields events until the connection is closed or unsubscribed.
        
        Args:
            job_id: Job ID to subscribe to
            
        Yields:
            Event dictionaries
        """
        channel = f"{self.JOB_PREFIX}{job_id}:events"
        pubsub = self.redis.pubsub()
        
        try:
            pubsub.subscribe(channel)
            logger.debug("Subscribed to job %s events", job_id[:8])
            
            for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        data = message["data"]
                        if isinstance(data, bytes):
                            data = data.decode()
                        event = json.loads(data)
                        yield event
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON in event: %s", message["data"])
                        
        except RedisError as e:
            logger.error("Pub/sub error for job %s: %s", job_id[:8], e)
        finally:
            pubsub.unsubscribe(channel)
            pubsub.close()
    
    def subscribe_to_job_async(
        self, 
        job_id: str, 
        callback: Callable[[Dict[str, Any]], None],
        stop_event: Optional[threading.Event] = None
    ) -> threading.Thread:
        """
        Subscribe to job events in a background thread.
        
        Args:
            job_id: Job ID to subscribe to
            callback: Function to call with each event
            stop_event: Event to signal stop (optional)
            
        Returns:
            Thread object (already started)
        """
        def subscriber():
            channel = f"{self.JOB_PREFIX}{job_id}:events"
            pubsub = self.redis.pubsub()
            
            try:
                pubsub.subscribe(channel)
                
                while stop_event is None or not stop_event.is_set():
                    message = pubsub.get_message(timeout=1.0)
                    if message and message["type"] == "message":
                        try:
                            data = message["data"]
                            if isinstance(data, bytes):
                                data = data.decode()
                            event = json.loads(data)
                            callback(event)
                        except (json.JSONDecodeError, Exception) as e:
                            logger.warning("Error processing event: %s", e)
                            
            except RedisError as e:
                logger.error("Async pub/sub error: %s", e)
            finally:
                pubsub.unsubscribe(channel)
                pubsub.close()
        
        thread = threading.Thread(target=subscriber, daemon=True)
        thread.start()
        return thread
    
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
            self.redis.hset(worker_key, mapping=worker.to_dict())
            self.redis.hset(self.WORKERS_KEY, worker.id, worker_key)
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
            self.redis.hset(worker_key, "last_heartbeat", datetime.utcnow().isoformat())
            return True
        except RedisError:
            return False
    
    def update_worker_status(
        self, 
        worker_id: str, 
        status: str, 
        current_job_id: Optional[str] = None
    ) -> bool:
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
                "last_heartbeat": datetime.utcnow().isoformat()
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
        now = datetime.utcnow()
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
                    logger.warning("Removed stale worker %s (last heartbeat: %ds ago)", 
                                 worker.id, int(age))
        
        return removed
