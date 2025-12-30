"""
Job Manager for training pipeline.

This module provides the JobManager class that serves as the main API
for job submission and management. Used by the FastAPI routes.

Features:
- Submit jobs to Redis queue
- Cancel running/pending jobs
- Query job status and progress
- List jobs with filtering
- Subscribe to job events

Usage:
    from ml_engine.jobs import JobManager
    
    manager = JobManager(redis_url="redis://localhost:6379")
    
    # Submit a job
    job = manager.submit_job(
        job_type="teacher_training",
        config={"data_path": "data/annotations.json", ...}
    )
    
    # Check status
    job = manager.get_job(job.id)
    print(job.status, job.progress)
    
    # Cancel job
    manager.cancel_job(job.id)
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Iterator, Callable

from ml_engine.jobs.models import Job, JobStatus, JobType, WorkerInfo
from ml_engine.jobs.redis_store import RedisJobStore

logger = logging.getLogger(__name__)


class JobManager:
    """
    High-level job management API.
    
    This class provides a clean interface for:
    - Job submission
    - Job cancellation
    - Status queries
    - Event subscription
    
    Thread-safe: All operations delegate to RedisJobStore which handles
    thread safety via Redis atomic operations.
    
    Example:
        >>> manager = JobManager()
        >>> 
        >>> # Submit teacher training job
        >>> job = manager.submit_job(
        ...     job_type="teacher_training",
        ...     config={
        ...         "data_path": "data/annotations.json",
        ...         "image_paths": ["/profile/upload/2025/12/16/xxx.jpeg", ...],
        ...         "training": {"epochs": 50, "batch_size": 8}
        ...     }
        ... )
        >>> print(f"Submitted job {job.id}")
        >>> 
        >>> # Wait for completion
        >>> for event in manager.subscribe_to_job(job.id):
        ...     if event["type"] == "progress":
        ...         print(f"Progress: {event['progress']}")
        ...     elif event["type"] == "job_completed":
        ...         print("Done!")
        ...         break
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize JobManager.
        
        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self.store = RedisJobStore(redis_url)
        logger.info("JobManager initialized with Redis at %s", redis_url)

    def close(self):
        """Close Redis connection."""
        self.store.close()

    # =========================================================================
    # Job Submission
    # =========================================================================

    def submit_job(
        self,
        job_type: str,
        config: Dict[str, Any],
        priority: int = 0,
        output_dir: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Job:
        """
        Submit a new training job.
        
        Creates job, stores in Redis, and adds to queue.
        Returns immediately - job runs asynchronously.
        
        Args:
            job_type: Type of job (teacher_training, student_distillation)
            config: Job configuration (data paths, hyperparameters)
            priority: Job priority (higher = more urgent)
            output_dir: Optional output directory (auto-generated if not provided)
            tags: Optional tags for filtering
            
        Returns:
            Created Job object
            
        Example:
            >>> job = manager.submit_job(
            ...     job_type="teacher_training",
            ...     config={
            ...         "data_path": "data/annotations.json",
            ...         "training": {"epochs": 50}
            ...     },
            ...     priority=1,  # High priority
            ...     tags=["production"]
            ... )
        """
        # Validate job type
        try:
            JobType(job_type)
        except ValueError as e:
            valid_types = [t.value for t in JobType]
            raise ValueError(f"Invalid job type: {job_type}. Must be one of: {valid_types}") from e

        # Create job
        job = Job(
            type=job_type,
            status=JobStatus.PENDING,
            config=config,
            priority=priority,
            output_dir=output_dir,
            tags=tags or []
        )

        logger.info("Submitting job %s (type=%s, priority=%d)",
                   job.id[:8], job_type, priority)

        # Enqueue (stores job and adds to queue atomically)
        self.store.enqueue_job(job)

        return job

    # =========================================================================
    # Job Cancellation
    # =========================================================================

    def cancel_job(self, job_id: str) -> bool:
        """
        Request job cancellation.
        
        - If PENDING: Removes from queue and marks CANCELLED
        - If RUNNING: Sets status to CANCELLING, worker will stop gracefully
        - If already terminal: Returns False
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            True if cancellation initiated, False if job not found or already terminal
        """
        job = self.store.get_job(job_id)
        if job is None:
            logger.warning("Cannot cancel job %s: not found", job_id[:8])
            return False

        if job.is_terminal:
            logger.info("Cannot cancel job %s: already in terminal state %s",
                       job_id[:8], job.status.value)
            return False

        if job.status == JobStatus.PENDING:
            # Job is in queue - mark as cancelled directly
            self.store.update_job(
                job_id,
                status=JobStatus.CANCELLED,
                finished_at=datetime.now()
            )
            # Note: Job will be skipped when worker tries to execute it
            logger.info("Cancelled pending job %s", job_id[:8])

        elif job.status == JobStatus.RUNNING:
            # Job is running - request graceful cancellation
            self.store.update_job(job_id, status=JobStatus.CANCELLING)
            logger.info("Requested cancellation for running job %s", job_id[:8])

        elif job.status == JobStatus.CANCELLING:
            # Already cancelling
            logger.info("Job %s is already cancelling", job_id[:8])

        # Publish cancellation event
        self.store.publish_event(job_id, {
            "type": "cancel_requested",
            "job_id": job_id,
            "timestamp": datetime.now().isoformat()
        })

        return True

    # =========================================================================
    # Job Queries
    # =========================================================================

    def get_job(self, job_id: str) -> Optional[Job]:
        """
        Get job by ID.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job object or None if not found
        """
        return self.store.get_job(job_id)

    def list_jobs(
        self,
        status: Optional[str] = None,
        job_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Job]:
        """
        List jobs with optional filtering.
        
        Args:
            status: Filter by status (pending, running, completed, etc.)
            job_type: Filter by job type
            limit: Maximum number of jobs to return
            offset: Pagination offset
            
        Returns:
            List of Job objects (sorted by created_at, newest first)
        """
        # Convert status string to enum
        status_enum = None
        if status:
            try:
                status_enum = JobStatus(status)
            except ValueError:
                logger.warning("Invalid status filter: %s", status)

        return self.store.list_jobs(
            status=status_enum,
            job_type=job_type,
            limit=limit,
            offset=offset
        )

    def get_job_count(self, status: Optional[str] = None) -> int:
        """
        Get count of jobs by status.
        
        Args:
            status: Optional status filter
            
        Returns:
            Number of jobs
        """
        jobs = self.list_jobs(status=status, limit=10000)
        return len(jobs)

    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job from the store.
        
        Only allows deletion of terminal jobs (completed, failed, cancelled).
        
        Args:
            job_id: Job ID to delete
            
        Returns:
            True if deleted, False otherwise
        """
        job = self.store.get_job(job_id)
        if job is None:
            return False

        if not job.is_terminal:
            logger.warning("Cannot delete non-terminal job %s (status=%s)",
                          job_id[:8], job.status.value)
            return False

        return self.store.delete_job(job_id)

    # =========================================================================
    # Queue Info
    # =========================================================================

    def get_queue_length(self) -> int:
        """Get number of pending jobs in queue."""
        return self.store.get_queue_length()

    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get overall queue status.
        
        Returns:
            Dict with queue length, active workers, job counts by status
        """
        return {
            "queue_length": self.store.get_queue_length(),
            "workers": [w.to_dict() for w in self.store.list_workers()],
            "job_counts": {
                "pending": self.get_job_count("pending"),
                "running": self.get_job_count("running"),
                "completed": self.get_job_count("completed"),
                "failed": self.get_job_count("failed"),
                "cancelled": self.get_job_count("cancelled"),
            }
        }

    # =========================================================================
    # Event Subscription
    # =========================================================================

    def subscribe_to_job(self, job_id: str) -> Iterator[Dict[str, Any]]:
        """
        Subscribe to job events (blocking iterator).
        
        Yields events until the job reaches a terminal state.
        
        Args:
            job_id: Job ID to subscribe to
            
        Yields:
            Event dictionaries with type, job_id, timestamp, and event-specific data
            
        Example:
            >>> for event in manager.subscribe_to_job(job_id):
            ...     if event["type"] == "progress":
            ...         print(f"Epoch {event['progress']['current_epoch']}")
            ...     elif event["type"] == "job_completed":
            ...         break
        """
        return self.store.subscribe_to_job(job_id)

    def subscribe_to_job_async(
        self,
        job_id: str,
        callback: Callable[[Dict[str, Any]], None]
    ):
        """
        Subscribe to job events in background thread.
        
        Args:
            job_id: Job ID to subscribe to
            callback: Function called with each event
            
        Returns:
            Thread object (already started)
        """
        return self.store.subscribe_to_job_async(job_id, callback)

    # =========================================================================
    # Worker Management
    # =========================================================================

    def list_workers(self, status: Optional[str] = None) -> List[WorkerInfo]:
        """
        List registered workers.
        
        Args:
            status: Optional status filter (idle, busy, offline)
            
        Returns:
            List of WorkerInfo objects
        """
        return self.store.list_workers(status=status)

    def cleanup_stale_workers(self, timeout_seconds: int = 60) -> int:
        """
        Remove workers that haven't sent heartbeat.
        
        Also requeues any jobs those workers were running.
        
        Args:
            timeout_seconds: Seconds since last heartbeat to consider stale
            
        Returns:
            Number of workers removed
        """
        return self.store.cleanup_stale_workers(timeout_seconds)


# Singleton for convenience
_default_manager: Optional[JobManager] = None


def get_job_manager(redis_url: str = "redis://localhost:6379") -> JobManager:
    """
    Get or create default JobManager instance.
    
    Args:
        redis_url: Redis connection URL
        
    Returns:
        JobManager instance
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = JobManager(redis_url)
    return _default_manager
