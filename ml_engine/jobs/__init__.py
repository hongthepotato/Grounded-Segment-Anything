"""
Job management module for training pipeline.

This module provides:
- Job models and status tracking
- Redis-based job queue and state persistence
- Worker process for executing training jobs
- JobManager for API interaction

Usage:
    from ml_engine.jobs import JobManager, Job, JobStatus
    
    manager = JobManager(redis_url="redis://localhost:6379")
    job = manager.submit_job(job_type="teacher_training", config={...})
    status = manager.get_job(job.id)
"""

from ml_engine.jobs.models import Job, JobStatus, JobProgress, JobType, WorkerInfo
from ml_engine.jobs.manager import JobManager, get_job_manager
from ml_engine.jobs.redis_store import RedisJobStore

__all__ = [
    # Models
    "Job",
    "JobStatus",
    "JobProgress",
    "JobType",
    "WorkerInfo",
    # Manager
    "JobManager",
    "get_job_manager",
    # Store
    "RedisJobStore",
]
