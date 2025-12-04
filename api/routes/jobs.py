"""
REST endpoints for job management.

Provides:
- POST /api/jobs - Submit new job
- GET /api/jobs - List all jobs
- GET /api/jobs/{id} - Get job details
- DELETE /api/jobs/{id} - Cancel job
- GET /api/jobs/{id}/logs - Get job logs (future)
- GET /api/queue/status - Get queue status
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Query

from api.schemas import (
    JobCreate,
    JobResponse,
    JobListResponse,
    JobProgressSchema,
    QueueStatusResponse,
    WorkerResponse,
)
from ml_engine.jobs import JobManager, get_job_manager, Job

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def get_manager() -> JobManager:
    """Dependency to get JobManager instance."""
    import os
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    return get_job_manager(redis_url)


def job_to_response(job: Job) -> JobResponse:
    """Convert Job model to JobResponse schema."""
    progress = None
    if job.progress:
        progress = JobProgressSchema(
            current_epoch=job.progress.current_epoch,
            total_epochs=job.progress.total_epochs,
            current_step=job.progress.current_step,
            total_steps=job.progress.total_steps,
            metrics=job.progress.metrics,
            message=job.progress.message,
        )
    
    return JobResponse(
        id=job.id,
        type=job.type,
        status=job.status.value,
        config=job.config,
        progress=progress,
        worker_id=job.worker_id,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error_message=job.error_message,
        output_dir=job.output_dir,
        priority=job.priority,
        tags=job.tags,
    )


@router.post("", response_model=JobResponse, status_code=201)
async def submit_job(
    request: JobCreate,
    manager: JobManager = Depends(get_manager)
):
    """
    Submit a new training job.
    
    The job is queued and will be executed by an available worker.
    Returns immediately with job details.
    
    Example:
        POST /api/jobs
        {
            "job_type": "teacher_training",
            "config": {
                "data_path": "data/annotations.json",
                "training": {"epochs": 50}
            }
        }
    """
    try:
        job = manager.submit_job(
            job_type=request.job_type,
            config=request.config,
            priority=request.priority,
            output_dir=request.output_dir,
            tags=request.tags,
        )
        logger.info("Submitted job %s (type=%s)", job.id[:8], request.job_type)
        return job_to_response(job)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to submit job: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}")


@router.get("", response_model=JobListResponse)
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    job_type: Optional[str] = Query(None, description="Filter by job type"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum jobs to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    manager: JobManager = Depends(get_manager)
):
    """
    List jobs with optional filtering.
    
    Example:
        GET /api/jobs?status=running&limit=10
    """
    jobs = manager.list_jobs(
        status=status,
        job_type=job_type,
        limit=limit,
        offset=offset,
    )
    
    # Get total count for pagination
    total = manager.get_job_count(status=status)
    
    return JobListResponse(
        jobs=[job_to_response(job) for job in jobs],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    manager: JobManager = Depends(get_manager)
):
    """
    Get job details by ID.
    
    Example:
        GET /api/jobs/a1b2c3d4-...
    """
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return job_to_response(job)


@router.delete("/{job_id}", status_code=200)
async def cancel_job(
    job_id: str,
    manager: JobManager = Depends(get_manager)
):
    """
    Cancel a job.
    
    - If PENDING: Removes from queue and marks CANCELLED
    - If RUNNING: Sets status to CANCELLING, worker will stop gracefully
    - If already terminal: Returns 400
    
    Example:
        DELETE /api/jobs/a1b2c3d4-...
    """
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if job.is_terminal:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job in terminal state: {job.status.value}"
        )
    
    success = manager.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to cancel job")
    
    # Get updated job
    job = manager.get_job(job_id)
    return job_to_response(job)


# Queue status endpoint (separate router for clarity)
queue_router = APIRouter(prefix="/api/queue", tags=["queue"])


@queue_router.get("/status", response_model=QueueStatusResponse)
async def get_queue_status(
    manager: JobManager = Depends(get_manager)
):
    """
    Get queue status including pending jobs and active workers.
    
    Example:
        GET /api/queue/status
    """
    status = manager.get_queue_status()
    
    workers = [
        WorkerResponse(
            id=w["id"],
            gpu_id=int(w["gpu_id"]),
            hostname=w["hostname"],
            status=w["status"],
            current_job_id=w.get("current_job_id") or None,
            last_heartbeat=w.get("last_heartbeat"),
            started_at=w.get("started_at"),
        )
        for w in status["workers"]
    ]
    
    return QueueStatusResponse(
        queue_length=status["queue_length"],
        workers=workers,
        job_counts=status["job_counts"],
    )





