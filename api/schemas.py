"""
Pydantic schemas for API request/response models.

This module defines:
- JobCreate: Request body for job submission
- JobResponse: Response body for job details
- JobListResponse: Response body for job list
- ProgressResponse: Progress information
- QueueStatusResponse: Queue status information
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class JobProgressSchema(BaseModel):
    """Progress information during training."""
    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0
    total_steps: int = 0
    metrics: Dict[str, float] = Field(default_factory=dict)
    message: str = ""

    class Config:
        from_attributes = True


class JobCreate(BaseModel):
    """
    Request body for job submission.
    
    Example:
        {
            "job_type": "teacher_training",
            "config": {
                "data_path": "data/annotations.json",
                "image_paths": [
                    "/profile/upload/2025/12/16/xxx1.jpeg",
                    "/profile/upload/2025/12/16/xxx2.jpeg"
                ],
                "training": {"epochs": 50, "batch_size": 8}
            },
            "priority": 0,
            "tags": ["experiment1"]
        }
    """
    job_type: str = Field(
        ...,
        description="Type of job (teacher_training, student_distillation)"
    )
    config: Dict[str, Any] = Field(
        ...,
        description="Job configuration (data paths, hyperparameters)"
    )
    priority: int = Field(
        default=0,
        description="Job priority (higher = more urgent)"
    )
    output_dir: Optional[str] = Field(
        default=None,
        description="Output directory (auto-generated if not provided)"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Optional tags for filtering"
    )


class JobResponse(BaseModel):
    """
    Response body for job details.
    
    Example:
        {
            "id": "a1b2c3d4-...",
            "type": "teacher_training",
            "status": "running",
            "progress": {"current_epoch": 5, "total_epochs": 50, ...},
            "created_at": "2024-01-01T12:00:00Z",
            ...
        }
    """
    id: str = Field(..., description="Job UUID")
    type: str = Field(..., description="Job type")
    status: str = Field(..., description="Job status")
    config: Dict[str, Any] = Field(default_factory=dict, description="Job configuration")
    progress: Optional[JobProgressSchema] = Field(default=None, description="Training progress")
    worker_id: Optional[str] = Field(default=None, description="Worker executing job")
    created_at: Optional[datetime] = Field(default=None, description="Creation timestamp")
    started_at: Optional[datetime] = Field(default=None, description="Start timestamp")
    finished_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    output_dir: Optional[str] = Field(default=None, description="Output directory")
    priority: int = Field(default=0, description="Job priority")
    tags: List[str] = Field(default_factory=list, description="Job tags")

    class Config:
        from_attributes = True


class JobListResponse(BaseModel):
    """Response body for job list."""
    jobs: List[JobResponse] = Field(..., description="List of jobs")
    total: int = Field(..., description="Total number of jobs matching filter")
    limit: int = Field(..., description="Pagination limit")
    offset: int = Field(..., description="Pagination offset")


class WorkerResponse(BaseModel):
    """Response body for worker details."""
    id: str = Field(..., description="Worker ID")
    gpu_id: int = Field(..., description="GPU device ID")
    hostname: str = Field(..., description="Machine hostname")
    status: str = Field(..., description="Worker status (idle, busy, offline)")
    current_job_id: Optional[str] = Field(default=None, description="Current job ID")
    last_heartbeat: Optional[datetime] = Field(default=None, description="Last heartbeat")
    started_at: Optional[datetime] = Field(default=None, description="Worker start time")

    class Config:
        from_attributes = True


class QueueStatusResponse(BaseModel):
    """Response body for queue status."""
    queue_length: int = Field(..., description="Number of pending jobs in queue")
    workers: List[WorkerResponse] = Field(..., description="Active workers")
    job_counts: Dict[str, int] = Field(..., description="Job counts by status")


class ErrorResponse(BaseModel):
    """Error response body."""
    detail: str = Field(..., description="Error message")


class WebSocketEvent(BaseModel):
    """WebSocket event message."""
    type: str = Field(..., description="Event type")
    job_id: str = Field(..., description="Job ID")
    timestamp: str = Field(..., description="Event timestamp (ISO format)")
    progress: Optional[JobProgressSchema] = Field(default=None, description="Progress info")
    error: Optional[str] = Field(default=None, description="Error message")
    output_dir: Optional[str] = Field(default=None, description="Output directory")
