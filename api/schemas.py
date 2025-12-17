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
                "image_dir": "data/images",
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


# =============================================================================
# Auto-Labeling Schemas
# =============================================================================

class AutoLabelRequest(BaseModel):
    """
    Request body for auto-labeling job submission.
    
    Example:
        {
            "image_dir": "/data/raw/images",
            "classes": ["ear of bag", "defect", "label"],
            "output_mode": "boxes",
            "box_threshold": 0.5
        }
    """
    image_dir: str = Field(
        ...,
        description="Path to images directory on server"
    )
    classes: List[str] = Field(
        ...,
        description="List of class names to detect"
    )
    output_mode: str = Field(
        default="boxes",
        description="Output mode: 'boxes', 'masks', or 'both'"
    )
    box_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Detection confidence threshold"
    )
    text_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Text matching threshold"
    )
    nms_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Non-Maximum Suppression threshold"
    )
    output_dir: Optional[str] = Field(
        default=None,
        description="Output directory (auto-generated if not provided)"
    )
    priority: int = Field(
        default=0,
        description="Job priority (higher = more urgent)"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Optional tags for filtering"
    )


class COCOImageSchema(BaseModel):
    """COCO image entry."""
    id: int = Field(..., description="Image ID")
    file_name: str = Field(..., description="Image filename")
    width: int = Field(..., description="Image width")
    height: int = Field(..., description="Image height")


class COCOAnnotationSchema(BaseModel):
    """COCO annotation entry."""
    id: int = Field(..., description="Annotation ID")
    image_id: int = Field(..., description="Image ID")
    category_id: int = Field(..., description="Category ID")
    bbox: Optional[List[float]] = Field(default=None, description="Bounding box [x, y, w, h]")
    segmentation: Optional[List[List[float]]] = Field(default=None, description="Polygon segmentation")
    area: Optional[float] = Field(default=None, description="Area in pixels")
    score: Optional[float] = Field(default=None, description="Detection confidence")
    iscrowd: int = Field(default=0, description="Is crowd annotation")


class COCOCategorySchema(BaseModel):
    """COCO category entry."""
    id: int = Field(..., description="Category ID")
    name: str = Field(..., description="Category name")


class AutoLabelResultResponse(BaseModel):
    """
    COCO-format annotations response.
    
    Contains complete COCO JSON structure with images, annotations, and categories.
    """
    images: List[COCOImageSchema] = Field(..., description="List of images")
    annotations: List[COCOAnnotationSchema] = Field(..., description="List of annotations")
    categories: List[COCOCategorySchema] = Field(..., description="List of categories")


class VisualizationInfo(BaseModel):
    """Information about a single visualization image."""
    filename: str = Field(..., description="Visualization filename")
    original: str = Field(..., description="Original image filename")
    annotation_count: int = Field(..., description="Number of annotations in this image")


class VisualizationListResponse(BaseModel):
    """List of visualization images for an auto-label job."""
    job_id: str = Field(..., description="Job ID")
    total: int = Field(..., description="Total number of visualizations")
    images: List[VisualizationInfo] = Field(..., description="List of visualization info")
