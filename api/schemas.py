"""
Pydantic schemas for API request/response models.

This module defines:
- ApiResponse: Unified wrapper for all API responses
- JobCreate: Request body for job submission
- JobResponse: Response body for job details
- JobListResponse: Response body for job list
- ProgressResponse: Progress information
- QueueStatusResponse: Queue status information
"""

import math
from datetime import datetime
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar

from pydantic import BaseModel, Field, model_validator

# Generic type for wrapped data
T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """
    Unified API response wrapper.

    All API responses are wrapped in this format for consistency.
    Frontend checks 'status' field to determine success or failure.

    Success example:
        {
            "code": 200,
            "status": "succeed",
            "data": { "id": "abc123", ... }
        }

    Error example:
        {
            "code": 422,
            "status": "failed",
            "error": "Validation failed: invalid job type"
        }
    """

    code: int = Field(..., ge=100, le=599, description="HTTP status code")
    status: Literal["succeed", "failed"] = Field(..., description="Business status")
    data: Optional[T] = Field(default=None, description="Response data (present when succeed)")
    error: Optional[str] = Field(default=None, description="Error message (present when failed)")

    class Config:
        from_attributes = True


def success_response(data: Any = None, code: int = 200) -> dict:
    """
    Helper function to create a success response.

    Args:
        data: Response data (task info, job details, etc.)
        code: HTTP status code (default 200)

    Returns:
        {
            "code": 200,
            "status": "succeed",
            "data": { ... }
        }
    """
    return {"code": code, "status": "succeed", "data": data}


def error_response(error: str, code: int = 400) -> dict:
    """
    Helper function to create an error response.

    Args:
        error: Error message describing what went wrong
        code: HTTP status code (default 400)

    Returns:
        {
            "code": 422,
            "status": "failed",
            "error": "Error message here"
        }
    """
    return {"code": code, "status": "failed", "error": error}


class JobProgressSchema(BaseModel):
    """Progress information during training."""

    current_epoch: int = Field(default=0, ge=0)
    total_epochs: int = Field(default=0, ge=0)
    current_step: int = Field(default=0, ge=0)
    total_steps: int = Field(default=0, ge=0)
    metrics: Dict[str, float] = Field(default_factory=dict)
    message: str = ""
    overall_progress: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall training progress (0.0 to 1.0)"
    )

    class Config:
        from_attributes = True

    @model_validator(mode="after")
    def _check_epoch_step_invariants(self) -> "JobProgressSchema":
        """Trainer-controlled invariants: current must not exceed total.
        A trainer that reports current_epoch=99 with total_epochs=10 has a
        bug — catch at the schema boundary instead of letting bad progress
        leak into the UI."""
        if self.current_epoch > self.total_epochs:
            raise ValueError(
                f"current_epoch ({self.current_epoch}) must be <= total_epochs ({self.total_epochs})"
            )
        if self.current_step > self.total_steps:
            raise ValueError(
                f"current_step ({self.current_step}) must be <= total_steps ({self.total_steps})"
            )
        return self


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

    job_type: Literal["teacher_training", "student_distillation"] = Field(..., description="Type of job")
    config: Dict[str, Any] = Field(..., description="Job configuration (data paths, hyperparameters)")
    priority: int = Field(default=0, description="Job priority (higher = more urgent)")
    output_dir: Optional[str] = Field(
        default=None, description="Output directory (auto-generated if not provided)"
    )
    tags: List[str] = Field(default_factory=list, description="Optional tags for filtering")


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
            "accuracy": 85.5,
            ...
        }
    """

    id: str = Field(..., description="Job UUID")
    type: str = Field(..., description="Job type")
    status: str = Field(..., description="Job status")
    progress: Optional[JobProgressSchema] = Field(default=None, description="Training progress")
    worker_id: Optional[str] = Field(default=None, description="Worker executing job")
    created_at: Optional[datetime] = Field(default=None, description="Creation timestamp")
    started_at: Optional[datetime] = Field(default=None, description="Start timestamp")
    finished_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    output_dir: Optional[str] = Field(default=None, description="Output directory")
    duration_seconds: Optional[float] = Field(
        default=None, description="Elapsed seconds (started_at to finished_at)"
    )
    accuracy: Optional[float] = Field(
        default=None, description="Model accuracy score (0-100) from evaluation"
    )
    # Commented out - not needed by frontend for now
    # priority: int = Field(default=0, description="Job priority")
    # tags: List[str] = Field(default_factory=list, description="Job tags")

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
    status: Literal["idle", "busy", "offline"] = Field(..., description="Worker status")
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
            "image_paths": [
                "upload/2025/12/16/xxx1.jpeg",
                "upload/2025/12/16/xxx2.jpeg"
            ],
            "classes": ["ear of bag", "defect", "label"],
            "output_mode": "boxes",
            "box_threshold": 0.5
        }
    """

    image_paths: List[str] = Field(..., min_length=1, description="List of image paths")
    classes: List[str] = Field(..., min_length=1, description="List of class names to detect")
    output_mode: Literal["boxes", "masks", "both"] = Field(default="boxes", description="Output mode")
    box_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Detection confidence threshold")
    text_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Text matching threshold")
    nms_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Non-Maximum Suppression threshold")
    output_dir: Optional[str] = Field(
        default=None, description="Output directory (auto-generated if not provided)"
    )
    priority: int = Field(default=0, description="Job priority (higher = more urgent)")
    tags: List[str] = Field(default_factory=list, description="Optional tags for filtering")


class DistillationRequest(BaseModel):
    """
    Request body for student distillation submission.

    This is a convenience schema for POST /api/distillation and maps to
    job_type="student_distillation" in the jobs queue.
    """

    data_path: str = Field(..., description="Path to labeled COCO annotations JSON")
    image_paths: List[str] = Field(..., description="Labeled image file paths")
    teacher_dir: Optional[str] = Field(
        default=None, description="Teacher output directory (required with unlabeled_image_paths)"
    )
    unlabeled_image_paths: Optional[List[str]] = Field(
        default=None, description="Unlabeled image file paths (required with teacher_dir)"
    )
    student_model: Optional[str] = Field(
        default=None, description="Optional direct student model name override"
    )
    student_size: Optional[str] = Field(default=None, description="Optional student size enum: n/s/m/l/x")
    split_config: Optional[Dict[str, float]] = Field(
        default=None, description="Dataset split ratios: train/val/test (sum=1.0)"
    )
    training: Optional[Dict[str, Any]] = Field(default=None, description="Training hyperparameter overrides")
    priority: int = Field(default=0, description="Job priority (higher = more urgent)")
    output_dir: Optional[str] = Field(
        default=None, description="Output directory (auto-generated if not provided)"
    )
    tags: List[str] = Field(default_factory=list, description="Optional tags for filtering")

    @model_validator(mode="after")
    def _check_paired_fields(self) -> "DistillationRequest":
        """teacher_dir and unlabeled_image_paths are paired — both or neither."""
        has_teacher = self.teacher_dir is not None
        has_unlabeled = self.unlabeled_image_paths is not None
        if has_teacher != has_unlabeled:
            raise ValueError(
                "teacher_dir and unlabeled_image_paths must be set together: provide both or neither."
            )
        return self

    @model_validator(mode="after")
    def _check_split_config(self) -> "DistillationRequest":
        """If split_config is provided, ratios must be non-negative and
        sum to ~1.0. math.isclose tolerates fp drift (0.7 + 0.15 + 0.15
        is exactly 1.0 most of the time, but not for all combinations)."""
        if self.split_config is None:
            return self
        for split_name, ratio in self.split_config.items():
            if ratio < 0:
                raise ValueError(f"split_config['{split_name}'] = {ratio} must be >= 0")
        total = sum(self.split_config.values())
        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"split_config ratios must sum to 1.0, got {total} (values: {self.split_config})"
            )
        return self


class COCOImageSchema(BaseModel):
    """COCO image entry."""

    id: int = Field(..., description="Image ID")
    file_name: str = Field(..., description="Image filename")
    width: int = Field(..., gt=0, description="Image width (positive)")
    height: int = Field(..., gt=0, description="Image height (positive)")


class COCOAnnotationSchema(BaseModel):
    """COCO annotation entry."""

    id: int = Field(..., description="Annotation ID")
    image_id: int = Field(..., description="Image ID")
    category_id: int = Field(..., description="Category ID")
    bbox: Optional[List[float]] = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="Bounding box [x, y, w, h] — exactly 4 elements per COCO spec",
    )
    segmentation: Optional[List[List[float]]] = Field(default=None, description="Polygon segmentation")
    area: Optional[float] = Field(default=None, description="Area in pixels")
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Detection confidence (0.0-1.0)")
    iscrowd: Literal[0, 1] = Field(default=0, description="Is crowd annotation (binary 0/1)")


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
    annotation_count: int = Field(..., ge=0, description="Number of annotations in this image (non-negative)")


class VisualizationListResponse(BaseModel):
    """List of visualization images for an auto-label job."""

    job_id: str = Field(..., description="Job ID")
    total: int = Field(..., description="Total number of visualizations")
    images: List[VisualizationInfo] = Field(..., description="List of visualization info")
