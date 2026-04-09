"""
REST endpoints for job management.

Provides:
- POST /api/jobs - Submit labeled YOLOv8n-seg training (always queues yolo_seg_labeled)
- GET /api/jobs - List all jobs
- GET /api/jobs/{id} - Get job details
- DELETE /api/jobs/{id} - Cancel job
- GET /api/jobs/{id}/logs - Get job logs (future)
- GET /api/queue/status - Get queue status
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse

from api.schemas import (
    JobCreate,
    JobResponse,
    JobListResponse,
    JobProgressSchema,
    QueueStatusResponse,
    WorkerResponse,
    success_response,
)
from ml_engine.jobs import JobManager, get_job_manager, Job

logger = logging.getLogger(__name__)


def get_job_accuracy(output_dir: Optional[str]) -> Optional[float]:
    """
    Read accuracy score from evaluation report if available.
    
    Args:
        output_dir: Job output directory
        
    Returns:
        Accuracy score (0-100) or None if not available
    """
    if not output_dir:
        return None

    # Look for grounding_dino evaluation report
    report_path = Path(output_dir) / "evaluation" / "grounding_dino_report.json"

    if not report_path.exists():
        return None

    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)

        # Get overall_score from simple_metrics
        accuracy = report.get("simple_metrics", {}).get("overall_score")
        if accuracy is not None:
            return float(accuracy)
    except (IOError, json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to read evaluation report: %s", e)

    return None

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


# =============================================================================
# Job Config Validation
# =============================================================================

# Required fields per job type
JOB_CONFIG_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "teacher_training": {
        "required": ["data_path", "image_paths"],
        "field_types": {
            "data_path": str,
            "image_paths": list,
        },
        "field_validations": {
            "image_paths": lambda v: len(v) > 0,  # Must have at least one image
        },
    },
    "student_distillation": {
        "required": ["data_path", "image_paths"],
        "field_types": {
            "data_path": str,
            "image_paths": list,
            "teacher_dir": str,
            "unlabeled_image_paths": list,
            "student_model": str,
            "student_size": str,
            "split_config": dict,
            "training": dict,
        },
        "field_validations": {
            "image_paths": lambda v: len(v) > 0,
        }
    },
    "yolo_seg_labeled": {
        "required": ["data_path", "image_paths"],
        "field_types": {
            "data_path": str,
            "image_paths": list,
            "split_config": dict,
            "training": dict,
        },
        "field_validations": {
            "image_paths": lambda v: len(v) > 0,
        },
    },
}

YOLO_SEG_LABELED_METADATA: Dict[str, Any] = {
    "required": ["data_path", "image_paths"],
    "optional": ["split_config", "training"],
    "constraints": {
        "annotations": "COCO must include segmentation (polygon) masks for yolov8n-seg",
        "split_config": "train/val/test must be non-negative and sum to 1.0",
    },
    "defaults": {
        "student_model": "yolov8n-seg",
        "split_config": {"train": 0.8, "val": 0.2},
    },
    "example": {
        "job_type": "teacher_training",
        "config": {
            "data_path": "upload/data/annotations/coco.json",
            "image_paths": ["upload/2025/12/17/boxes/rgb_image_20251229_141826_663.png"],
            "training": {"epochs": 100, "batch_size": 16},
        },
        "note": "POST /api/jobs accepts this envelope; job_type is ignored; job stored as yolo_seg_labeled",
    },
}


def _validate_split_config(split_cfg: Dict[str, Any]) -> Optional[str]:
    """Validate train/val/test split ratios."""
    allowed_keys = {"train", "val", "test"}
    unknown_keys = set(split_cfg.keys()) - allowed_keys
    if unknown_keys:
        return f"split_config contains unsupported keys: {sorted(unknown_keys)}"

    values: Dict[str, float] = {}
    for k, v in split_cfg.items():
        if not isinstance(v, (int, float)):
            return f"split_config['{k}'] must be numeric"
        if float(v) < 0:
            return f"split_config['{k}'] must be >= 0"
        values[k] = float(v)

    total = sum(values.values())
    if total <= 0:
        return "split_config sum must be > 0"
    if abs(total - 1.0) > 1e-6:
        return f"split_config values must sum to 1.0, got {total:.6f}"
    return None


def validate_job_config(job_type: str, config: Dict[str, Any]) -> List[str]:
    """
    Validate job config before submission.
    
    Args:
        job_type: Type of job (teacher_training, student_distillation)
        config: Job configuration dict
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Check if job type is known
    if job_type not in JOB_CONFIG_REQUIREMENTS:
        # Unknown job type - skip validation (let worker handle it)
        return errors

    requirements = JOB_CONFIG_REQUIREMENTS[job_type]

    # Check required fields
    for field in requirements.get("required", []):
        if field not in config or config[field] is None:
            errors.append(f"'{field}' is required for {job_type}")
        elif field in requirements.get("field_types", {}):
            expected_type = requirements["field_types"][field]
            if not isinstance(config[field], expected_type):
                errors.append(f"'{field}' must be a {expected_type.__name__}")

    # Run custom validations
    for field, validator in requirements.get("field_validations", {}).items():
        if field in config and config[field] is not None:
            if not validator(config[field]):
                if field == "image_paths":
                    errors.append(f"'{field}' must contain at least one image path")
                else:
                    errors.append(f"'{field}' validation failed")

    if job_type == "student_distillation":
        teacher_dir = config.get("teacher_dir")
        unlabeled = config.get("unlabeled_image_paths")

        # Require teacher_dir + unlabeled_image_paths as a pair.
        if bool(teacher_dir) != bool(unlabeled):
            errors.append(
                "'teacher_dir' and 'unlabeled_image_paths' must be provided together "
                "(or both omitted)"
            )

        if unlabeled is not None:
            if not isinstance(unlabeled, list):
                errors.append("'unlabeled_image_paths' must be a list")
            elif len(unlabeled) == 0:
                errors.append("'unlabeled_image_paths' must contain at least one image path")

        student_size = config.get("student_size")
        if student_size is not None and student_size not in {"n", "s", "m", "l", "x"}:
            errors.append("'student_size' must be one of: n, s, m, l, x")

        split_cfg = config.get("split_config")
        if split_cfg is not None:
            if not isinstance(split_cfg, dict):
                errors.append("'split_config' must be an object")
            else:
                split_error = _validate_split_config(split_cfg)
                if split_error:
                    errors.append(split_error)

    if job_type == "teacher_training":
        auto = config.get("auto_student_distillation")
        if auto is not None and not isinstance(auto, bool):
            errors.append("'auto_student_distillation' must be a boolean")
        dsplit = config.get("distillation_split_config")
        if dsplit is not None:
            if not isinstance(dsplit, dict):
                errors.append("'distillation_split_config' must be an object")
            else:
                split_error = _validate_split_config(dsplit)
                if split_error:
                    errors.append(f"distillation_split_config: {split_error}")

    if job_type == "yolo_seg_labeled":
        split_cfg = config.get("split_config")
        if split_cfg is not None:
            if not isinstance(split_cfg, dict):
                errors.append("'split_config' must be an object")
            else:
                split_error = _validate_split_config(split_cfg)
                if split_error:
                    errors.append(split_error)

    return errors


def get_manager() -> JobManager:
    """Dependency to get JobManager instance."""
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

    # Get accuracy from evaluation report (only for completed jobs)
    accuracy = None
    if job.status.value == "completed":
        accuracy = get_job_accuracy(job.output_dir)

    return JobResponse(
        id=job.id,
        type=job.type,
        status=job.status.value,
        progress=progress,
        worker_id=job.worker_id,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error_message=job.error_message,
        output_dir=job.output_dir,
        accuracy=accuracy,
        # Commented out - not needed by frontend for now
        # priority=job.priority,
        # tags=job.tags,
    )


@router.post("", status_code=200)
async def submit_job(
    request: JobCreate,
    manager: JobManager = Depends(get_manager),
):
    """
    Submit **yolov8n-seg** training on labeled COCO only (no teachers).

    Body is `JobCreate` (`job_type`, `config`, `priority`, `output_dir`, `tags`).
    The top-level ``job_type`` field is kept for client parity (e.g. ``teacher_training``)
    but is **ignored**; the queue always receives ``yolo_seg_labeled``.

    ``config`` must include ``data_path`` and ``image_paths``. Optional: ``split_config``,
    ``training``. COCO must include segmentation masks.
    """
    config: Dict[str, Any] = dict(request.config)

    validation_errors = validate_job_config("yolo_seg_labeled", config)
    if validation_errors:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid job config: {'; '.join(validation_errors)}",
        )

    try:
        job = manager.submit_job(
            job_type="yolo_seg_labeled",
            config=config,
            priority=request.priority,
            output_dir=request.output_dir,
            tags=request.tags,
        )
        logger.info(
            "Submitted yolo_seg_labeled job %s (client job_type=%r)",
            job.id[:8],
            request.job_type,
        )
        return JSONResponse(
            status_code=200,
            content=success_response(
                data={"jobs": [job_to_response(job).model_dump(mode="json")]},
                code=200,
            ),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("Failed to submit job: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}") from e


@router.get("")
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

    response_data = JobListResponse(
        jobs=[job_to_response(job) for job in jobs],
        total=total,
        limit=limit,
        offset=offset,
    )

    return JSONResponse(
        status_code=200,
        content=success_response(
            data=response_data.model_dump(mode='json')
        )
    )


@router.get("/types")
async def list_job_types():
    """
    Metadata for **POST /api/jobs** (labeled YOLO-seg submission).

    Other job kinds (e.g. ``student_distillation``) use dedicated endpoints; types may
    still appear on listed jobs from earlier runs or internal queues.
    """
    data = {
        "job_types": {
            "yolo_seg_labeled": YOLO_SEG_LABELED_METADATA,
        },
        "other_routes": {
            "student_distillation": "POST /api/distillation",
        },
    }
    return JSONResponse(status_code=200, content=success_response(data=data))


@router.get("/{job_id}")
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

    return JSONResponse(
        status_code=200,
        content=success_response(
            data={"jobs": [job_to_response(job).model_dump(mode='json')]}
        )
    )


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
    return JSONResponse(
        status_code=200,
        content=success_response(
            data={"jobs": [job_to_response(job).model_dump(mode='json')]}
        )
    )


# Queue status endpoint (separate router for clarity)
queue_router = APIRouter(prefix="/api/queue", tags=["queue"])


@queue_router.get("/status")
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

    response_data = QueueStatusResponse(
        queue_length=status["queue_length"],
        workers=workers,
        job_counts=status["job_counts"],
    )

    return JSONResponse(
        status_code=200,
        content=success_response(
            data=response_data.model_dump(mode='json')
        )
    )
