"""
REST endpoints for inference with fine-tuned models.

Provides:
- POST /api/inference - Submit inference job
- GET /api/inference/{job_id}/metrics - Get inference metrics
- GET /api/inference/{job_id}/visualizations - List visualization images
- GET /api/inference/{job_id}/visualizations/{filename} - Get visualization image
"""

import os
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse

from api.schemas import (
    InferenceRequest,
    InferenceMetricsResponse,
    InferenceMetricsSummary,
    InferenceClassMetrics,
    InferenceProcessingMetrics,
    VisualizationListResponse,
    VisualizationInfo,
    JobResponse,
    JobProgressSchema,
)
from ml_engine.jobs import JobManager, get_job_manager, Job
from core.config import load_json

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/inference", tags=["inference"])


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
    )


@router.post("", response_model=JobResponse, status_code=201)
async def submit_inference(
    request: InferenceRequest,
    manager: JobManager = Depends(get_manager)
):
    """
    Submit an inference job using a fine-tuned model.
    
    The job uses LoRA adapters from a completed training job to run
    inference on new images.
    
    Example:
        POST /api/inference
        {
            "training_job_id": "abc123-def456-...",
            "image_paths": [
                "upload/2025/01/04/image1.jpeg",
                "upload/2025/01/04/image2.jpeg"
            ],
            "output_mode": "both",
            "box_threshold": 0.5
        }
    """
    try:
        # Validate training job exists and is completed
        training_job = manager.get_job(request.training_job_id)
        if training_job is None:
            raise ValueError(f"Training job not found: {request.training_job_id}")

        if training_job.status.value != "completed":
            raise ValueError(
                f"Training job is not completed (status: {training_job.status.value})"
            )

        # Build config for inference job
        config = {
            "training_job_id": request.training_job_id,
            "image_paths": request.image_paths,
            "output_mode": request.output_mode,
            "box_threshold": request.box_threshold,
            "nms_threshold": request.nms_threshold,
        }

        job = manager.submit_job(
            job_type="inference",
            config=config,
            priority=request.priority,
            output_dir=request.output_dir,
            tags=request.tags,
        )

        logger.info("Submitted inference job %s using model from %s",
                   job.id[:8], request.training_job_id[:8])
        return job_to_response(job)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("Failed to submit inference job: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}") from e


@router.get("/{job_id}/metrics", response_model=InferenceMetricsResponse)
async def get_metrics(
    job_id: str,
    manager: JobManager = Depends(get_manager)
):
    """
    Get inference metrics for a completed inference job.
    
    Example:
        GET /api/inference/abc123/metrics
        
    Returns:
        Metrics summary with per-class breakdowns and processing times
    """
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status.value != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed (status: {job.status.value}). Cannot get metrics."
        )

    # Load metrics from output directory
    output_dir = Path(job.output_dir)
    metrics_path = output_dir / "metrics.json"

    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Metrics not found. The job may not have completed successfully."
        )

    try:
        metrics_data = load_json(str(metrics_path))

        # Build response
        summary = InferenceMetricsSummary(
            total_images=metrics_data["summary"]["total_images"],
            total_detections=metrics_data["summary"]["total_detections"],
            avg_detections_per_image=metrics_data["summary"]["avg_detections_per_image"],
            avg_confidence=metrics_data["summary"]["avg_confidence"]
        )

        per_class = {}
        for class_name, class_data in metrics_data.get("per_class", {}).items():
            per_class[class_name] = InferenceClassMetrics(
                count=class_data["count"],
                avg_confidence=class_data["avg_confidence"]
            )

        processing = InferenceProcessingMetrics(
            total_time_seconds=metrics_data["processing"]["total_time_seconds"],
            avg_time_per_image=metrics_data["processing"]["avg_time_per_image"]
        )

        return InferenceMetricsResponse(
            summary=summary,
            per_class=per_class,
            processing=processing,
            config=metrics_data.get("config", {})
        )

    except Exception as e:
        logger.error("Failed to load metrics: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to load metrics: {str(e)}") from e


@router.get("/{job_id}/visualizations", response_model=VisualizationListResponse)
async def list_visualizations(
    job_id: str,
    manager: JobManager = Depends(get_manager)
):
    """
    List visualization images for an inference job.
    
    Example:
        GET /api/inference/abc123/visualizations
        
    Returns:
        List of visualization image info
    """
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status.value != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed (status: {job.status.value}). Visualizations not available."
        )

    output_dir = Path(job.output_dir)
    viz_dir = output_dir / "visualizations"

    if not viz_dir.exists():
        return VisualizationListResponse(
            job_id=job_id,
            total=0,
            images=[]
        )

    # Load metrics to get detection counts per image
    metrics_path = output_dir / "metrics.json"
    detection_counts = {}

    if metrics_path.exists():
        try:
            metrics_data = load_json(str(metrics_path))
            # For inference, we track total detections, not per-image
            # Just use 0 as placeholder
        except Exception as e:
            logger.warning("Failed to load metrics for counts: %s", e)

    # List visualization files
    viz_files = sorted(viz_dir.glob("*_viz.jpg"))

    images = []
    for viz_path in viz_files:
        # Extract original filename from viz filename
        original_stem = viz_path.stem.replace("_viz", "")

        # Try common extensions
        original_name = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"]:
            candidate = f"{original_stem}{ext}"
            original_name = candidate
            break

        if original_name is None:
            original_name = f"{original_stem}.jpg"

        images.append(VisualizationInfo(
            filename=viz_path.name,
            original=original_name,
            annotation_count=detection_counts.get(original_name, 0)
        ))

    return VisualizationListResponse(
        job_id=job_id,
        total=len(images),
        images=images
    )


@router.get("/{job_id}/visualizations/{filename}")
async def get_visualization(
    job_id: str,
    filename: str,
    manager: JobManager = Depends(get_manager)
):
    """
    Get a visualization image file.
    
    Example:
        GET /api/inference/abc123/visualizations/image1_viz.jpg
        
    Returns:
        The visualization image file
    """
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    output_dir = Path(job.output_dir)
    viz_path = output_dir / "visualizations" / filename

    if not viz_path.exists():
        raise HTTPException(status_code=404, detail=f"Visualization not found: {filename}")

    # Security check: ensure filename doesn't escape directory
    try:
        viz_path.resolve().relative_to(output_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename")

    return FileResponse(
        path=str(viz_path),
        media_type="image/jpeg",
        filename=filename
    )
