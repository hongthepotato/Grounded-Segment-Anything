"""
REST endpoints for auto-labeling.

Provides:
- POST /api/autolabel - Submit auto-labeling job
- GET /api/autolabel/{job_id}/results - Get COCO annotations
- GET /api/autolabel/{job_id}/visualizations - List visualization images
- GET /api/autolabel/{job_id}/visualizations/{filename} - Get visualization image
- PUT /api/autolabel/{job_id}/annotations - Save edited annotations
"""

import os
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse

from api.schemas import (
    AutoLabelRequest,
    AutoLabelResultResponse,
    VisualizationListResponse,
    VisualizationInfo,
    JobResponse,
    JobProgressSchema,
)
from ml_engine.jobs import JobManager, get_job_manager, Job
from core.config import load_json, save_json

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/autolabel", tags=["autolabel"])


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
        priority=job.priority,
        tags=job.tags,
    )


@router.post("", response_model=JobResponse, status_code=201)
async def submit_autolabel(
    request: AutoLabelRequest,
    manager: JobManager = Depends(get_manager)
):
    """
    Submit an auto-labeling job.
    
    The job is queued and executed by a worker with GPU access.
    Returns immediately with job details - poll /api/jobs/{id} for status.
    
    Example:
        POST /api/autolabel
        {
            "image_dir": "/data/raw/images",
            "classes": ["ear of bag", "defect"],
            "output_mode": "boxes",
            "box_threshold": 0.5
        }
    """
    try:
        # Build config for auto-label job
        config = {
            "image_dir": request.image_dir,
            "classes": request.classes,
            "output_mode": request.output_mode,
            "box_threshold": request.box_threshold,
            "text_threshold": request.text_threshold,
            "nms_threshold": request.nms_threshold,
        }

        # Generate output dir if not provided
        output_dir = request.output_dir
        if not output_dir:
            # Will be set by Job.__post_init__ but we can override for auto_label
            output_dir = None  # Let the worker handle it

        job = manager.submit_job(
            job_type="auto_label",
            config=config,
            priority=request.priority,
            output_dir=output_dir,
            tags=request.tags,
        )
        
        logger.info("Submitted auto-label job %s for %s", job.id[:8], request.image_dir)
        return job_to_response(job)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("Failed to submit auto-label job: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}") from e


@router.get("/{job_id}/results", response_model=AutoLabelResultResponse)
async def get_results(
    job_id: str,
    manager: JobManager = Depends(get_manager)
):
    """
    Get COCO-format annotations for a completed auto-label job.
    
    Example:
        GET /api/autolabel/abc123/results
        
    Returns:
        COCO JSON with images, annotations, and categories
    """
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status.value != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed (status: {job.status.value}). Cannot get results."
        )

    # Load annotations from output directory
    output_dir = Path(job.output_dir)
    annotations_path = output_dir / "annotations.json"

    if not annotations_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Annotations not found. The job may not have completed successfully."
        )

    try:
        coco_data = load_json(str(annotations_path))
        return AutoLabelResultResponse(
            images=coco_data.get("images", []),
            annotations=coco_data.get("annotations", []),
            categories=coco_data.get("categories", [])
        )
    except Exception as e:
        logger.error("Failed to load annotations: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to load annotations: {str(e)}") from e


@router.get("/{job_id}/visualizations", response_model=VisualizationListResponse)
async def list_visualizations(
    job_id: str,
    manager: JobManager = Depends(get_manager)
):
    """
    List visualization images for an auto-label job.
    
    Example:
        GET /api/autolabel/abc123/visualizations
        
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

    # Load annotations to get annotation counts per image
    annotations_path = output_dir / "annotations.json"
    annotation_counts = {}
    
    if annotations_path.exists():
        try:
            coco_data = load_json(str(annotations_path))
            # Build map of filename -> annotation count
            image_id_to_filename = {
                img["id"]: img["file_name"]
                for img in coco_data.get("images", [])
            }
            for ann in coco_data.get("annotations", []):
                img_id = ann.get("image_id")
                if img_id in image_id_to_filename:
                    filename = image_id_to_filename[img_id]
                    annotation_counts[filename] = annotation_counts.get(filename, 0) + 1
        except Exception as e:
            logger.warning("Failed to load annotations for counts: %s", e)

    # List visualization files
    viz_files = sorted(viz_dir.glob("*_viz.jpg"))
    
    images = []
    for viz_path in viz_files:
        # Extract original filename from viz filename
        # e.g., "img001_viz.jpg" -> "img001.jpg"
        original_stem = viz_path.stem.replace("_viz", "")
        
        # Try common extensions
        original_name = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"]:
            candidate = f"{original_stem}{ext}"
            if candidate in annotation_counts or original_name is None:
                original_name = candidate
                break
        
        if original_name is None:
            original_name = f"{original_stem}.jpg"

        images.append(VisualizationInfo(
            filename=viz_path.name,
            original=original_name,
            annotation_count=annotation_counts.get(original_name, 0)
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
        GET /api/autolabel/abc123/visualizations/img001_viz.jpg
        
    Returns:
        JPEG image file
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
    viz_path = output_dir / "visualizations" / filename

    if not viz_path.exists():
        raise HTTPException(status_code=404, detail=f"Visualization {filename} not found")

    # Security: ensure path is within expected directory
    try:
        viz_path.resolve().relative_to((output_dir / "visualizations").resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename")

    return FileResponse(
        path=str(viz_path),
        media_type="image/jpeg",
        filename=filename
    )


@router.put("/{job_id}/annotations")
async def save_annotations(
    job_id: str,
    annotations: dict,
    manager: JobManager = Depends(get_manager)
):
    """
    Save edited annotations for an auto-label job.
    
    Accepts full COCO JSON and saves as annotations_edited.json.
    Original annotations.json is preserved.
    
    Example:
        PUT /api/autolabel/abc123/annotations
        {
            "images": [...],
            "annotations": [...],
            "categories": [...]
        }
        
    Returns:
        Save confirmation with path and counts
    """
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status.value != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed (status: {job.status.value}). Cannot save annotations."
        )

    output_dir = Path(job.output_dir)
    
    if not output_dir.exists():
        raise HTTPException(
            status_code=404,
            detail="Output directory not found"
        )

    # Validate structure
    if "images" not in annotations or "annotations" not in annotations or "categories" not in annotations:
        raise HTTPException(
            status_code=400,
            detail="Invalid COCO format. Must contain 'images', 'annotations', and 'categories' keys."
        )

    # Save edited annotations
    edited_path = output_dir / "annotations_edited.json"
    
    try:
        save_json(annotations, str(edited_path))
        logger.info("Saved edited annotations to %s", edited_path)
        
        return {
            "saved": True,
            "path": str(edited_path),
            "image_count": len(annotations.get("images", [])),
            "annotation_count": len(annotations.get("annotations", []))
        }
    except Exception as e:
        logger.error("Failed to save annotations: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to save annotations: {str(e)}") from e


