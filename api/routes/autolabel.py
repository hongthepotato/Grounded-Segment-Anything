"""
REST endpoints for auto-labeling.

Provides:
- POST /api/autolabel - Submit auto-labeling job
- GET /api/autolabel/{job_id}/results - Get COCO annotations
- GET /api/autolabel/{job_id}/visualizations - List visualization images
- GET /api/autolabel/{job_id}/visualizations/{filename} - Get visualization image
- PUT /api/autolabel/{job_id}/annotations - Save edited annotations
"""

import logging
import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from api.routes.jobs import job_to_response
from api.schemas import (
    AutoLabelRequest,
    AutoLabelResultResponse,
    VisualizationInfo,
    VisualizationListResponse,
    success_response,
)
from core.config import load_json, save_json
from ml_engine.jobs import AsyncJobManager, get_async_job_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/autolabel", tags=["autolabel"])


def get_manager() -> AsyncJobManager:
    """Dependency to get AsyncJobManager instance."""
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    return get_async_job_manager(redis_url)


# `job_to_response` is imported from api.routes.jobs (the canonical version).
# A local duplicate used to live here but had drifted: it passed `config`,
# `priority`, `tags` (silently dropped by pydantic's extra='ignore') and
# never computed `accuracy` for completed jobs. Importing the canonical one
# keeps autolabel responses in sync with the rest of the API.


@router.post("")
async def submit_autolabel(request: AutoLabelRequest, manager: AsyncJobManager = Depends(get_manager)):
    """
    Submit an auto-labeling job.

    The job is queued and executed by a worker with GPU access.
    Returns immediately with job details - poll /api/jobs/{id} for status.

    Example:
        POST /api/autolabel
        {
            "image_paths": [
                "upload/2025/12/16/xxx1.jpeg",
                "upload/2025/12/16/xxx2.jpeg",
            ],
            "classes": ["ear of bag", "defect"],
            "output_mode": "boxes",
            "box_threshold": 0.5
        }
    """
    try:
        # Build config for auto-label job
        config = {
            "image_paths": request.image_paths,
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

        job = await manager.submit_job(
            job_type="auto_label",
            config=config,
            priority=request.priority,
            output_dir=output_dir,
            tags=request.tags,
        )

        logger.info("Submitted auto-label job %s for %s", job.id[:8], request.image_paths)
        return JSONResponse(
            status_code=200,
            content=success_response(data=job_to_response(job).model_dump(mode="json")),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("Failed to submit auto-label job: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}") from e


@router.get("/{job_id}/results", response_model=AutoLabelResultResponse)
async def get_results(job_id: str, manager: AsyncJobManager = Depends(get_manager)):
    """
    Get COCO-format annotations for a completed auto-label job.

    Example:
        GET /api/autolabel/abc123/results

    Returns:
        COCO JSON with images, annotations, and categories
    """
    job = await manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status.value != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed (status: {job.status.value}). Cannot get results.",
        )
    if job.output_dir is None:
        raise HTTPException(status_code=500, detail=f"Job {job_id} has no output_dir")

    # Load annotations from output directory
    output_dir = Path(job.output_dir)
    annotations_path = output_dir / "annotations.json"

    if not annotations_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Annotations not found. The job may not have completed successfully.",
        )

    try:
        coco_data = load_json(str(annotations_path))
        return AutoLabelResultResponse(
            images=coco_data.get("images", []),
            annotations=coco_data.get("annotations", []),
            categories=coco_data.get("categories", []),
        )
    except Exception as e:
        logger.error("Failed to load annotations: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to load annotations: {str(e)}") from e


@router.get("/{job_id}/visualizations", response_model=VisualizationListResponse)
async def list_visualizations(job_id: str, manager: AsyncJobManager = Depends(get_manager)):
    """
    List visualization images for an auto-label job.

    Example:
        GET /api/autolabel/abc123/visualizations

    Returns:
        List of visualization image info
    """
    job = await manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status.value != "completed":
        raise HTTPException(
            status_code=400,
            detail=(f"Job is not completed (status: {job.status.value}). Visualizations not available."),
        )
    if job.output_dir is None:
        raise HTTPException(status_code=500, detail=f"Job {job_id} has no output_dir")

    output_dir = Path(job.output_dir)
    viz_dir = output_dir / "visualizations"

    if not viz_dir.exists():
        return VisualizationListResponse(job_id=job_id, total=0, images=[])

    # Load annotations to get annotation counts per image
    annotations_path = output_dir / "annotations.json"
    annotation_counts: dict[str, int] = {}

    if annotations_path.exists():
        try:
            coco_data = load_json(str(annotations_path))
            # Build map of filename -> annotation count
            image_id_to_filename = {img["id"]: img["file_name"] for img in coco_data.get("images", [])}
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

        images.append(
            VisualizationInfo(
                filename=viz_path.name,
                original=original_name,
                annotation_count=annotation_counts.get(original_name, 0),
            )
        )

    return VisualizationListResponse(job_id=job_id, total=len(images), images=images)


@router.get("/{job_id}/visualizations/{filename}")
async def get_visualization(job_id: str, filename: str, manager: AsyncJobManager = Depends(get_manager)):
    """
    Get a visualization image file.

    Example:
        GET /api/autolabel/abc123/visualizations/img001_viz.jpg

    Returns:
        JPEG image file
    """
    job = await manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status.value != "completed":
        raise HTTPException(
            status_code=400,
            detail=(f"Job is not completed (status: {job.status.value}). Visualizations not available."),
        )
    if job.output_dir is None:
        raise HTTPException(status_code=500, detail=f"Job {job_id} has no output_dir")

    output_dir = Path(job.output_dir)
    viz_path = output_dir / "visualizations" / filename

    if not viz_path.exists():
        raise HTTPException(status_code=404, detail=f"Visualization {filename} not found")

    # Security: ensure path is within expected directory
    try:
        viz_path.resolve().relative_to((output_dir / "visualizations").resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename")

    return FileResponse(path=str(viz_path), media_type="image/jpeg", filename=filename)


@router.put("/{job_id}/annotations")
async def save_annotations(job_id: str, annotations: dict, manager: AsyncJobManager = Depends(get_manager)):
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
    job = await manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status.value != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed (status: {job.status.value}). Cannot save annotations.",
        )
    if job.output_dir is None:
        raise HTTPException(status_code=500, detail=f"Job {job_id} has no output_dir")

    output_dir = Path(job.output_dir)

    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="Output directory not found")

    # Validate structure
    if "images" not in annotations or "annotations" not in annotations or "categories" not in annotations:
        raise HTTPException(
            status_code=400,
            detail=("Invalid COCO format. Must contain 'images', 'annotations', and 'categories' keys."),
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
            "annotation_count": len(annotations.get("annotations", [])),
        }
    except Exception as e:
        logger.error("Failed to save annotations: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to save annotations: {str(e)}") from e
