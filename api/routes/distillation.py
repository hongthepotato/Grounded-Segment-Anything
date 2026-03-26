"""
Convenience REST endpoint for student distillation.

Provides:
- POST /api/distillation - Submit student distillation job
"""

import os
import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from api.schemas import DistillationRequest, success_response
from api.routes.jobs import validate_job_config, job_to_response
from ml_engine.jobs import JobManager, get_job_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/distillation", tags=["distillation"])


def get_manager() -> JobManager:
    """Dependency to get JobManager instance."""
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    return get_job_manager(redis_url)


@router.post("", status_code=200)
async def submit_distillation(
    request: DistillationRequest,
    manager: JobManager = Depends(get_manager)
):
    """
    Submit a student distillation job.

    This is a convenience wrapper around POST /api/jobs with:
    job_type="student_distillation".
    """
    config: Dict[str, Any] = {
        "data_path": request.data_path,
        "image_paths": request.image_paths,
    }
    if request.teacher_dir is not None:
        config["teacher_dir"] = request.teacher_dir
    if request.unlabeled_image_paths is not None:
        config["unlabeled_image_paths"] = request.unlabeled_image_paths
    if request.student_model is not None:
        config["student_model"] = request.student_model
    if request.student_size is not None:
        config["student_size"] = request.student_size
    if request.split_config is not None:
        config["split_config"] = request.split_config
    if request.training is not None:
        config["training"] = request.training

    validation_errors = validate_job_config("student_distillation", config)
    if validation_errors:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid distillation config: {'; '.join(validation_errors)}"
        )

    try:
        job = manager.submit_job(
            job_type="student_distillation",
            config=config,
            priority=request.priority,
            output_dir=request.output_dir,
            tags=request.tags,
        )
        logger.info("Submitted distillation job %s", job.id[:8])
        return JSONResponse(
            status_code=200,
            content=success_response(
                data={"jobs": [job_to_response(job).model_dump(mode='json')]},
                code=200
            )
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("Failed to submit distillation job: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}") from e
