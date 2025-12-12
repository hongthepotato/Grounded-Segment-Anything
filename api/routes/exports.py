"""
REST endpoints for model export and download.

Provides:
- GET /api/jobs/{job_id}/export - Download trained model package
- GET /api/jobs/{job_id}/exports - List available exports
"""

import os
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel, Field
import tempfile
import zipfile

from ml_engine.jobs import JobManager, get_job_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["exports"])


def get_manager() -> JobManager:
    """Dependency to get JobManager instance."""
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    return get_job_manager(redis_url)


class ExportInfo(BaseModel):
    """Information about available exports."""
    available: list = Field(default_factory=list, description="Already exported formats")
    exportable: list = Field(default_factory=list, description="Formats that can be exported")
    package_size_mb: Optional[float] = Field(default=None, description="Package size in MB")


@router.get("/{job_id}/exports", response_model=ExportInfo)
async def list_exports(
    job_id: str,
    manager: JobManager = Depends(get_manager)
):
    """
    List available export formats for a completed job.
    
    Example:
        GET /api/jobs/abc123/exports
        
    Returns:
        {
            "available": ["merged_pth"],
            "exportable": [],
            "package_size_mb": 1850.5
        }
    """
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status.value != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed (status: {job.status.value}). Cannot export."
        )

    output_dir = Path(job.output_dir)
    exports_dir = output_dir / "exports"

    available = []
    package_size_mb = None

    # Check for merged model package
    zip_path = exports_dir / "model_package.zip"
    if zip_path.exists():
        available.append("merged_pth")
        package_size_mb = zip_path.stat().st_size / (1024 * 1024)

    # Check for LoRA adapters
    lora_dir = output_dir / "teachers" / "grounding_dino_lora_adapters"
    if lora_dir.exists():
        available.append("lora_adapters")

    return ExportInfo(
        available=available,
        exportable=[],  # Future: ONNX, TorchScript
        package_size_mb=round(package_size_mb, 1) if package_size_mb else None
    )


@router.get("/{job_id}/export")
async def download_model(
    job_id: str,
    format: str = Query(
        default="merged_pth",
        description="Export format: merged_pth (default), lora_adapters"
    ),
    manager: JobManager = Depends(get_manager)
):
    """
    Download trained model package.
    
    Example:
        GET /api/jobs/abc123/export?format=merged_pth
        
    Returns:
        ZIP file containing model weights and inference scripts
        
    Formats:
        - merged_pth: Full merged model (~2GB) + inference scripts
        - lora_adapters: Just LoRA weights (~5MB)
    """
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status.value != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed (status: {job.status.value}). Cannot download."
        )

    output_dir = Path(job.output_dir)

    if format == "merged_pth":
        # Full merged model package
        zip_path = output_dir / "exports" / "model_package.zip"
        if not zip_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Export package not found. The job may not have completed successfully."
            )

        filename = f"grounding_dino_model_{job_id[:8]}.zip"
        return FileResponse(
            path=str(zip_path),
            filename=filename,
            media_type="application/zip"
        )

    if format == "lora_adapters":
        # LoRA adapters only
        lora_dir = output_dir / "teachers" / "grounding_dino_lora_adapters"
        if not lora_dir.exists():
            raise HTTPException(
                status_code=404,
                detail="LoRA adapters not found."
            )

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in lora_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(lora_dir)
                    zipf.write(file_path, arcname)

        filename = f"grounding_dino_lora_{job_id[:8]}.zip"
        return FileResponse(
            path=str(tmp_path),
            filename=filename,
            media_type="application/zip",
            background=BackgroundTask(tmp_path.unlink)
        )

    raise HTTPException(
        status_code=400,
        detail=f"Unknown format: {format}. Available: merged_pth, lora_adapters"
    )
