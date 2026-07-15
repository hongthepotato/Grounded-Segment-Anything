"""
REST endpoints for model export and download.

Provides:
- GET  /api/jobs/{job_id}/exports     - List available exports (per model)
- GET  /api/jobs/{job_id}/export      - Download trained model package
- POST /api/jobs/{job_id}/build-ros2  - Trigger ROS2 container build
- GET  /api/jobs/{job_id}/deploy-info - Pull/run commands for edge device
"""

import logging
import os
import tempfile
import threading
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

from api.schemas import success_response
from ml_engine.jobs import AsyncJobManager, get_async_job_manager, get_job_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["exports"])


def get_manager() -> AsyncJobManager:
    """Dependency to get AsyncJobManager instance."""
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    return get_async_job_manager(redis_url)


def _find_model_packages(output_dir: Path) -> Dict[str, Path]:
    """Scan exports/ for per-model ZIP packages and student model."""
    exports_dir = output_dir / "exports"
    packages = {}
    if exports_dir.is_dir():
        for zp in exports_dir.glob("*_package.zip"):
            model = zp.stem.replace("_package", "")
            packages[model] = zp

    student_pt = output_dir / "student_model" / "best.pt"
    if student_pt.exists():
        packages["student_model"] = student_pt

    return packages


def _find_lora_adapters(output_dir: Path) -> Dict[str, Path]:
    """Scan for per-model lora_adapters/ directories."""
    adapters: Dict[str, Path] = {}
    if not output_dir.is_dir():
        return adapters
    for child in output_dir.iterdir():
        if child.is_dir():
            lora_dir = child / "lora_adapters"
            if lora_dir.is_dir() and any(lora_dir.iterdir()):
                adapters[child.name] = lora_dir
    return adapters


async def _validate_completed_job(job_id: str, manager: AsyncJobManager):
    """Return job after validating it exists and is completed."""
    job = await manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.status.value != "completed":
        raise HTTPException(status_code=400, detail=f"Job is not completed (status: {job.status.value})")
    return job


@router.get("/{job_id}/exports")
async def list_exports(job_id: str, manager: AsyncJobManager = Depends(get_manager)):
    """
    List available export formats for a completed job.

    Returns per-model availability of merged packages and LoRA adapters.

    Example response:
        {
            "code": 200,
            "data": {
                "models": {
                    "grounding_dino": {
                        "merged_pth": true,
                        "lora_adapters": true,
                        "package_size_mb": 1850.5
                    },
                    "sam": {
                        "merged_pth": true,
                        "lora_adapters": false,
                        "package_size_mb": 375.2
                    }
                }
            }
        }
    """
    job = await _validate_completed_job(job_id, manager)
    output_dir = Path(job.output_dir)

    packages = _find_model_packages(output_dir)
    adapters = _find_lora_adapters(output_dir)

    all_models = sorted(set(packages.keys()) | set(adapters.keys()))

    models_info: Dict[str, Any] = {}
    for model in all_models:
        info: Dict[str, Any] = {
            "merged_pth": model in packages,
            "lora_adapters": model in adapters,
        }
        if model in packages:
            info["package_size_mb"] = round(packages[model].stat().st_size / (1024 * 1024), 1)
        models_info[model] = info

    return JSONResponse(status_code=200, content=success_response(data={"models": models_info}))


@router.get("/{job_id}/export")
async def download_model(
    job_id: str,
    format: str = Query(default="merged_pth", description="Export format: merged_pth, lora_adapters"),
    model: Optional[str] = Query(
        default=None, description="Model name (grounding_dino, sam). Auto-detected if omitted."
    ),
    manager: AsyncJobManager = Depends(get_manager),
):
    """
    Download trained model package.

    Query params:
        format: merged_pth (default) or lora_adapters
        model:  grounding_dino or sam (auto-detected if only one model was trained)

    Returns:
        ZIP file containing model weights
    """
    job = await _validate_completed_job(job_id, manager)
    output_dir = Path(job.output_dir)

    if format == "merged_pth":
        packages = _find_model_packages(output_dir)
        if not packages:
            raise HTTPException(status_code=404, detail="No export packages found.")

        model_name = _resolve_model(model, list(packages.keys()), "merged package")
        zip_path = packages[model_name]

        return FileResponse(
            path=str(zip_path),
            filename=f"{model_name}_model_{job_id[:8]}.zip",
            media_type="application/zip",
        )

    if format == "student_model":
        student_pt = output_dir / "student_model" / "best.pt"
        if not student_pt.exists():
            raise HTTPException(
                status_code=404,
                detail="Student model not found. Was this a student_distillation job?",
            )

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(student_pt, "best.pt")
            class_names_file = output_dir / "yolo_dataset" / "data.yaml"
            if class_names_file.exists():
                zipf.write(class_names_file, "data.yaml")

        return FileResponse(
            path=str(tmp_path),
            filename=f"student_model_{job_id[:8]}.zip",
            media_type="application/zip",
            background=BackgroundTask(tmp_path.unlink),
        )

    if format == "lora_adapters":
        adapters = _find_lora_adapters(output_dir)
        if not adapters:
            raise HTTPException(status_code=404, detail="No LoRA adapters found.")

        model_name = _resolve_model(model, list(adapters.keys()), "LoRA adapters")
        lora_dir = adapters[model_name]

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in lora_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(lora_dir)
                    zipf.write(file_path, arcname)

        return FileResponse(
            path=str(tmp_path),
            filename=f"{model_name}_lora_{job_id[:8]}.zip",
            media_type="application/zip",
            background=BackgroundTask(tmp_path.unlink),
        )

    raise HTTPException(
        status_code=400,
        detail=f"Unknown format: {format}. Available: merged_pth, lora_adapters, student_model",
    )


def _resolve_model(requested: Optional[str], available: List[str], label: str) -> str:
    """Pick the model name, auto-selecting if only one is available."""
    if requested:
        if requested not in available:
            raise HTTPException(
                status_code=404,
                detail=f"No {label} for model '{requested}'. Available: {available}",
            )
        return requested

    if len(available) == 1:
        return available[0]

    raise HTTPException(
        status_code=400, detail=f"Multiple models have {label}: {available}. Specify ?model=<name>"
    )


@router.post("/{job_id}/build-ros2")
async def build_ros2(
    job_id: str,
    manager: AsyncJobManager = Depends(get_manager),
):
    """
    Trigger a ROS2 inference container build for a completed distillation job.

    Runs in the background — returns immediately. Poll /deploy-info to check
    when ros2_image_tag appears (set after push completes).

    Returns:
        {"status": "building", "message": "..."}
    """
    job = await _validate_completed_job(job_id, manager)
    output_dir = Path(job.output_dir)
    student_pt = output_dir / "student_model" / "best.pt"

    if not student_pt.exists():
        raise HTTPException(
            status_code=404, detail="student_model/best.pt not found. Was this a student_distillation job?"
        )

    registry_url = os.environ.get("REGISTRY_PUSH_URL", "localhost:5000")
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")

    def _build_in_background():
        try:
            from ml_engine.export.container_builder import build_ros2_container

            image_tag = build_ros2_container(
                model_weights=student_pt,
                job_id=job_id,
                registry_url=registry_url,
            )
            # Persist via a sync JobManager — mirrors worker.py's pattern and
            # avoids coupling the thread's lifetime to the FastAPI event loop.
            sync_manager = get_job_manager(redis_url)
            sync_manager.store.update_job(job_id, ros2_image_tag=image_tag)
            logger.info("Background ROS2 build complete: %s", image_tag)
        except Exception as exc:
            logger.error("Background ROS2 build failed for job %s: %s", job_id[:8], exc)

    thread = threading.Thread(target=_build_in_background, daemon=True)
    thread.start()

    return JSONResponse(
        status_code=202,
        content=success_response(
            data={
                "status": "building",
                "message": (
                    "ROS2 container build started in background. "
                    "This takes 5-90 min on first ARM64 build. "
                    "Poll GET /api/jobs/{job_id}/deploy-info for the image_tag."
                ),
            }
        ),
    )


@router.get("/{job_id}/deploy-info")
async def deploy_info(
    job_id: str,
    manager: AsyncJobManager = Depends(get_manager),
):
    """
    Return pull/run commands for deploying the ROS2 inference container on an edge device.

    Returns 404 if the container has not been built yet (trigger via POST /build-ros2
    or by setting build_ros2_container=true in the distillation job config).

    Example response:
        {
          "image_tag": "workstation:5000/yolo-inference-abc12345:20260406",
          "pull_command": "docker pull workstation:5000/yolo-inference-abc12345:20260406",
          "run_command": "docker run --gpus all ...",
          "setup_script_url": "http://workstation:8080/api/setup-edge-device.sh",
          "topics": {...},
          "notes": "..."
        }
    """
    job = await _validate_completed_job(job_id, manager)

    # image_tag persisted by worker._complete_job reading image_tag.txt,
    # or by the background /build-ros2 endpoint.
    ros2_image_tag = getattr(job, "ros2_image_tag", None)

    # Fallback: check the filesystem directly (handles re-runs / legacy)
    if not ros2_image_tag and job.output_dir:
        tag_file = Path(job.output_dir) / "image_tag.txt"
        if tag_file.exists():
            ros2_image_tag = tag_file.read_text().strip() or None

    if not ros2_image_tag:
        raise HTTPException(
            status_code=404,
            detail=(
                "ROS2 container not yet built for this job. "
                "Trigger via POST /api/jobs/{job_id}/build-ros2 "
                "or rerun with build_ros2_container=true."
            ),
        )

    registry_external = os.environ.get("REGISTRY_EXTERNAL_URL", "workstation:5000")
    registry_push = os.environ.get("REGISTRY_PUSH_URL", "localhost:5000")
    # Swap internal push URL for the edge-accessible external URL
    external_tag = ros2_image_tag.replace(registry_push, registry_external)

    workstation_host = registry_external.split(":")[0]
    api_port = os.environ.get("API_PORT", "8080")

    run_command = (
        f"docker run --gpus all --network host "
        f"-v yolo-cache:/model/cache "
        f"{external_tag} "
        f"--ros-args -p input_topic:=/camera/image_raw -p confidence:=0.5"
    )

    return JSONResponse(
        status_code=200,
        content=success_response(
            data={
                "image_tag": external_tag,
                "pull_command": f"docker pull {external_tag}",
                "run_command": run_command,
                "setup_script_url": (f"http://{workstation_host}:{api_port}/api/setup-edge-device.sh"),
                "topics": {
                    "subscribe": "/camera/image_raw (sensor_msgs/msg/Image)",
                    "publish": "/detections (vision_msgs/msg/Detection2DArray)",
                    "diagnostics": "/yolo_inference/diagnostics (std_msgs/msg/String, JSON)",
                },
                "notes": (
                    "First boot converts model to TensorRT (~2 min). "
                    "Engine cached in yolo-cache volume for instant subsequent starts. "
                    "WARNING: -v yolo-cache:/model/cache is required. "
                    "Without it, TensorRT reconverts on every restart (~2 min cold start)."
                ),
            }
        ),
    )


@router.get("/setup-edge-device.sh", include_in_schema=False)
async def serve_setup_script():
    """Serve the edge device setup script for download."""
    script_path = Path(__file__).parent.parent.parent / "serve" / "setup_edge_device.sh"
    if not script_path.exists():
        raise HTTPException(status_code=404, detail="setup_edge_device.sh not found")
    return FileResponse(
        path=str(script_path),
        filename="setup_edge_device.sh",
        media_type="text/x-sh",
    )
