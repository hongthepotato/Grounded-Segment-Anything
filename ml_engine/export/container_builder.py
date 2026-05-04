"""
ROS2 inference container builder.

Assembles a Docker build context containing:
  - serve/Dockerfile.ros2 (multi-arch)
  - serve/ros2_ws/ (ROS2 Python package)
  - serve/entrypoint.sh
  - model/best.pt (student weights baked in)

Then runs `docker buildx build --platform linux/amd64,linux/arm64 --push`.

Prerequisite (one-time on host):
    docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
    docker buildx create --use --name multiarch --driver docker-container \
        --driver-opt network=host
    docker buildx inspect --bootstrap
"""

import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Relative to repo root — resolved at call time
_SERVE_DIR = Path(__file__).parent.parent.parent / "serve"


def build_ros2_container(
    model_weights: Path,
    job_id: str,
    registry_url: Optional[str] = None,
    platforms: str = "linux/amd64,linux/arm64",
    # Any: callers pass either threading.Event (web background thread) or
    # multiprocessing.synchronize.Event (training subprocess). Both duck-type
    # via .is_set(); a Protocol is overkill for a single-method usage.
    cancel_event: Optional[Any] = None,
    report_fn=None,
) -> str:
    """
    Build and push a multi-arch ROS2 inference container.

    Args:
        model_weights: Path to best.pt
        job_id: Job ID (used in image name and versioned tag)
        registry_url: Registry push URL e.g. "localhost:5000".
                      Falls back to REGISTRY_PUSH_URL env var, then "localhost:5000".
        platforms: Comma-separated buildx platforms.
        cancel_event: If set, build is cancelled when the event fires.
        report_fn: Optional callable(message: str) for progress reporting.

    Returns:
        Versioned image tag (e.g. "localhost:5000/yolo-inference-abc12345:20260406")

    Raises:
        RuntimeError: If docker buildx fails.
    """
    if registry_url is None:
        registry_url = os.environ.get("REGISTRY_PUSH_URL", "host-gateway:5000")

    date_str = datetime.now().strftime("%Y%m%d")
    short_id = job_id[:8]
    versioned_tag = f"{registry_url}/yolo-inference-{short_id}:{date_str}"
    latest_tag = f"{registry_url}/yolo-inference-{short_id}:latest"
    global_latest_tag = f"{registry_url}/yolo-inference:latest"

    def _report(msg: str):
        logger.info(msg)
        if report_fn:
            report_fn(msg)

    _report("Building ROS2 container (this takes 5-90 min on first ARM64 build via QEMU)...")

    tmpdir = tempfile.mkdtemp(prefix=f"ros2_build_{short_id}_")
    try:
        _assemble_context(tmpdir, model_weights)
        _run_buildx(
            build_dir=tmpdir,
            tags=[versioned_tag, latest_tag, global_latest_tag],
            platforms=platforms,
            cancel_event=cancel_event,
            report_fn=_report,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    _report(f"ROS2 container pushed: {versioned_tag}")

    # Write tag to output_dir so worker._complete_job can persist it
    output_dir = model_weights.parent.parent  # .../experiments/student_distillation_XXXXXXXX
    tag_file = output_dir / "image_tag.txt"
    tag_file.write_text(versioned_tag)
    logger.info("Image tag written to %s", tag_file)

    # Fire-and-forget webhook if configured
    _notify_webhook(job_id, versioned_tag)

    return versioned_tag


def _assemble_context(tmpdir: str, model_weights: Path):
    """Copy Dockerfile, ros2_ws, entrypoint, and model weights into build context."""
    ctx = Path(tmpdir)

    # Dockerfile.ros2
    shutil.copy2(_SERVE_DIR / "Dockerfile.ros2", ctx / "Dockerfile")

    # ROS2 workspace
    shutil.copytree(_SERVE_DIR / "ros2_ws", ctx / "ros2_ws")

    # Entrypoint
    shutil.copy2(_SERVE_DIR / "entrypoint.sh", ctx / "entrypoint.sh")

    # Model weights
    model_dir = ctx / "model"
    model_dir.mkdir()
    shutil.copy2(model_weights, model_dir / "best.pt")

    logger.debug("Build context assembled at %s", tmpdir)


def _run_buildx(
    build_dir: str,
    tags: list,
    platforms: str,
    cancel_event: Optional[Any],  # see build_ros2_container; threading or mp Event
    report_fn,
):
    """Run docker buildx build with interruptible cancel support."""
    tag_args = []
    for t in tags:
        tag_args += ["-t", t]

    cmd = [
        "docker",
        "buildx",
        "build",
        "--platform",
        platforms,
        *tag_args,
        "--push",
        build_dir,
    ]

    logger.info("Running: %s", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None  # PIPE above guarantees this; assert tells mypy

    # Background thread: poll cancel_event every 30s and kill buildx if fired
    def _cancel_watcher():
        while proc.poll() is None:
            if cancel_event and cancel_event.is_set():
                logger.warning("Cancel event set — terminating buildx")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return
            time.sleep(30)

    watcher = threading.Thread(target=_cancel_watcher, daemon=True)
    watcher.start()

    output_lines = []
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            output_lines.append(line)
            logger.debug("[buildx] %s", line)

    proc.wait()
    watcher.join(timeout=1)

    if cancel_event and cancel_event.is_set():
        raise RuntimeError("Container build cancelled by job cancellation.")

    if proc.returncode != 0:
        tail = "\n".join(output_lines[-20:])
        raise RuntimeError(f"docker buildx failed (exit {proc.returncode}):\n{tail}")


def _notify_webhook(job_id: str, image_tag: str):
    """POST deployment webhook if DEPLOY_WEBHOOK_URL is set. Fire-and-forget."""
    webhook_url = os.environ.get("DEPLOY_WEBHOOK_URL", "")
    if not webhook_url:
        return

    registry_external = os.environ.get("REGISTRY_EXTERNAL_URL", "workstation:5000")
    short_id = job_id[:8]
    external_tag = image_tag.replace(
        os.environ.get("REGISTRY_PUSH_URL", "localhost:5000"),
        registry_external,
    )

    payload = {
        "event": "model_deployed",
        "job_id": short_id,
        "image_tag": external_tag,
        "pull_command": f"docker pull {external_tag}",
        "run_command": (
            f"docker run --gpus all --network host "
            f"-v yolo-cache:/model/cache {external_tag} "
            f"--ros-args -p input_topic:=/camera/image_raw"
        ),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    def _post():
        try:
            import json
            import urllib.request

            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
            logger.info("Webhook delivered to %s", webhook_url)
        except Exception as exc:
            logger.warning("Webhook delivery failed (non-fatal): %s", exc)

    threading.Thread(target=_post, daemon=True).start()
