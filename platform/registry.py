"""Block registry — discovers blocks from the blocks/ directory.

Scans blocks/{block-id}/block.json at startup and on refresh.
Validates each block.json against the schema.
Maintains detected hardware state for palette badge display.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonschema

logger = logging.getLogger(__name__)

# Port alias compatibility map. Wires can connect ports with the same alias.
# The second entry per alias is the set of accepted ROS2 types (informational — not enforced
# at this layer; block authors are responsible for publishing the correct type).
ALIAS_COMPAT_MAP: Dict[str, List[str]] = {
    "image_frame":        ["sensor_msgs/Image", "sensor_msgs/CompressedImage"],
    "detections":         ["vision_msgs/Detection2DArray"],
    "filtered_detections": ["vision_msgs/Detection2DArray"],
    "flag_event":         ["std_msgs/Bool", "std_msgs/String"],
    "annotated_image":    ["sensor_msgs/Image"],
    "depth_frame":        ["sensor_msgs/PointCloud2", "sensor_msgs/Image"],
    "joint_state":        ["sensor_msgs/JointState"],
    "ee_pose":            ["geometry_msgs/PoseStamped"],
    "force_reading":      ["geometry_msgs/WrenchStamped"],
    "motion_id":          ["std_msgs/String"],
    "motion_complete":    ["std_msgs/Bool"],
    "gripper_cmd":        ["std_msgs/String"],
    "grasp_state":        ["std_msgs/Bool"],
}

# annotated_image is NOT wirable to image_frame inputs (visualization only).
# This exclusion is enforced in connection validation.
_ANNOTATION_OUTPUT_ONLY = {"annotated_image"}

_SCHEMA_PATH = Path(__file__).parent.parent / "blocks" / "sdk" / "block.schema.json"
_BLOCKS_DIR = Path(__file__).parent.parent / "blocks"


def _load_schema() -> Optional[Dict]:
    if _SCHEMA_PATH.exists():
        with open(_SCHEMA_PATH) as f:
            return json.load(f)
    logger.warning("block.schema.json not found — skipping validation")
    return None


_schema = _load_schema()


def _validate_block(block: Dict) -> List[str]:
    """Return a list of validation error messages, empty if valid."""
    if _schema is None:
        return []
    errors = []
    validator = jsonschema.Draft7Validator(_schema)
    for err in validator.iter_errors(block):
        errors.append(f"{err.path}: {err.message}" if err.path else err.message)
    return errors


def scan_blocks() -> List[Dict[str, Any]]:
    """Scan blocks/ directory and return list of valid block definitions."""
    blocks = []
    sdk_dirs = {"base", "base-cuda", "sdk"}

    for block_dir in sorted(_BLOCKS_DIR.iterdir()):
        if not block_dir.is_dir():
            continue
        if block_dir.name in sdk_dirs:
            continue

        block_json_path = block_dir / "block.json"
        if not block_json_path.exists():
            continue

        try:
            with open(block_json_path) as f:
                block = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load {block_json_path}: {e}")
            continue

        errors = _validate_block(block)
        if errors:
            logger.warning(f"Block {block_dir.name} has schema errors: {errors}")
            block["_validation_errors"] = errors
            block["_status"] = "error"
        else:
            block["_status"] = "ok"
            block.pop("_validation_errors", None)

        block["_dir"] = str(block_dir)
        blocks.append(block)

    logger.info(f"Block registry: {len(blocks)} blocks found")
    return blocks


def detect_hardware() -> List[str]:
    """Probe for connected hardware. Returns list of hint strings."""
    detected = []

    # USB cameras — check /dev/video*
    try:
        import glob
        if glob.glob("/dev/video*"):
            detected.append("usb_cam")
    except Exception:
        pass

    # Intel RealSense
    try:
        result = subprocess.run(
            ["rs-enumerate-devices"],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode == 0 and "Intel RealSense" in result.stdout:
            detected.append("realsense")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Zed camera
    try:
        result = subprocess.run(
            ["ZEDMiniDetection"],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode == 0:
            detected.append("zed")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    logger.info(f"Detected hardware: {detected}")
    return detected


def aliases_compatible(alias_a: str, alias_b: str) -> bool:
    """Return True if two port aliases can be connected.

    annotated_image outputs are NOT connectable to image_frame inputs —
    it's a visualization-only alias.
    """
    if alias_a not in ALIAS_COMPAT_MAP or alias_b not in ALIAS_COMPAT_MAP:
        return False
    # Annotated image cannot be wired to image_frame sinks
    if alias_a in _ANNOTATION_OUTPUT_ONLY and alias_b == "image_frame":
        return False
    if alias_b in _ANNOTATION_OUTPUT_ONLY and alias_a == "image_frame":
        return False
    return alias_a == alias_b


def get_block_by_id(blocks: List[Dict], block_id: str) -> Optional[Dict]:
    for b in blocks:
        if b.get("id") == block_id:
            return b
    return None
