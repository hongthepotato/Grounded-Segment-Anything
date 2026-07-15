"""
Constants and default values for the platform.

This module defines TRUE constants that should not change:
- Directory paths
- Model names (string literals)
- Pretrained model URLs
- Model architectural parameters (input sizes, normalization)
- Validation enums (annotation modes, export formats)
- Logging configuration

NOTE: Configurable training defaults are in configs/defaults/*.yaml
      Do NOT add training hyperparameters here - use YAML configs instead.
"""

from pathlib import Path
from typing import List

# ============================================================================
# Directory Paths
# ============================================================================

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "configs"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
LOGS_DIR = PROJECT_ROOT / "logs"

# Data subdirectories
MODELS_DIR = DATA_DIR / "models"
PRETRAINED_MODELS_DIR = MODELS_DIR / "pretrained"

# Config subdirectories
DEFAULT_CONFIGS_DIR = CONFIGS_DIR / "defaults"
EXPERIMENT_CONFIGS_DIR = CONFIGS_DIR / "experiments"

# ============================================================================
# Image Path Transformation
# ============================================================================

# Actual filesystem base path for images
IMAGE_PATH_BASE = "/srv/shared/images/"


def transform_image_path(path: str) -> str:
    """
    Transform frontend/COCO image path to actual filesystem path.

    Frontend sends paths like: upload/2025/12/17/xxx.png
    Transforms to:            /srv/shared/images/upload/2025/12/17/xxx.png

    Args:
        path: Image path from frontend or COCO file_name (e.g., "upload/...")

    Returns:
        Actual filesystem path
    """
    if path.startswith("upload/"):
        # upload/... -> /srv/shared/images/upload/...
        return IMAGE_PATH_BASE + path
    return path  # Return as-is if no transform needed


# ============================================================================
# Model Names (String Literals)
# ============================================================================

# Teacher models
GROUNDING_DINO = "grounding_dino"
SAM = "sam"
POSE_MODEL = "pose_model"

# Student models - Detection
YOLOV8_N = "yolov8n"
YOLOV8_S = "yolov8s"
YOLOV8_M = "yolov8m"
YOLOV8_L = "yolov8l"
YOLOV8_X = "yolov8x"

# Student models - Segmentation
YOLOV8_N_SEG = "yolov8n-seg"
YOLOV8_S_SEG = "yolov8s-seg"
YOLOV8_M_SEG = "yolov8m-seg"
YOLOV8_L_SEG = "yolov8l-seg"
YOLOV8_X_SEG = "yolov8x-seg"

# Alternative lightweight models
FASTSAM_S = "fastsam-s"
FASTSAM_X = "fastsam-x"
MOBILESAM = "mobilesam"

# All teacher models list
TEACHER_MODELS: List[str] = [GROUNDING_DINO, SAM, POSE_MODEL]

# All student detection models
STUDENT_DETECTION_MODELS: List[str] = [YOLOV8_N, YOLOV8_S, YOLOV8_M, YOLOV8_L, YOLOV8_X]

# All student segmentation models
STUDENT_SEGMENTATION_MODELS: List[str] = [
    YOLOV8_N_SEG,
    YOLOV8_S_SEG,
    YOLOV8_M_SEG,
    YOLOV8_L_SEG,
    YOLOV8_X_SEG,
    FASTSAM_S,
    FASTSAM_X,
    MOBILESAM,
]

# ============================================================================
# Pretrained Model URLs and Paths
# ============================================================================

PRETRAINED_MODEL_URLS = {
    "groundingdino_swint_ogc": {
        "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
        "filename": "groundingdino_swint_ogc.pth",
        "size_mb": 2900,
    },
    "sam_vit_h": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "filename": "sam_vit_h_4b8939.pth",
        "size_mb": 2400,
    },
    "sam_vit_l": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "filename": "sam_vit_l_0b3195.pth",
        "size_mb": 1200,
    },
    "sam_vit_b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "filename": "sam_vit_b_01ec64.pth",
        "size_mb": 375,
    },
}

# ============================================================================
# Model Input Sizes (Architectural Constants)
# These are fixed by the model architecture and should not be changed
# ============================================================================

MODEL_INPUT_SIZES = {
    GROUNDING_DINO: {"min_size": 800, "max_size": 1333},
    SAM: {"height": 1024, "width": 1024},
    "yolov8": 640,
    "fastsam": 1024,
}

# Normalization parameters (determined by model pretraining)
MODEL_NORMALIZATION = {
    GROUNDING_DINO: {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "pixel_range": [0, 1],
    },
    SAM: {
        "mean": [123.675, 116.28, 103.53],
        "std": [58.395, 57.12, 57.375],
        "pixel_range": [0, 255],
    },
    "yolov8": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0], "pixel_range": [0, 1]},
}

# ============================================================================
# Annotation Modes (for data inspection)
# ============================================================================

# Annotation mode values (used in inspect_dataset return value)
ANNOTATION_MODE_DETECTION = "DETECTION_ONLY"
ANNOTATION_MODE_SEGMENTATION = "SEGMENTATION_ONLY"
ANNOTATION_MODE_COMBINED = "DETECTION_AND_SEGMENTATION"

# All valid annotation modes
ANNOTATION_MODES: List[str] = [
    ANNOTATION_MODE_DETECTION,
    ANNOTATION_MODE_SEGMENTATION,
    ANNOTATION_MODE_COMBINED,
]

# Mode for model selection (simpler format)
MODE_DETECTION = "detection"
MODE_SEGMENTATION = "segmentation"
MODE_COMBINED = "combined"

# ============================================================================
# Data Augmentation (Validation Lists)
# ============================================================================

# Available object characteristics
OBJECT_CHARACTERISTICS: List[str] = [
    "changes_shape",
    "changes_size",
    "reflective_surface",
    "low_contrast",
    "moves_or_vibrates",
    "semi_transparent",
    "similar_to_background",
    "multiple_objects",
    "partially_hidden",
]

# Environment conditions
ENVIRONMENT_CONDITIONS = {
    "lighting": ["stable", "variable", "poor"],
    "camera": ["fixed", "moving", "shaky"],
    "background": ["clean", "busy", "changing"],
    "distance": ["fixed", "variable", "close"],
}

# Augmentation intensity levels
AUGMENTATION_INTENSITIES: List[str] = ["low", "medium", "high"]

# ============================================================================
# Evaluation Metrics
# ============================================================================

# Detection metrics
DETECTION_METRICS: List[str] = ["mAP50", "mAP50-95", "precision", "recall", "f1"]

# Segmentation metrics
SEGMENTATION_METRICS: List[str] = ["mask_IoU", "mask_precision", "mask_recall"]

# ============================================================================
# Export Formats
# ============================================================================

SUPPORTED_EXPORT_FORMATS: List[str] = ["onnx", "tensorrt", "tflite", "openvino"]

QUANTIZATION_MODES: List[str] = ["int8", "fp16", "fp32"]

# ============================================================================
# Device Settings
# ============================================================================

# Edge device targets
EDGE_DEVICES = {
    "jetson_orin": {"compute": "high", "memory": "high"},
    "jetson_xavier": {"compute": "medium", "memory": "medium"},
    "jetson_nano": {"compute": "low", "memory": "low"},
    "raspberry_pi": {"compute": "very_low", "memory": "low"},
    "mobile": {"compute": "low", "memory": "low"},
}

# ============================================================================
# Logging Configuration
# ============================================================================

# Actual format strings (for logging.Formatter)
LOG_FORMAT_STRING = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
DATE_FORMAT_STRING = "%Y-%m-%d %H:%M:%S"

# Format type identifiers (used to select formatter class)
FORMAT_TYPE_TEXT = "text"
FORMAT_TYPE_JSON = "json"

# Default log level
DEFAULT_LOG_LEVEL = "INFO"

# Valid log levels
VALID_LOG_LEVELS: List[str] = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# ============================================================================
# Default LoRA Configurations
# ============================================================================

DEFAULT_DINO_LORA_CONFIG = {
    "lora": {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["out_proj", "in_proj", "linear1", "linear2"],
        "lora_dropout": 0.1,
    }
}

DEFAULT_SAM_LORA_CONFIG = {
    "lora": {"r": 16, "lora_alpha": 32, "target_modules": ["qkv", "proj"], "lora_dropout": 0.1}
}

# ============================================================================
# Agent / Pipeline Error Classification
# ============================================================================

# Exception types that indicate a transient infrastructure failure (network
# blip, Redis restart, worker OOM) rather than a logic bug. Used by the
# Coordinator crash handler to decide whether to retry via failed_retrying
# or to go straight to failed_unrecoverable.
#
# These are specific OSError subclasses, NOT OSError itself. Using the parent
# OSError would catch FileNotFoundError, PermissionError, ChildProcessError,
# etc. — permanent failures that should never trigger a retry.
#
# redis.exceptions are NOT included here because they require the redis
# package to be imported, which would add a hard dependency to core/. The
# crash handler checks those types separately via a lazy import.
TRANSIENT_EXCEPTION_TYPES = (
    ConnectionError,  # includes BrokenPipeError, ConnectionReset, ConnectionRefused
    TimeoutError,  # system-level operation timeout
    InterruptedError,  # EINTR — system call interrupted by a signal
)

# ============================================================================
# Version Info
# ============================================================================

PLATFORM_VERSION = "0.1.0"
PLATFORM_NAME = "Grounded-SAM Edge Deployment Platform"
