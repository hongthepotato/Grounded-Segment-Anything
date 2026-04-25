"""
Configuration contracts for inference pipelines.

This module defines typed model-loading specs and runtime policy knobs used by
AutoLabeler and distillation pseudo-labeling.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch

# Output mode options
OUTPUT_BOXES_ONLY = "boxes"
OUTPUT_MASKS_ONLY = "masks"
OUTPUT_BOTH = "both"

# Segmenter backend options
SEGMENTER_MOBILE_SAM = "mobile_sam"
SEGMENTER_SAM_HQ = "sam_hq"

# Detector model source options
DETECTOR_SOURCE_CHECKPOINT = "checkpoint"
DETECTOR_SOURCE_BASE_LORA = "base_plus_lora"


@dataclass
class DetectionThresholds:
    """Detection threshold policy."""

    box: float = 0.5
    text: float = 0.5
    nms: float = 0.7


@dataclass
class GroundingDINOModelSpec:
    """
    GroundingDINO model loading specification.

    source:
      - checkpoint: load directly from checkpoint_path
      - base_plus_lora: merge LoRA into base checkpoint at load time
    """

    source: str = DETECTOR_SOURCE_CHECKPOINT
    config_path: str = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    checkpoint_path: Optional[str] = "data/models/pretrained/groundingdino_swint_ogc.pth"
    base_checkpoint: Optional[str] = None
    lora_adapter_path: Optional[str] = None
    merged_cache_path: Optional[str] = None


@dataclass
class SegmenterModelSpec:
    """Segmentation backend loading specification."""

    backend: str = SEGMENTER_MOBILE_SAM
    checkpoint_path: Optional[str] = "data/models/pretrained/mobile_sam.pt"
    base_checkpoint: Optional[str] = None
    lora_adapter_path: Optional[str] = None
    model_type: str = "vit_b"


@dataclass
class AutoLabelerConfig:
    """
    Configuration for AutoLabeler.

    This config is intentionally split into:
    - model specs (what to load)
    - runtime policy (thresholds, output mode, device)
    """

    detector: GroundingDINOModelSpec = field(default_factory=GroundingDINOModelSpec)
    segmenter: SegmenterModelSpec = field(default_factory=SegmenterModelSpec)
    thresholds: DetectionThresholds = field(default_factory=DetectionThresholds)
    output_mode: str = OUTPUT_BOXES_ONLY
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
