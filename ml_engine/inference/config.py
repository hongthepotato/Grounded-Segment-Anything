"""
Configuration for auto-labeling inference.

Contains AutoLabelerConfig and output mode constants.
"""

from dataclasses import dataclass

import torch


# Output mode options
OUTPUT_BOXES_ONLY = "boxes"
OUTPUT_MASKS_ONLY = "masks"
OUTPUT_BOTH = "both"


@dataclass
class AutoLabelerConfig:
    """
    Configuration for AutoLabeler.
    
    Attributes:
        grounding_dino_config: Path to GroundingDINO config file
        grounding_dino_checkpoint: Path to GroundingDINO weights
        mobile_sam_checkpoint: Path to MobileSAM weights
        box_threshold: Detection confidence threshold
        text_threshold: Text matching threshold
        nms_threshold: Non-maximum suppression threshold
        output_mode: Output format - "boxes", "masks", or "both"
        device: Device for inference - "cuda" or "cpu"
    """
    # Model paths
    grounding_dino_config: str = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    grounding_dino_checkpoint: str = "data/models/pretrained/groundingdino_swint_ogc.pth"
    mobile_sam_checkpoint: str = "data/models/pretrained/mobile_sam.pt"

    # Detection thresholds
    box_threshold: float = 0.5
    text_threshold: float = 0.5
    nms_threshold: float = 0.7

    # Output mode
    output_mode: str = OUTPUT_BOXES_ONLY

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
