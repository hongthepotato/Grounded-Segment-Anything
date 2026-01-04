"""
MobileSAM segmenter implementation.

Box-prompted segmentation using MobileSAM.
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

# Add project paths for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "deps" / "segment_anything"))
sys.path.insert(0, str(project_root / "EfficientSAM"))

from MobileSAM.setup_mobile_sam import setup_model as setup_mobile_sam
from segment_anything import SamPredictor

logger = logging.getLogger(__name__)


class MobileSAMSegmenter:
    """
    Segmenter using MobileSAM.
    
    Generates segmentation masks from bounding box prompts.
    
    Example:
        segmenter = MobileSAMSegmenter(
            checkpoint_path="data/models/pretrained/mobile_sam.pt",
            device="cuda"
        )
        masks = segmenter.segment(image_rgb, boxes_xyxy)
    """

    def __init__(
        self,
        checkpoint_path: str = "data/models/pretrained/mobile_sam.pt",
        device: str = "cuda"
    ):
        """
        Initialize MobileSAM segmenter.
        
        Args:
            checkpoint_path: Path to MobileSAM checkpoint
            device: Device for inference ("cuda" or "cpu")
        """
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)

        self._predictor: Optional[SamPredictor] = None

    def _load_model(self) -> None:
        """Load model lazily on first use."""
        if self._predictor is not None:
            return

        logger.info("Loading MobileSAM model...")
        mobile_sam = setup_mobile_sam()
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        mobile_sam.load_state_dict(checkpoint, strict=True)
        mobile_sam.to(device=self.device)
        mobile_sam.eval()
        self._predictor = SamPredictor(mobile_sam)
        logger.info("MobileSAM loaded successfully")

    def segment(
        self,
        image: np.ndarray,
        boxes: np.ndarray
    ) -> List[np.ndarray]:
        """
        Generate segmentation masks for detected boxes.
        
        Args:
            image: RGB image
            boxes: Array of boxes in xyxy format, shape (N, 4)
            
        Returns:
            List of binary masks, one per box
        """
        if len(boxes) == 0:
            return []

        self._load_model()

        # Set image (encodes once)
        self._predictor.set_image(image)

        masks = []
        for box in boxes:
            # Predict mask for this box
            mask_predictions, scores, _ = self._predictor.predict(
                box=box,
                multimask_output=True
            )
            # Select best mask (highest score)
            best_idx = np.argmax(scores)
            masks.append(mask_predictions[best_idx])

        return masks
