"""
GroundingDINO detector implementation.

Text-prompted object detection using Grounding DINO.
Uses sequential (single-image) inference for consistent performance
with variable-sized images.
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torchvision.ops

# Add project paths for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "GroundingDINO"))

from groundingdino.util.inference import Model as GroundingDINOModel

from ml_engine.inference.detectors.base import DetectionResult

logger = logging.getLogger(__name__)


class GroundingDINODetector:
    """
    Object detector using Grounding DINO.
    
    Detects objects based on text prompts using the GroundingDINO model.
    
    Example:
        detector = GroundingDINODetector(
            config_path="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            checkpoint_path="data/models/pretrained/groundingdino_swint_ogc.pth",
            device="cuda"
        )
        result = detector.detect(image, ["cat", "dog"])
    """

    def __init__(
        self,
        config_path: str = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        checkpoint_path: str = "data/models/pretrained/groundingdino_swint_ogc.pth",
        device: str = "cuda"
    ):
        """
        Initialize GroundingDINO detector.
        
        Args:
            config_path: Path to GroundingDINO config file
            checkpoint_path: Path to model checkpoint
            device: Device for inference ("cuda" or "cpu")
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)

        self._model: Optional[GroundingDINOModel] = None

    def _load_model(self) -> None:
        """Load model lazily on first use."""
        if self._model is not None:
            return

        logger.info("Loading Grounding DINO model...")
        self._model = GroundingDINOModel(
            model_config_path=self.config_path,
            model_checkpoint_path=self.checkpoint_path,
            device=str(self.device)
        )
        logger.info("Grounding DINO loaded successfully")

    def detect(
        self,
        image: np.ndarray,
        prompts: List[str],
        box_threshold: float = 0.5,
        text_threshold: float = 0.5,
        nms_threshold: float = 0.7,
    ) -> DetectionResult:
        """
        Detect objects in a single image.
        
        Args:
            image: BGR image (OpenCV format)
            prompts: List of class names to detect
            box_threshold: Detection confidence threshold
            text_threshold: Text matching threshold
            nms_threshold: NMS threshold
            
        Returns:
            DetectionResult with boxes, confidences, and class_ids
        """
        self._load_model()

        # Use DINO's predict_with_classes
        detections = self._model.predict_with_classes(
            image=image,
            classes=prompts,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        # Check if any detections
        if len(detections.xyxy) == 0:
            return DetectionResult(
                boxes_xyxy=np.array([]),
                confidences=np.array([]),
                class_ids=np.array([])
            )

        # Apply NMS
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            nms_threshold
        ).numpy().tolist()

        return DetectionResult(
            boxes_xyxy=detections.xyxy[nms_idx],
            confidences=detections.confidence[nms_idx],
            class_ids=detections.class_id[nms_idx]
        )
