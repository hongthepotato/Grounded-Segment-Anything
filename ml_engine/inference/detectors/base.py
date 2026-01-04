"""
Base protocol for object detectors.

Defines the interface that all detectors must implement.
"""

from dataclasses import dataclass
from typing import Protocol, List, runtime_checkable

import numpy as np


@dataclass
class DetectionResult:
    """
    Result from object detection.
    
    Attributes:
        boxes_xyxy: Array of boxes in [x1, y1, x2, y2] format, shape (N, 4)
        confidences: Array of confidence scores, shape (N,)
        class_ids: Array of class IDs, shape (N,)
    """
    boxes_xyxy: np.ndarray
    confidences: np.ndarray
    class_ids: np.ndarray

    def __len__(self) -> int:
        return len(self.boxes_xyxy)

    @property
    def is_empty(self) -> bool:
        return len(self.boxes_xyxy) == 0


@runtime_checkable
class DetectorProtocol(Protocol):
    """
    Protocol for object detectors.
    
    Implementations detect objects in images based on text prompts.
    Uses sequential (single-image) inference for simplicity and
    consistent performance with variable-sized images.
    """

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
        ...
