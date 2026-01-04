"""
Base protocol for segmenters.

Defines the interface that all segmenters must implement.
"""

from typing import Protocol, List, runtime_checkable

import numpy as np


@runtime_checkable
class SegmenterProtocol(Protocol):
    """
    Protocol for segmentation models.
    
    Implementations generate segmentation masks from bounding boxes.
    """

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
        ...
