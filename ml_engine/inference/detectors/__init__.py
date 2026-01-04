"""
Object detectors for auto-labeling.
"""

from ml_engine.inference.detectors.base import DetectorProtocol, DetectionResult
from ml_engine.inference.detectors.grounding_dino import GroundingDINODetector

__all__ = [
    "DetectorProtocol",
    "DetectionResult",
    "GroundingDINODetector",
]
