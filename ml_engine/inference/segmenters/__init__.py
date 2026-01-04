"""
Segmentation models for auto-labeling.
"""

from ml_engine.inference.segmenters.base import SegmenterProtocol
from ml_engine.inference.segmenters.mobile_sam import MobileSAMSegmenter

__all__ = [
    "SegmenterProtocol",
    "MobileSAMSegmenter",
]
