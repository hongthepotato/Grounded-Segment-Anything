"""
Inference module for auto-labeling and model inference.
"""
from ml_engine.inference.auto_labeler import (
    AutoLabeler,
    AutoLabelerConfig,
    export_to_coco,
    OUTPUT_BOXES_ONLY,
    OUTPUT_MASKS_ONLY,
    OUTPUT_BOTH,
)

__all__ = [
    'AutoLabeler',
    'AutoLabelerConfig',
    'export_to_coco',
    'OUTPUT_BOXES_ONLY',
    'OUTPUT_MASKS_ONLY',
    'OUTPUT_BOTH',
]

