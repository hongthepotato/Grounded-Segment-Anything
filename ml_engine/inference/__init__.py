"""
Inference module for auto-labeling and model inference.

Public API:
    - AutoLabeler: Main coordinator class for auto-labeling
    - AutoLabelerConfig: Configuration dataclass
    - COCOExporter: Export results to COCO format
    - visualize_detections: Visualize single image results
    - visualize_batch: Visualize multiple image results
    - OUTPUT_BOXES_ONLY, OUTPUT_MASKS_ONLY, OUTPUT_BOTH: Output mode constants

Extensibility:
    - detectors/: Object detection backends (GroundingDINO, future: YOLO, RT-DETR)
    - segmenters/: Segmentation backends (MobileSAM, future: SAM2, EfficientSAM)
    - exporters/: Output format converters (COCO, future: YOLO, VOC)
"""

# Configuration
from ml_engine.inference.config import (
    AutoLabelerConfig,
    OUTPUT_BOXES_ONLY,
    OUTPUT_MASKS_ONLY,
    OUTPUT_BOTH,
)

# Main coordinator
from ml_engine.inference.auto_labeler import AutoLabeler

# Exporters
from ml_engine.inference.exporters.coco import COCOExporter

# Visualization
from ml_engine.inference.visualizer import (
    visualize_detections,
    visualize_batch,
)

__all__ = [
    # Main API
    'AutoLabeler',
    'AutoLabelerConfig',
    'COCOExporter',

    # Visualization
    'visualize_detections',
    'visualize_batch',

    # Constants
    'OUTPUT_BOXES_ONLY',
    'OUTPUT_MASKS_ONLY',
    'OUTPUT_BOTH',
]
