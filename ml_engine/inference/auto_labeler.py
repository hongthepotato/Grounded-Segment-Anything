"""
Auto-labeling service using Grounding DINO + MobileSAM.

This module provides automatic annotation generation for images using:
1. Grounding DINO: Text-prompted object detection (text -> boxes)
2. NMS: Non-Maximum Suppression to filter duplicate detections
3. MobileSAM: Box-prompted segmentation (boxes -> masks)

This is the coordinator class that delegates to:
- detectors/ for object detection
- segmenters/ for mask generation
- exporters/ for output format conversion

Usage:
    from ml_engine.inference import AutoLabeler, AutoLabelerConfig, COCOExporter
    
    labeler = AutoLabeler(config)
    results = labeler.label_images(image_paths, class_prompts)
    coco_output = COCOExporter.export(results, class_prompts)
"""

import os
import logging
from typing import List, Dict, Any, Optional, Callable

import cv2

from ml_engine.inference.config import (
    AutoLabelerConfig,
    OUTPUT_BOXES_ONLY,
    OUTPUT_MASKS_ONLY,
    OUTPUT_BOTH,
)
from ml_engine.inference.detectors.grounding_dino import GroundingDINODetector
from ml_engine.inference.segmenters.mobile_sam import MobileSAMSegmenter

logger = logging.getLogger(__name__)


class AutoLabeler:
    """
    Auto-labeling coordinator using Grounding DINO + MobileSAM.
    
    This class coordinates detection and segmentation to generate
    annotations for images based on text prompts.
    
    Uses sequential (single-image) inference for consistent performance
    with variable-sized images.
    
    Example:
        config = AutoLabelerConfig(
            box_threshold=0.5,
            output_mode="both"
        )
        labeler = AutoLabeler(config)
        results = labeler.label_images(
            image_paths=['img1.jpg', 'img2.jpg'],
            class_prompts=['ear of bag', 'defect']
        )
        # Use COCOExporter to convert results to COCO format
    """

    def __init__(self, config: Optional[AutoLabelerConfig] = None):
        """
        Initialize AutoLabeler.
        
        Args:
            config: Configuration object. If None, uses defaults.
        """
        self.config = config or AutoLabelerConfig()

        # Initialize detector and segmenter (lazy loading)
        self._detector: Optional[GroundingDINODetector] = None
        self._segmenter: Optional[MobileSAMSegmenter] = None

        logger.info("AutoLabeler initialized (device: %s, mode: %s)",
                   self.config.device, self.config.output_mode)

    def _get_detector(self) -> GroundingDINODetector:
        """Get or create detector instance."""
        if self._detector is None:
            self._detector = GroundingDINODetector(
                config_path=self.config.grounding_dino_config,
                checkpoint_path=self.config.grounding_dino_checkpoint,
                device=self.config.device
            )
        return self._detector

    def _get_segmenter(self) -> MobileSAMSegmenter:
        """Get or create segmenter instance."""
        if self._segmenter is None:
            self._segmenter = MobileSAMSegmenter(
                checkpoint_path=self.config.mobile_sam_checkpoint,
                device=self.config.device
            )
        return self._segmenter

    def label_images(
        self,
        image_paths: List[str],
        class_prompts: List[str],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate annotations for multiple images.
        
        This is the main entry point for auto-labeling. It handles:
        - Loading images
        - Sequential detection (one image at a time)
        - Per-image segmentation (if needed)
        - Progress reporting
        
        Args:
            image_paths: List of paths to image files
            class_prompts: List of class names to detect
            progress_callback: Optional callback(current, total, message)
            
        Returns:
            List of result dicts, each containing:
                - class_ids: List of class IDs
                - scores: List of confidence scores
                - boxes: List of [x, y, w, h] (if output_mode includes boxes)
                - masks: List of binary masks (if output_mode includes masks)
                - image_info: Dict with file_name, width, height
        """
        if len(image_paths) == 0:
            return []

        detector = self._get_detector()
        needs_masks = self.config.output_mode in (OUTPUT_MASKS_ONLY, OUTPUT_BOTH)
        segmenter = self._get_segmenter() if needs_masks else None

        results = []
        total_images = len(image_paths)

        # Sequential processing - one image at a time
        for i, image_path in enumerate(image_paths):
            # Load image
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                logger.warning("Could not load image: %s", image_path)
                results.append(self._empty_result(image_path))
                continue

            height, width = image_bgr.shape[:2]
            file_name = os.path.basename(image_path)

            # Single-image detection
            detection = detector.detect(
                image=image_bgr,
                prompts=class_prompts,
                box_threshold=self.config.box_threshold,
                text_threshold=self.config.text_threshold,
                nms_threshold=self.config.nms_threshold
            )

            # Convert boxes to COCO format [x, y, width, height]
            boxes_coco = []
            for box in detection.boxes_xyxy:
                x1, y1, x2, y2 = box
                boxes_coco.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])

            # Generate masks if needed
            masks = []
            if needs_masks and segmenter is not None and len(detection.boxes_xyxy) > 0:
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                masks = segmenter.segment(image_rgb, detection.boxes_xyxy)

            # Build result
            result = {
                'class_ids': detection.class_ids.tolist() if len(detection.class_ids) > 0 else [],
                'scores': detection.confidences.tolist() if len(detection.confidences) > 0 else [],
                'image_info': {
                    'file_name': file_name,
                    'width': width,
                    'height': height
                }
            }

            if self.config.output_mode in (OUTPUT_BOXES_ONLY, OUTPUT_BOTH):
                result['boxes'] = boxes_coco

            if self.config.output_mode in (OUTPUT_MASKS_ONLY, OUTPUT_BOTH):
                result['masks'] = masks

            results.append(result)

            # Report progress
            if progress_callback:
                progress_callback(i + 1, total_images, f"Processed {file_name}")

        logger.info("Labeled %d images", len(results))
        return results

    def _empty_result(self, image_path: str) -> Dict[str, Any]:
        """Create empty result for failed image."""
        result = {
            'class_ids': [],
            'scores': [],
            'image_info': {
                'file_name': os.path.basename(image_path),
                'width': 0,
                'height': 0
            }
        }
        if self.config.output_mode in (OUTPUT_BOXES_ONLY, OUTPUT_BOTH):
            result['boxes'] = []
        if self.config.output_mode in (OUTPUT_MASKS_ONLY, OUTPUT_BOTH):
            result['masks'] = []
        return result
