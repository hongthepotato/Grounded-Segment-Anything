"""
Auto-labeling service using Grounding DINO + MobileSAM.

This module provides automatic annotation generation for images using:
1. Grounding DINO: Text-prompted object detection (text -> boxes)
2. NMS: Non-Maximum Suppression to filter duplicate detections
3. MobileSAM: Box-prompted segmentation (boxes -> masks)

Output is in standard COCO format with both bounding boxes and segmentation masks.

Usage:
    from ml_engine.inference.auto_labeler import AutoLabeler
    
    labeler = AutoLabeler()
    coco_annotations = labeler.label_images(
        image_paths=['img1.jpg', 'img2.jpg'],
        class_prompts=['ear of bag', 'defect']
    )
"""
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torchvision.ops

# Add project paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "GroundingDINO"))
sys.path.insert(0, str(project_root / "segment_anything"))
# EfficientSAM directory contains MobileSAM as a subdirectory
sys.path.insert(0, str(project_root / "EfficientSAM"))

# Grounding DINO - reuse existing inference module
from groundingdino.util.inference import Model as GroundingDINOModel

# MobileSAM - reuse existing setup from EfficientSAM/MobileSAM/
# Note: MobileSAM is bundled in this project under EfficientSAM/MobileSAM/
# No need to install from https://github.com/ChaoningZhang/MobileSAM
from MobileSAM.setup_mobile_sam import setup_model as setup_mobile_sam
from segment_anything import SamPredictor

# Config utilities
from core.config import save_json

logger = logging.getLogger(__name__)


# Output mode options
OUTPUT_BOXES_ONLY = "boxes"
OUTPUT_MASKS_ONLY = "masks"
OUTPUT_BOTH = "both"


@dataclass
class AutoLabelerConfig:
    """Configuration for AutoLabeler."""
    # Model paths
    grounding_dino_config: str = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    grounding_dino_checkpoint: str = "data/models/pretrained/groundingdino_swint_ogc.pth"
    mobile_sam_checkpoint: str = "data/models/pretrained/mobile_sam.pt"

    # Detection thresholds
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    nms_threshold: float = 0.5

    # Output mode: "boxes", "masks", or "both"
    # - "boxes": Only bounding boxes (faster, no SAM needed, default)
    # - "masks": Only segmentation masks (boxes used internally but not in output)
    # - "both": Both boxes and masks
    output_mode: str = OUTPUT_BOXES_ONLY

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class AutoLabeler:
    """
    Auto-labeling service using Grounding DINO + MobileSAM.
    
    Pipeline:
    1. Grounding DINO: Detect objects based on text prompts
    2. NMS: Filter overlapping detections
    3. MobileSAM: Generate segmentation masks for each detection
    4. Export: Convert to COCO format
    
    Example:
        labeler = AutoLabeler()
        coco_json = labeler.label_images(
            image_paths=['img1.jpg', 'img2.jpg'],
            class_prompts=['ear of bag', 'defect']
        )
        # Returns COCO-format dict with images, annotations, categories
    """

    def __init__(self, config: Optional[AutoLabelerConfig] = None):
        """
        Initialize AutoLabeler with models.
        
        Args:
            config: Configuration object. If None, uses defaults.
        """
        self.config = config or AutoLabelerConfig()
        self.device = torch.device(self.config.device)

        # Models will be loaded lazily
        self._grounding_dino = None
        self._sam_predictor = None
        self._models_loaded = False

        logger.info("AutoLabeler initialized (device: %s)", self.device)

    def _load_models(self) -> None:
        """
        Load Grounding DINO and MobileSAM models.
        
        Reuses existing model loading patterns from:
        - grounded_sam_simple_demo.py (DINO)
        - EfficientSAM/grounded_mobile_sam.py (MobileSAM)
        
        Note: MobileSAM is only loaded if output_mode requires masks.
        """
        if self._models_loaded:
            return

        logger.info("Loading Grounding DINO model...")
        self._grounding_dino = GroundingDINOModel(
            model_config_path=self.config.grounding_dino_config,
            model_checkpoint_path=self.config.grounding_dino_checkpoint,
            device=str(self.device)
        )

        # Only load SAM if we need masks
        if self.config.output_mode in (OUTPUT_MASKS_ONLY, OUTPUT_BOTH):
            logger.info("Loading MobileSAM model...")
            mobile_sam = setup_mobile_sam()
            checkpoint = torch.load(self.config.mobile_sam_checkpoint, map_location="cpu")
            mobile_sam.load_state_dict(checkpoint, strict=True)
            mobile_sam.to(device=self.device)
            mobile_sam.eval()
            
            self._sam_predictor = SamPredictor(mobile_sam)
        else:
            logger.info("Skipping MobileSAM (output_mode='boxes')")
            self._sam_predictor = None
        
        self._models_loaded = True
        logger.info("Models loaded successfully")
    
    def _detect_objects(
        self,
        image: np.ndarray,
        class_prompts: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect objects using Grounding DINO.
        
        Args:
            image: BGR image (OpenCV format)
            class_prompts: List of class names to detect
            
        Returns:
            Tuple of (boxes_xyxy, confidences, class_ids) after NMS
        """
        # Use DINO's predict_with_classes (from grounded_sam_simple_demo.py pattern)
        detections = self._grounding_dino.predict_with_classes(
            image=image,
            classes=class_prompts,
            box_threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold
        )
        
        # Check if any detections
        if len(detections.xyxy) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Apply NMS (from all demo patterns)
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            self.config.nms_threshold
        ).numpy().tolist()
        
        boxes = detections.xyxy[nms_idx]
        confidences = detections.confidence[nms_idx]
        class_ids = detections.class_id[nms_idx]
        
        logger.debug(f"Detected {len(boxes)} objects after NMS")
        return boxes, confidences, class_ids
    
    def _segment_boxes(
        self,
        image: np.ndarray,
        boxes: np.ndarray
    ) -> List[np.ndarray]:
        """
        Generate segmentation masks for detected boxes using MobileSAM.
        
        Args:
            image: RGB image
            boxes: Array of boxes in xyxy format
            
        Returns:
            List of binary masks (one per box)
        """
        if len(boxes) == 0:
            return []
        
        # Set image (encodes once, from grounded_mobile_sam.py pattern)
        self._sam_predictor.set_image(image)
        
        masks = []
        for box in boxes:
            # Predict mask for this box
            mask_predictions, scores, _ = self._sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            # Select best mask (highest score)
            best_idx = np.argmax(scores)
            masks.append(mask_predictions[best_idx])
        
        return masks
    
    def label_single_image(
        self,
        image_path: str,
        class_prompts: List[str]
    ) -> Dict[str, Any]:
        """
        Generate annotations for a single image.
        
        Args:
            image_path: Path to image file
            class_prompts: List of class names to detect
            
        Returns:
            Dict with (depending on output_mode):
                - boxes: List of [x, y, w, h] in COCO format (if output_mode is 'boxes' or 'both')
                - masks: List of binary masks (if output_mode is 'masks' or 'both')
                - class_ids: List of category IDs
                - scores: List of confidence scores
                - image_info: Dict with file_name, width, height
        """
        self._load_models()
        
        # Load image
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_bgr.shape[:2]
        
        # Detect objects
        boxes_xyxy, confidences, class_ids = self._detect_objects(image_bgr, class_prompts)
        
        # Convert boxes to COCO format [x, y, width, height]
        boxes_coco = []
        for box in boxes_xyxy:
            x1, y1, x2, y2 = box
            boxes_coco.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
        
        # Generate masks if needed
        masks = []
        if self.config.output_mode in (OUTPUT_MASKS_ONLY, OUTPUT_BOTH):
            masks = self._segment_boxes(image_rgb, boxes_xyxy)
        
        # Build result based on output_mode
        result = {
            'class_ids': class_ids.tolist() if len(class_ids) > 0 else [],
            'scores': confidences.tolist() if len(confidences) > 0 else [],
            'image_info': {
                'file_name': os.path.basename(image_path),
                'width': width,
                'height': height
            }
        }
        
        # Include boxes if requested
        if self.config.output_mode in (OUTPUT_BOXES_ONLY, OUTPUT_BOTH):
            result['boxes'] = boxes_coco
        
        # Include masks if requested
        if self.config.output_mode in (OUTPUT_MASKS_ONLY, OUTPUT_BOTH):
            result['masks'] = masks
        
        return result
    
    def label_images(
        self,
        image_paths: List[str],
        class_prompts: List[str],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate COCO-format annotations for multiple images.
        
        Args:
            image_paths: List of paths to image files
            class_prompts: List of class names to detect
            output_path: Optional path to save COCO JSON
            
        Returns:
            COCO-format dictionary with images, annotations, categories.
            Annotation content depends on output_mode:
            - "boxes": bbox only, no segmentation
            - "masks": segmentation only, bbox auto-generated from mask
            - "both": both bbox and segmentation
        """
        self._load_models()
        
        # Initialize COCO structure
        coco_output = {
            'images': [],
            'annotations': [],
            'categories': [
                {'id': i, 'name': name}
                for i, name in enumerate(class_prompts)
            ]
        }
        
        annotation_id = 1
        output_mode = self.config.output_mode
        
        for image_id, image_path in enumerate(image_paths, start=1):
            logger.info(f"Processing image {image_id}/{len(image_paths)}: {image_path}")
            
            try:
                result = self.label_single_image(image_path, class_prompts)
            except Exception as e:
                logger.warning(f"Failed to process {image_path}: {e}")
                continue
            
            # Add image info
            coco_output['images'].append({
                'id': image_id,
                'file_name': result['image_info']['file_name'],
                'width': result['image_info']['width'],
                'height': result['image_info']['height']
            })
            
            # Get data based on output mode
            boxes = result.get('boxes', [])
            masks = result.get('masks', [])
            num_detections = len(result['class_ids'])
            
            # Add annotations
            for i in range(num_detections):
                class_id = result['class_ids'][i]
                score = result['scores'][i]
                
                annotation = {
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': int(class_id) if class_id is not None else 0,
                    'iscrowd': 0,
                    'score': float(score)
                }
                
                # Add bbox if available
                if output_mode in (OUTPUT_BOXES_ONLY, OUTPUT_BOTH) and i < len(boxes):
                    box = boxes[i]
                    annotation['bbox'] = box
                    annotation['area'] = box[2] * box[3]  # width * height
                
                # Add segmentation if available
                if output_mode in (OUTPUT_MASKS_ONLY, OUTPUT_BOTH) and i < len(masks):
                    mask = masks[i]
                    segmentation = self._mask_to_polygon(mask)
                    annotation['segmentation'] = segmentation
                    annotation['area'] = float(mask.sum()) if mask is not None and mask.sum() > 0 else annotation.get('area', 0)
                    
                    # For masks-only mode, generate bbox from mask
                    if output_mode == OUTPUT_MASKS_ONLY and 'bbox' not in annotation:
                        annotation['bbox'] = self._bbox_from_mask(mask)
                
                coco_output['annotations'].append(annotation)
                annotation_id += 1
        
        # Save if output path provided
        if output_path:
            save_json(coco_output, output_path)
            logger.info(f"Saved COCO annotations to: {output_path}")
        
        logger.info(
            f"Auto-labeling complete: {len(coco_output['images'])} images, "
            f"{len(coco_output['annotations'])} annotations (mode: {output_mode})"
        )
        
        return coco_output
    
    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> List[float]:
        """
        Generate bounding box from mask.
        
        Args:
            mask: Binary mask (H, W)
            
        Returns:
            Bounding box [x, y, width, height] in COCO format
        """
        if mask is None or mask.sum() == 0:
            return [0.0, 0.0, 0.0, 0.0]
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
    
    @staticmethod
    def _mask_to_polygon(mask: np.ndarray) -> List[List[float]]:
        """
        Convert binary mask to COCO polygon format.
        
        Args:
            mask: Binary mask (H, W)
            
        Returns:
            List of polygons, each polygon is [x1, y1, x2, y2, ...]
        """
        if mask is None or mask.sum() == 0:
            return []
        
        # Ensure mask is uint8
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        polygons = []
        for contour in contours:
            # Skip small contours
            if len(contour) < 3:
                continue
            
            # Flatten to [x1, y1, x2, y2, ...]
            polygon = contour.flatten().tolist()
            
            # COCO requires at least 6 points (3 vertices)
            if len(polygon) >= 6:
                polygons.append(polygon)
        
        return polygons


def export_to_coco(
    detections_list: List[Dict[str, Any]],
    class_prompts: List[str],
    output_mode: str = OUTPUT_BOTH
) -> Dict[str, Any]:
    """
    Convert a list of detection results to COCO format.
    
    This is a standalone function for flexibility.
    
    Args:
        detections_list: List of detection dicts (from label_single_image)
        class_prompts: List of class names
        output_mode: "boxes", "masks", or "both" (default: "both")
        
    Returns:
        COCO-format dictionary
    """
    coco_output = {
        'images': [],
        'annotations': [],
        'categories': [
            {'id': i, 'name': name}
            for i, name in enumerate(class_prompts)
        ]
    }
    
    annotation_id = 1
    
    for image_id, detection in enumerate(detections_list, start=1):
        # Add image info
        coco_output['images'].append({
            'id': image_id,
            'file_name': detection['image_info']['file_name'],
            'width': detection['image_info']['width'],
            'height': detection['image_info']['height']
        })
        
        # Get data based on what's available
        boxes = detection.get('boxes', [])
        masks = detection.get('masks', [])
        num_detections = len(detection['class_ids'])
        
        # Add annotations
        for i in range(num_detections):
            class_id = detection['class_ids'][i]
            score = detection['scores'][i]
            
            annotation = {
                'id': annotation_id,
                'image_id': image_id,
                'category_id': int(class_id) if class_id is not None else 0,
                'iscrowd': 0,
                'score': float(score)
            }
            
            # Add bbox if available
            if output_mode in (OUTPUT_BOXES_ONLY, OUTPUT_BOTH) and i < len(boxes):
                box = boxes[i]
                annotation['bbox'] = box
                annotation['area'] = box[2] * box[3]
            
            # Add segmentation if available
            if output_mode in (OUTPUT_MASKS_ONLY, OUTPUT_BOTH) and i < len(masks):
                mask = masks[i]
                segmentation = AutoLabeler._mask_to_polygon(mask)
                annotation['segmentation'] = segmentation
                annotation['area'] = float(mask.sum()) if mask is not None and mask.sum() > 0 else annotation.get('area', 0)
                
                # For masks-only mode, generate bbox from mask
                if output_mode == OUTPUT_MASKS_ONLY and 'bbox' not in annotation:
                    annotation['bbox'] = AutoLabeler._bbox_from_mask(mask)
            
            coco_output['annotations'].append(annotation)
            annotation_id += 1
    
    return coco_output
