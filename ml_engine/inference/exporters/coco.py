"""
COCO format exporter.

Single source of truth for converting detection results to COCO format.
"""

from typing import List, Dict, Any

import cv2
import numpy as np

from ml_engine.inference.config import (
    OUTPUT_BOXES_ONLY,
    OUTPUT_MASKS_ONLY,
    OUTPUT_BOTH,
)


class COCOExporter:
    """
    Exports detection results to COCO format.
    
    This is the single source of truth for COCO format generation.
    All code that needs COCO output should use this class.
    """

    @staticmethod
    def export(
        results: List[Dict[str, Any]],
        class_prompts: List[str],
        output_mode: str = OUTPUT_BOTH
    ) -> Dict[str, Any]:
        """
        Convert detection results to COCO format.
        
        Args:
            results: List of detection results, each containing:
                - class_ids: List of class IDs
                - scores: List of confidence scores
                - boxes: List of [x, y, w, h] boxes (if output_mode includes boxes)
                - masks: List of binary masks (if output_mode includes masks)
                - image_info: Dict with file_name, width, height
            class_prompts: List of class names
            output_mode: "boxes", "masks", or "both"
            
        Returns:
            COCO-format dictionary with images, annotations, categories
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

        for image_id, result in enumerate(results, start=1):
            # Add image info
            coco_output['images'].append({
                'id': image_id,
                'file_name': result['image_info']['file_name'],
                'width': result['image_info']['width'],
                'height': result['image_info']['height']
            })

            # Get data based on what's available
            boxes = result.get('boxes', [])
            masks = result.get('masks', [])
            class_ids = result.get('class_ids', [])
            scores = result.get('scores', [])
            num_detections = len(class_ids)

            # Add annotations
            for i in range(num_detections):
                class_id = class_ids[i]
                score = scores[i] if i < len(scores) else 0.0

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
                    segmentation = COCOExporter.mask_to_polygon(mask)
                    annotation['segmentation'] = segmentation
                    if mask is not None and hasattr(mask, 'sum'):
                        annotation['area'] = float(mask.sum())

                    # For masks-only mode, generate bbox from mask
                    if output_mode == OUTPUT_MASKS_ONLY and 'bbox' not in annotation:
                        annotation['bbox'] = COCOExporter.bbox_from_mask(mask)

                coco_output['annotations'].append(annotation)
                annotation_id += 1

        return coco_output

    @staticmethod
    def bbox_from_mask(mask: np.ndarray) -> List[float]:
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
    def mask_to_polygon(mask: np.ndarray) -> List[List[float]]:
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
