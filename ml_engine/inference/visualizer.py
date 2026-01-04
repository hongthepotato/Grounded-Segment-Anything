"""
Visualization utilities for auto-labeling results.

Provides functions to draw bounding boxes, masks, and labels on images.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np

from ml_engine.inference.config import (
    OUTPUT_BOXES_ONLY,
    OUTPUT_MASKS_ONLY,
    OUTPUT_BOTH,
)

logger = logging.getLogger(__name__)

# Color palette (BGR format for OpenCV)
COLORS = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 255),  # Purple
    (255, 128, 0),  # Orange
]


def visualize_detections(
    image_path: str,
    result: Dict[str, Any],
    class_prompts: List[str],
    output_path: str,
    show_boxes: bool = True,
    show_masks: bool = True,
    show_labels: bool = True,
    show_scores: bool = True
) -> None:
    """
    Visualize auto-labeling results on an image.
    
    Draws bounding boxes and/or masks with class labels and confidence scores.
    
    Args:
        image_path: Path to original image
        result: Detection result from label_single_image()
        class_prompts: List of class names
        output_path: Path to save visualized image
        show_boxes: Whether to draw bounding boxes
        show_masks: Whether to overlay masks
        show_labels: Whether to show class labels
        show_scores: Whether to show confidence scores
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        logger.warning(f"Could not load image for visualization: {image_path}")
        return

    boxes = result.get('boxes', [])
    masks = result.get('masks', [])
    class_ids = result.get('class_ids', [])
    scores = result.get('scores', [])

    # Draw masks first (so boxes appear on top)
    if show_masks and masks:
        mask_overlay = image.copy()
        for i, mask in enumerate(masks):
            if mask is None:
                continue
            color = COLORS[class_ids[i] % len(COLORS)] if i < len(class_ids) else COLORS[0]
            # Create colored mask
            colored_mask = np.zeros_like(image)
            colored_mask[mask > 0] = color
            # Blend with original
            mask_overlay = cv2.addWeighted(
                mask_overlay, 1.0,
                colored_mask, 0.4,
                0
            )
            # Draw mask contour
            contours, _ = cv2.findContours(
                (mask > 0).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(mask_overlay, contours, -1, color, 2)
        image = mask_overlay

    # Draw boxes
    if show_boxes and boxes:
        for i, box in enumerate(boxes):
            class_id = class_ids[i] if i < len(class_ids) else 0
            score = scores[i] if i < len(scores) else 0.0
            color = COLORS[class_id % len(COLORS)]

            # COCO format [x, y, w, h] -> xyxy
            x, y, w, h = box
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Build label text
            if show_labels or show_scores:
                label_parts = []
                if show_labels:
                    class_name = class_prompts[class_id] if class_id < len(class_prompts) else f"cls_{class_id}"
                    label_parts.append(class_name)
                if show_scores:
                    label_parts.append(f"{score:.2f}")
                label_text = " ".join(label_parts)
                
                # Draw label background
                (text_w, text_h), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    image,
                    (x1, y1 - text_h - 8),
                    (x1 + text_w + 4, y1),
                    color,
                    -1
                )
                # Draw label text
                cv2.putText(
                    image, label_text,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1
                )

    # Save visualization
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, image)
    logger.debug(f"Saved visualization: {output_path}")


def visualize_batch(
    image_paths: List[str],
    results: List[Dict[str, Any]],
    class_prompts: List[str],
    output_dir: str,
    output_mode: str = OUTPUT_BOXES_ONLY
) -> int:
    """
    Visualize auto-labeling results for multiple images.
    
    Args:
        image_paths: List of paths to original images
        results: List of detection results from label_single_image()
        class_prompts: List of class names
        output_dir: Directory to save visualizations
        output_mode: "boxes", "masks", or "both"
        
    Returns:
        Number of images visualized
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    show_boxes = output_mode in (OUTPUT_BOXES_ONLY, OUTPUT_BOTH)
    show_masks = output_mode in (OUTPUT_MASKS_ONLY, OUTPUT_BOTH)

    count = 0
    for image_path, result in zip(image_paths, results):
        filename = Path(image_path).stem + "_viz.jpg"
        save_path = str(output_path / filename)

        visualize_detections(
            image_path=image_path,
            result=result,
            class_prompts=class_prompts,
            output_path=save_path,
            show_boxes=show_boxes,
            show_masks=show_masks
        )
        count += 1

    logger.info(f"Saved {count} visualizations to: {output_dir}")
    return count
