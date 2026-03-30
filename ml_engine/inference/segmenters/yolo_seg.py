"""
YOLOv8-seg inference wrapper for prompt-free instance segmentation.

Loads a trained YOLOv8-seg model and outputs polygon coordinates.
This is the end product of the distillation pipeline.
"""

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class YOLOSegInference:
    """
    Prompt-free instance segmentation using a trained YOLOv8-seg model.

    Input: raw image (numpy RGB or BGR, or file path)
    Output: list of detections with polygon coordinates

    Example:
        model = YOLOSegInference("experiments/student/weights/best.pt")
        detections = model.predict(image_rgb)
        for det in detections:
            print(det['class_id'], det['confidence'], det['polygon'])
    """

    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        conf: float = 0.5,
    ):
        from ultralytics import YOLO

        self.device = device
        self.conf = conf
        self.model = YOLO(model_path)
        logger.info("Loaded YOLOv8-seg model from %s", model_path)

    def predict(self, image: np.ndarray) -> List[Dict]:
        """
        Run prompt-free inference on a single image.

        Args:
            image: Image as numpy array (H, W, 3), RGB or BGR

        Returns:
            List of detection dicts, each containing:
                - class_id (int)
                - class_name (str)
                - confidence (float)
                - bbox (List[float]): [x1, y1, x2, y2]
                - polygon (List[List[float]]): [[x1,y1], [x2,y2], ...] in pixel coords
        """
        results = self.model(
            image,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        return self._parse_results(results[0])

    def _parse_results(self, result) -> List[Dict]:
        """Extract structured detections from a single ultralytics Result."""
        detections = []

        boxes = result.boxes
        masks = result.masks
        names = result.names or {}

        if boxes is None or len(boxes) == 0:
            return detections

        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].item())
            confidence = float(boxes.conf[i].item())
            bbox = boxes.xyxy[i].cpu().tolist()

            polygon = []
            if masks is not None and i < len(masks.xy):
                polygon = masks.xy[i].tolist()

            detections.append({
                'class_id': class_id,
                'class_name': names.get(class_id, str(class_id)),
                'confidence': confidence,
                'bbox': bbox,
                'polygon': polygon,
            })

        return detections
