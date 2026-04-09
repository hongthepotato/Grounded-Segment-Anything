"""
Labeled-only YOLOv8-seg training job handler.

Trains ``yolov8n-seg`` from COCO labels (no teachers, no pseudo-labeling).
Reuses the student distillation pipeline's COCO→YOLO conversion and StudentTrainer.
"""

import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict

from ml_engine.jobs.handlers.base import JobHandler

logger = logging.getLogger(__name__)


class YOLOSegLabeledHandler(JobHandler):
    """
    Job type ``yolo_seg_labeled``: train yolov8n-seg on a labeled COCO dataset.

    Mimics the student-training phase of distillation (convert COCO → YOLO-seg,
    ultralytics train) without teacher models or unlabeled images.
    """

    def run(
        self,
        job_config: Dict[str, Any],
        output_dir: str,
        progress_queue: mp.Queue,
        cancel_event: mp.Event,
    ) -> None:
        from core.constants import transform_image_path
        from ml_engine.data.inspection import inspect_dataset
        from ml_engine.jobs.handlers.distillation import StudentDistillationHandler

        data_path_raw = job_config.get("data_path")
        data_path = transform_image_path(data_path_raw) if data_path_raw else None
        image_paths = job_config.get("image_paths", [])

        if not data_path:
            raise ValueError("data_path required in job config")
        if not image_paths:
            raise ValueError("image_paths required in job config")

        with open(data_path, "r", encoding="utf-8") as f:
            labeled_coco = json.load(f)

        info = inspect_dataset(labeled_coco)
        if not info.get("has_masks"):
            raise ValueError(
                "yolov8n-seg requires COCO polygon/segmentation annotations; "
                "this dataset has no mask annotations."
            )

        distill_cfg: Dict[str, Any] = {
            "data_path": data_path_raw,
            "image_paths": image_paths,
            "student_model": "yolov8n-seg",
            "split_config": job_config.get("split_config", {"train": 0.8, "val": 0.2}),
            "training": job_config.get("training") or {},
        }

        logger.info(
            "Starting yolo_seg_labeled job: %d images, %d classes, output=%s",
            len(image_paths),
            info.get("num_classes", 0),
            output_dir,
        )

        handler = StudentDistillationHandler()
        handler.run(
            job_config=distill_cfg,
            output_dir=output_dir,
            progress_queue=progress_queue,
            cancel_event=cancel_event,
        )
