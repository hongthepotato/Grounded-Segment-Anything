"""
Auto-labeling job handler.

Handles the auto_label job type for automatic annotation using GroundingDINO + SAM.
"""

import logging
import multiprocessing as mp
import queue
from pathlib import Path
from typing import Dict, Any

from ml_engine.jobs.handlers.base import JobHandler, TrainingCancelledError


class AutoLabelHandler(JobHandler):
    """
    Handler for auto-labeling jobs.
    
    Uses GroundingDINO for detection and MobileSAM for segmentation
    to automatically generate annotations for images.
    """

    def run(
        self,
        job_config: Dict[str, Any],
        output_dir: str,
        progress_queue: mp.Queue,
        cancel_event: mp.Event,
    ) -> None:
        """
        Execute auto-labeling job.
        
        Args:
            job_config: Configuration containing:
                - image_paths: List of image paths to process
                - classes: List of class names to detect
                - output_mode: "boxes", "masks", or "both"
                - box_threshold: Detection confidence threshold (default: 0.5)
                - text_threshold: Text threshold (default: 0.5)
                - nms_threshold: NMS threshold (default: 0.7)
            output_dir: Directory for annotation outputs
            progress_queue: Queue for progress updates
            cancel_event: Cancellation signal
        """
        # Late imports - these load in subprocess, not parent
        from ml_engine.inference import (
            AutoLabeler,
            AutoLabelerConfig,
            COCOExporter,
            visualize_detections,
        )
        from core.config import save_json
        from core.constants import transform_image_path

        sub_logger = logging.getLogger(__name__)

        # Extract and validate config
        raw_image_paths = job_config.get("image_paths", [])
        if not raw_image_paths:
            raise ValueError("image_paths required in job config")

        classes = job_config.get("classes", [])
        if not classes:
            raise ValueError("classes required in job config")

        # Transform paths
        image_paths = []
        for raw_path in raw_image_paths:
            actual_path = transform_image_path(raw_path)
            if not Path(actual_path).exists():
                raise ValueError(f"Image path not found: {raw_path} -> {actual_path}")
            image_paths.append(actual_path)

        sub_logger.info("Transformed %d image paths", len(image_paths))

        # Optional config with defaults
        output_mode = job_config.get("output_mode", "boxes")
        box_threshold = job_config.get("box_threshold", 0.5)
        text_threshold = job_config.get("text_threshold", 0.5)
        nms_threshold = job_config.get("nms_threshold", 0.7)

        # Setup output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        viz_dir = output_path / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        sub_logger.info("Auto-labeling %d images with classes: %s",
                       len(image_paths), classes)

        # Create AutoLabeler config
        labeler_config = AutoLabelerConfig(
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            nms_threshold=nms_threshold,
            output_mode=output_mode,
            device="cuda"
        )

        # Create labeler
        labeler = AutoLabeler(labeler_config)

        # Track progress
        show_boxes = output_mode in ("boxes", "both")
        show_masks = output_mode in ("masks", "both")
        total_images = len(image_paths)
        annotation_count = 0

        # Progress callback for AutoLabeler
        def on_progress(current: int, total: int, message: str):
            nonlocal annotation_count

            # Check for cancellation
            if cancel_event.is_set():
                raise TrainingCancelledError("Auto-labeling cancelled by user")
            
            try:
                progress_queue.put_nowait({
                    "current_step": current,
                    "total_steps": total,
                    "current_epoch": 0,
                    "total_epochs": 1,
                    "message": message,
                    "metrics": {
                        "images_processed": current,
                        "annotations_found": annotation_count,
                    }
                })
            except queue.Full:
                pass

        try:
            # Sequential processing
            results = labeler.label_images(
                image_paths=image_paths,
                class_prompts=classes,
                progress_callback=on_progress
            )
        except TrainingCancelledError:
            raise
        except Exception as e:
            sub_logger.error("Auto-labeling failed: %s", e)
            raise

        # Count annotations
        for result in results:
            annotation_count += len(result.get('class_ids', []))

        # Generate visualizations
        for image_path, result in zip(image_paths, results):
            try:
                viz_filename = Path(image_path).stem + "_viz.jpg"
                viz_path = str(viz_dir / viz_filename)
                visualize_detections(
                    image_path=image_path,
                    result=result,
                    class_prompts=classes,
                    output_path=viz_path,
                    show_boxes=show_boxes,
                    show_masks=show_masks
                )
            except Exception as viz_e:
                sub_logger.warning("Failed to visualize %s: %s", image_path, viz_e)

        # Use COCOExporter (single source of truth)
        coco_output = COCOExporter.export(results, classes, output_mode)

        # Save annotations
        annotations_path = output_path / "annotations.json"
        save_json(coco_output, str(annotations_path))

        sub_logger.info("Auto-labeling complete: %d images, %d annotations",
                       len(coco_output['images']), len(coco_output['annotations']))
        sub_logger.info("Results saved to: %s", output_dir)
