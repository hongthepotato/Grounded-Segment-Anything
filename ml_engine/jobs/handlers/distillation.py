"""
Student distillation job handler.

Orchestrates the full offline distillation pipeline:
1. Pseudo-label unlabeled images using fine-tuned teachers
2. Merge GT labels with pseudo-labels
3. Convert merged COCO to YOLO format
4. Train YOLOv8-seg student via ultralytics
"""

import json
import logging
import multiprocessing as mp
import os
import queue
from pathlib import Path
from typing import Dict, Any

from ml_engine.jobs.handlers.base import JobHandler, TrainingCancelledError

logger = logging.getLogger(__name__)


class StudentDistillationHandler(JobHandler):
    """
    Handler for student_distillation jobs.

    Trains a prompt-free YOLOv8-seg student model through offline
    knowledge distillation from fine-tuned teacher models.
    """

    def run(
        self,
        job_config: Dict[str, Any],
        output_dir: str,
        progress_queue: mp.Queue,
        cancel_event: mp.Event,
    ) -> None:
        from core.constants import transform_image_path, DEFAULT_CONFIGS_DIR
        from core.config import load_config, merge_configs
        from ml_engine.data.inspection import (
            detect_annotation_mode,
            get_recommended_student_model,
        )
        from ml_engine.distillation.pseudo_label import generate_pseudo_labels
        from ml_engine.distillation.student_trainer import StudentTrainer
        from ml_engine.distillation.utils import merge_coco_datasets, convert_coco_to_yolo_seg

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        def _report(msg: str, **kwargs):
            try:
                progress_queue.put_nowait({'message': msg, **kwargs})
            except queue.Full:
                pass

        def _cancel_check() -> bool:
            return cancel_event.is_set()

        
        logger.info("Loading distillation configuration...")
        distillation_cfg = load_config(str(DEFAULT_CONFIGS_DIR / 'distillation.yaml'))
        distillation_cfg = distillation_cfg.get('distillation', distillation_cfg)

        data_path_raw = job_config.get("data_path")
        data_path = transform_image_path(data_path_raw) if data_path_raw else None
        image_paths = job_config.get("image_paths", [])
        teacher_dir = job_config.get("teacher_dir")
        unlabeled_image_paths = job_config.get("unlabeled_image_paths", [])

        if not data_path:
            raise ValueError("data_path required in job config")
        if not image_paths:
            raise ValueError("image_paths required in job config")

        _report("Loading labeled dataset...")
        with open(data_path, 'r', encoding='utf-8') as f:
            labeled_coco = json.load(f)

        annotation_mode = detect_annotation_mode(labeled_coco)
        class_names = [c['name'] for c in labeled_coco.get('categories', [])]
        logger.info("Annotation mode: %s, classes: %s", annotation_mode, class_names)

        # --- Step 1: Pseudo-label unlabeled images ---
        training_coco = labeled_coco

        if teacher_dir and unlabeled_image_paths:
            _report("Generating pseudo-labels on unlabeled images...")

            logger.info("Found %d unlabeled images", len(unlabeled_image_paths))
            pseudo_path = str(out / 'pseudo_labels.json')

            def pseudo_progress(current, total, msg):
                _report(f"Pseudo-labeling: {msg}", current_step=current, total_steps=total)

            pseudo_coco = generate_pseudo_labels(
                image_paths=unlabeled_image_paths,
                class_names=class_names,
                teacher_dir=teacher_dir,
                output_path=pseudo_path,        # pseudo-labeled data will be saved to this path
                distillation_cfg=distillation_cfg,
                progress_callback=pseudo_progress,
            )

            _report("Merging GT + pseudo-labels...")
            training_coco = merge_coco_datasets(labeled_coco, pseudo_coco)

            merged_path = out / 'merged_labels.json'
            with open(merged_path, 'w', encoding='utf-8') as f:
                json.dump(training_coco, f)
            logger.info("Merged dataset saved to %s", merged_path)
        else:
            logger.info("No teacher_dir or unlabeled_image_paths; training on labeled data only")

        if _cancel_check():
            raise TrainingCancelledError("Cancelled before YOLO conversion")

        # --- Step 2: Convert to YOLO format ---
        _report("Converting to YOLO format...")

        yolo_dir = str(out / 'yolo_dataset')

        split_config = job_config.get("split_config", {'train': 0.8, 'val': 0.2})

        data_yaml = convert_coco_to_yolo_seg(
            coco_data=training_coco,
            output_dir=yolo_dir,
            split_ratios=split_config,
            class_names=class_names,
        )
        logger.info("YOLO dataset at %s", data_yaml)

        if _cancel_check():
            raise TrainingCancelledError("Cancelled before student training")

        # --- Step 3: Select student model ---
        student_model = job_config.get("student_model")
        if not student_model:
            size = job_config.get("student_size", "s")
            student_model = get_recommended_student_model(annotation_mode, size)
        logger.info("Student model: %s", student_model)

        # --- Step 4: Train ---
        _report("Starting student training...")


        user_overrides = job_config.get("training", {})
        if user_overrides:
            distillation_cfg = merge_configs(distillation_cfg, {'training': user_overrides})

        def training_progress(info):
            _report(info.get('message', ''), **info)

        trainer = StudentTrainer(
            data_yaml=data_yaml,
            model_name=student_model,
            config=distillation_cfg,
            output_dir=str(out),
        )

        best_pt = trainer.train(
            progress_callback=training_progress,
            cancel_check=_cancel_check,
        )

        final_dir = out / 'student_model'
        final_dir.mkdir(parents=True, exist_ok=True)
        final_weights = final_dir / 'best.pt'

        import shutil
        shutil.copy2(best_pt, str(final_weights))
        logger.info("Student model saved to %s", final_weights)

        _report("Student distillation complete!", best_pt=str(final_weights))

        # --- Step 5 (optional): Build + push ROS2 inference container ---
        if job_config.get("build_ros2_container", False):
            registry_url = job_config.get(
                "registry_url",
                os.environ.get("REGISTRY_PUSH_URL", "host-gateway:5000"),
            )
            job_id = job_config.get("job_id", "unknown")

            _report(
                "Building ROS2 container (this takes 5-90 min on first ARM64 build)...",
                current_step="ros2_build",
                total_steps=1,
            )
            try:
                from ml_engine.export.container_builder import build_ros2_container

                image_tag = build_ros2_container(
                    model_weights=final_weights,
                    job_id=job_id,
                    registry_url=registry_url,
                    cancel_event=cancel_event,
                    report_fn=lambda msg: _report(msg),
                )
                _report(
                    f"ROS2 container ready: {image_tag}",
                    ros2_image_tag=image_tag,
                )
            except Exception as ros2_err:
                # Build failure is WARNING-only: job still completes.
                # Operator can retry via POST /api/jobs/{job_id}/build-ros2
                logger.warning(
                    "ROS2 container build failed (job still COMPLETED): %s", ros2_err
                )
                _report(
                    "Container build failed — use POST /api/jobs/{job_id}/build-ros2 to retry.",
                    ros2_build_error=str(ros2_err),
                )
