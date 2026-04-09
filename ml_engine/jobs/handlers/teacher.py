"""
Teacher training job handler.

Handles the teacher_training job type for fine-tuning GroundingDINO and SAM models.
"""

import logging
import multiprocessing as mp
import os
import queue
from pathlib import Path
from typing import Any, Dict, Optional

from ml_engine.jobs.handlers.base import JobHandler, TrainingCancelledError

logger = logging.getLogger(__name__)


class _NullProgressQueue:
    """Drop progress updates (e.g. CLI runs without a worker queue)."""

    def put_nowait(self, _item: Any) -> None:
        pass


def auto_student_distillation_enabled(job_config: Dict[str, Any]) -> bool:
    """True by default; set ``auto_student_distillation`` to false to skip chained student training."""
    v = job_config.get("auto_student_distillation", True)
    if v is None:
        return True
    return bool(v)


def run_chained_student_distillation_after_teacher(
    job_config: Dict[str, Any],
    data_path_raw: str,
    image_paths: list,
    output_dir: str,
    data_manager: Any,
    progress_queue: Optional[Any] = None,
    cancel_event: Optional[mp.Event] = None,
) -> None:
    """
    Run student YOLO training after teacher training when enabled (default on).

    Only runs if the dataset required SAM (masks). Default student model is ``yolov8n-seg``.
    Writes under ``{output_dir}/student_distillation/``.
    """
    from core.constants import SAM
    from ml_engine.jobs.handlers.distillation import StudentDistillationHandler

    required = data_manager.get_required_models()
    if SAM not in required:
        logger.warning(
            "Chained student distillation is enabled but SAM is not in required_models %s; "
            "skipping (train masks in COCO to enable SAM + YOLO-seg).",
            required,
        )
        return

    student_out = str(Path(output_dir) / "student_distillation")
    Path(student_out).mkdir(parents=True, exist_ok=True)

    distill_cfg: Dict[str, Any] = {
        "data_path": data_path_raw,
        "image_paths": image_paths,
        "teacher_dir": os.path.abspath(output_dir),
        "unlabeled_image_paths": job_config.get("unlabeled_image_paths") or [],
        "student_model": job_config.get("distillation_student_model", "yolov8n-seg"),
        "split_config": job_config.get(
            "distillation_split_config", {"train": 0.8, "val": 0.2}
        ),
        "training": job_config.get("distillation_training") or {},
    }

    pq = progress_queue if progress_queue is not None else _NullProgressQueue()
    ce = cancel_event if cancel_event is not None else mp.Event()

    try:
        pq.put_nowait(
            {
                "message": "Starting automatic student distillation (YOLO) after SAM training...",
                "phase": "student_distillation",
            }
        )
    except queue.Full:
        pass

    logger.info(
        "Chaining student distillation: output=%s student_model=%s teacher_dir=%s",
        student_out,
        distill_cfg["student_model"],
        distill_cfg["teacher_dir"],
    )

    handler = StudentDistillationHandler()
    handler.run(
        job_config=distill_cfg,
        output_dir=student_out,
        progress_queue=pq,
        cancel_event=ce,
    )


class TeacherTrainingHandler(JobHandler):
    """
    Handler for teacher model training jobs.
    
    Trains GroundingDINO and/or SAM models with LoRA adapters on custom datasets.
    """

    def run(
        self,
        job_config: Dict[str, Any],
        output_dir: str,
        progress_queue: mp.Queue,
        cancel_event: mp.Event,
    ) -> None:
        """
        Execute teacher training job.
        
        Args:
            job_config: Configuration containing:
                - data_path: Path to dataset
                - image_paths: List of image paths
                - split_config: Train/val/test split ratios (optional)
                - training: Training hyperparameter overrides (optional)
                - auto_student_distillation: After teacher training, run YOLO student training
                  when SAM is in the dataset (default: true; set false to skip). Default student:
                  yolov8n-seg; output in ``{output_dir}/student_distillation/``
                - distillation_student_model, distillation_split_config, distillation_training,
                  unlabeled_image_paths: Optional overrides forwarded to student distillation
            output_dir: Directory for checkpoints and logs
            progress_queue: Queue for progress updates
            cancel_event: Cancellation signal
        """
        # Late imports - these load in subprocess, not parent
        from ml_engine.training import Trainer, TrainingCancelledException
        from ml_engine.data.manager import DataManager
        from core.constants import transform_image_path

        # Extract paths from config
        data_path_raw = job_config.get("data_path")
        data_path = transform_image_path(data_path_raw) if data_path_raw else None
        image_paths = job_config.get("image_paths", [])

        if not data_path:
            raise ValueError("data_path required in job config")
        if not image_paths:
            raise ValueError("image_paths required in job config")

        # Create DataManager
        # Note: Normalization (bbox from masks, etc.) is always applied during loading
        split_config = job_config.get("split_config", {"train": 0.7, "val": 0.15, "test": 0.15})
        data_manager = DataManager.from_file(
            data_path=data_path,
            image_paths=image_paths,
            split_config=split_config
        )

        # Build config
        config = self._build_config(data_manager, job_config)

        # Progress callback that sends to queue
        def progress_callback(progress_info: Dict[str, Any]):
            try:
                progress_queue.put_nowait(progress_info)
            except queue.Full:
                pass  # Drop if queue is full

        # Cancel check that reads event
        def cancel_check() -> bool:
            return cancel_event.is_set()

        # Create and run trainer
        trainer = Trainer(
            data_manager=data_manager,
            output_dir=output_dir,
            config=config,
            progress_callback=progress_callback,
            cancel_check=cancel_check
        )

        try:
            trainer.train()
        except TrainingCancelledException as e:
            raise TrainingCancelledError("Training cancelled by user") from e

        if auto_student_distillation_enabled(job_config):
            run_chained_student_distillation_after_teacher(
                job_config=job_config,
                data_path_raw=data_path_raw,
                image_paths=image_paths,
                output_dir=output_dir,
                data_manager=data_manager,
                progress_queue=progress_queue,
                cancel_event=cancel_event,
            )

    def _build_config(
        self,
        data_manager,
        job_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build complete teacher training config from defaults + job overrides.
        
        Args:
            data_manager: DataManager instance with dataset info
            job_config: User-provided job configuration
            
        Returns:
            Complete training configuration dictionary
        """
        from core.config import load_config, merge_configs
        from core.constants import DEFAULT_CONFIGS_DIR

        logger = logging.getLogger(__name__)

        # Load shared training defaults
        shared_config_path = DEFAULT_CONFIGS_DIR / 'teacher_training.yaml'
        shared_config = load_config(str(shared_config_path))
        logger.info("Loaded shared training config from %s", shared_config_path)

        # Load model-specific configs based on dataset
        dataset_info = data_manager.get_dataset_info()
        required_models = data_manager.get_required_models()
        logger.info("Required teacher models: %s", required_models)

        model_configs = {}
        if 'grounding_dino' in required_models:
            dino_config_path = DEFAULT_CONFIGS_DIR / 'teacher_grounding_dino_lora.yaml'
            model_configs['grounding_dino'] = load_config(str(dino_config_path))
            logger.info("Loaded Grounding DINO config")

        if 'sam' in required_models:
            sam_config_path = DEFAULT_CONFIGS_DIR / 'teacher_sam_lora.yaml'
            model_configs['sam'] = load_config(str(sam_config_path))
            logger.info("Loaded SAM config")

        if not model_configs:
            raise ValueError("No models to train! Dataset has no valid annotations.")

        # Build base config
        config = {
            **shared_config['training'],
            'num_classes': dataset_info['num_classes'],
            'class_names': list(dataset_info['class_mapping'].values()),
            'class_mapping': dataset_info['class_mapping'],
            'augmentation': shared_config.get('augmentation'),
            'evaluation': shared_config.get('evaluation'),
            'checkpointing': shared_config.get('checkpointing'),
            'models': model_configs
        }

        # Merge user overrides
        user_overrides = job_config.get("training", {})
        if user_overrides:
            config = merge_configs(config, user_overrides)
            logger.info("Applied user config overrides")

        return config
