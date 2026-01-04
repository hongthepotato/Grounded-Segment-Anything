"""
Teacher training job handler.

Handles the teacher_training job type for fine-tuning GroundingDINO and SAM models.
"""

import logging
import multiprocessing as mp
import queue
from typing import Dict, Any

from ml_engine.jobs.handlers.base import JobHandler, TrainingCancelledError


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
            output_dir: Directory for checkpoints and logs
            progress_queue: Queue for progress updates
            cancel_event: Cancellation signal
        """
        # Late imports - these load in subprocess, not parent
        from ml_engine.training.teacher_trainer import TeacherTrainer, TrainingCancelledException
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
        data_manager = DataManager(
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
        trainer = TeacherTrainer(
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
