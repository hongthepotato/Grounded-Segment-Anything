"""
Teacher training job handler.

Handles the teacher_training job type for fine-tuning GroundingDINO and SAM models.
"""

import dataclasses
import logging
import multiprocessing as mp
import queue
from typing import Dict, Any

from ml_engine.jobs.handlers.base import JobHandler, TrainingCancelledError
from ml_engine.training.config_types import TeacherTrainingConfig


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

    def _build_config(
        self,
        data_manager,
        job_config: Dict[str, Any]
    ) -> TeacherTrainingConfig:
        """
        Build typed teacher training config from defaults + job overrides.

        Args:
            data_manager: DataManager instance with dataset info
            job_config: User-provided job configuration

        Returns:
            TeacherTrainingConfig with typed fields for all training parameters
        """
        from core.config import load_config, merge_configs
        from core.constants import DEFAULT_CONFIGS_DIR
        from ml_engine.training.config_types import (
            ConfigurationError, GroundingDINOConfig, LoopConfig, LoraConfig,
            SAMConfig, TeacherTrainingConfig,
        )

        logger = logging.getLogger(__name__)

        shared_raw = load_config(str(DEFAULT_CONFIGS_DIR / 'teacher_training.yaml'))
        logger.info("Loaded shared training config")

        dataset_info = data_manager.get_dataset_info()
        required_models = data_manager.get_required_models()
        logger.info("Required teacher models: %s", required_models)

        user_overrides = job_config.get("training", {})
        if user_overrides:
            shared_raw = merge_configs(shared_raw, user_overrides)
            logger.info("Applied user config overrides")

        # Build immutable loop config from shared training section
        # Unknown YAML keys are silently discarded — intentional to allow YAML evolution
        try:
            loop_field_names = {f.name for f in dataclasses.fields(LoopConfig)}
            loop = LoopConfig(**{k: v for k, v in shared_raw['training'].items()
                                 if k in loop_field_names})
        except (TypeError, KeyError) as e:
            raise ConfigurationError(
                f"Failed to build LoopConfig from teacher_training.yaml: {e}"
            ) from e

        lora_field_names = {f.name for f in dataclasses.fields(LoraConfig)}
        models = {}

        if 'grounding_dino' in required_models:
            raw = load_config(str(DEFAULT_CONFIGS_DIR / 'teacher_grounding_dino_lora.yaml'))
            if 'models' in user_overrides and 'grounding_dino' in user_overrides['models']:
                raw = merge_configs(raw, user_overrides['models']['grounding_dino'])
            try:
                models['grounding_dino'] = GroundingDINOConfig(
                    lora=LoraConfig(**{k: v for k, v in raw['lora'].items()
                                       if k in lora_field_names}),
                    learning_rate=raw['learning_rate'],
                    base_checkpoint=raw['model']['base_checkpoint'],
                    config_path=raw['model']['config_path'],
                    freeze_backbone=raw.get('freeze_backbone', True),
                    freeze_bbox_embed=raw.get('freeze_bbox_embed', False),
                    bert_model_path=raw.get('bert_model_path'),
                    momentum=raw.get('momentum', 0.9),
                    evaluation_metric=raw.get('evaluation', {}).get('metric', 'mAP50'),
                )
                logger.info("Loaded Grounding DINO config")
            except (TypeError, KeyError) as e:
                raise ConfigurationError(
                    f"Failed to build GroundingDINOConfig from teacher_grounding_dino_lora.yaml: {e}"
                ) from e

        if 'sam' in required_models:
            raw = load_config(str(DEFAULT_CONFIGS_DIR / 'teacher_sam_lora.yaml'))
            if 'models' in user_overrides and 'sam' in user_overrides['models']:
                raw = merge_configs(raw, user_overrides['models']['sam'])
            try:
                models['sam'] = SAMConfig(
                    lora=LoraConfig(**{k: v for k, v in raw['lora'].items()
                                       if k in lora_field_names}),
                    learning_rate=raw['learning_rate'],
                    base_checkpoint=raw['model']['base_checkpoint'],
                    model_type=raw['model'].get('model_type', 'vit_b'),
                    image_encoder_mode=raw.get('image_encoder_mode', 'lora'),
                    prompt_encoder_mode=raw.get('prompt_encoder_mode', 'frozen'),
                    mask_decoder_mode=raw.get('mask_decoder_mode', 'full'),
                    mask_decoder_lr_multiplier=raw.get('mask_decoder_lr_multiplier', 0.1),
                    prompt_type=raw.get('prompt_type', 'boxes'),
                    multimask_output=raw.get('multimask_output', True),
                    single_object_sampling=raw.get('training', {}).get('single_object_sampling', True),
                    evaluation_metric=raw.get('evaluation', {}).get('metric', 'mask_IoU'),
                )
                logger.info("Loaded SAM config")
            except (TypeError, KeyError) as e:
                raise ConfigurationError(
                    f"Failed to build SAMConfig from teacher_sam_lora.yaml: {e}"
                ) from e

        if not models:
            raise ValueError("No models to train — dataset has no valid annotations.")

        return TeacherTrainingConfig(
            loop=loop,
            num_classes=dataset_info['num_classes'],
            class_names=list(dataset_info['class_mapping'].values()),
            class_mapping=dataset_info['class_mapping'],
            models=models,
            augmentation=shared_raw.get('augmentation'),
            evaluation=shared_raw.get('evaluation'),
            checkpointing=shared_raw.get('checkpointing'),
        )
