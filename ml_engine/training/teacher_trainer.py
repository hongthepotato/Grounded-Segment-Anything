"""
Teacher Model Trainer with Data-Driven Model Loading.

This module provides the main training orchestrator for fine-tuning
teacher models (Grounding DINO and/or SAM) with LoRA.

Key features:
- Data-driven: Automatically loads models based on dataset annotations
- LoRA integration: Memory-efficient fine-tuning
- Multi-model support: Can train DINO, SAM, or both
- Automatic config generation
"""

import logging
from typing import Dict, Optional, Any, Callable
from pathlib import Path
import torch
from tqdm import tqdm
from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span

from ml_engine.data.loaders import create_dataloader
from ml_engine.data.manager import DataManager
from ml_engine.data.dataset_factory import DatasetFactory
from ml_engine.evaluation import PredictionVisualizer
from ml_engine.evaluation.evaluator import ModelEvaluator
from ml_engine.evaluation.report import ModelReportGenerator
from ml_engine.export import create_export_package
from ml_engine.models.teacher.grounding_dino_lora import load_grounding_dino_with_lora
from ml_engine.models.teacher.sam_lora import load_sam_hq_with_lora
from ml_engine.training.losses import build_criterion, SegmentationLoss
from ml_engine.training.training_manager import TrainingManager
from ml_engine.training.checkpoint_manager import CheckpointManager
from ml_engine.training.memory_monitor import GPUMemoryMonitor, log_memory_snapshot
from core.logger import TensorBoardLogger, log_config, log_metrics
from core.constants import DEFAULT_CONFIGS_DIR


class TrainingCancelledException(Exception):
    """Raised when training is cancelled by user request."""
    pass

# Import official Grounding DINO utilities for proper token mapping
# import sys
# sys.path.insert(0, str(Path(__file__).parent.parent.parent / "GroundingDINO"))

logger = logging.getLogger(__name__)


class TeacherTrainer:
    """
    Main trainer for teacher models with data-driven model loading.
    
    This class:
    - Inspects dataset to determine which models to load
    - Loads only required models (DINO if has_boxes, SAM if has_masks)
    - Applies LoRA for memory-efficient training
    - Handles training loop with proper gradient management
    - Saves LoRA adapters (not full models)
    
    Example:
        >>> manager = DataManager(
        >>>     data_path='data/raw/annotations.json',
        >>>     image_paths=['/profile/upload/2025/12/16/xxx.jpeg', ...]
        >>> )
        >>> trainer = TeacherTrainer(
        >>>     data_manager=manager,
        >>>     output_dir='experiments/exp1',
        >>>     config=config
        >>> )
        >>> trainer.train()
    """

    def __init__(
        self,
        data_manager: DataManager,
        output_dir: str,
        config: Dict[str, Any],
        resume_from: Optional[str] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ):
        """
        Initialize TeacherTrainer with DataManager.
        
        Args:
            data_manager: DataManager instance with train/val splits
            output_dir: Output directory for checkpoints and logs
            config: Training configuration
            resume_from: Optional checkpoint path to resume from
            progress_callback: Optional callback function called after each epoch
                              with progress dict containing epoch, metrics, etc.
            cancel_check: Optional function that returns True if training should
                         be cancelled. Checked at start of each batch.
        
        Design philosophy:
        - User provides ONE dataset file
        - Platform handles splitting (train/val/test)
        - Trainer uses 'train' and 'val' splits from the manager
        
        Example:
            >>> # User provides single annotations.json with image paths
            >>> manager = DataManager(
            >>>     data_path='data/raw/annotations.json',
            >>>     image_paths=['/profile/upload/2025/12/16/xxx.jpeg', ...],
            >>>     split_config={'train': 0.7, 'val': 0.2, 'test': 0.1}
            >>> )
            >>> 
            >>> # Trainer automatically uses 'train' and 'val' splits
            >>> trainer = TeacherTrainer(
            >>>     data_manager=manager,
            >>>     output_dir='experiments/exp1',
            >>>     config=config
            >>> )
            >>> 
            >>> trainer.train()
        """
        # Store DataManager (single source of truth)
        self.data_manager = data_manager
        self.output_dir = Path(output_dir)
        self.config = config
        self.resume_from = resume_from

        # Job manager integration - callbacks for progress reporting and cancellation
        self.progress_callback = progress_callback
        self.cancel_check = cancel_check

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'teachers').mkdir(exist_ok=True)

        # Setup device
        # CRITICAL: When CUDA_VISIBLE_DEVICES is set (by subprocess_runner),
        # PyTorch sees only the visible GPUs and remaps them starting from 0.
        # Example: CUDA_VISIBLE_DEVICES=3 → PyTorch cuda:0 = Physical GPU 3
        # So we always use cuda:0 here, which maps to the correct physical GPU.
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logger.info("Using device: %s", self.device)
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info("GPU: %s", gpu_name)

        self.dataset_info = data_manager.get_dataset_info()

        logger.info("Dataset has boxes: %s", self.dataset_info['has_boxes'])
        logger.info("Dataset has masks: %s", self.dataset_info['has_masks'])
        logger.info("Num classes: %s", self.dataset_info['num_classes'])

        self.required_models = data_manager.get_required_models()
        logger.info("Required teacher models: %s", self.required_models)

        # Initialize components
        self._init_datasets()
        self._init_models()
        self._init_losses()
        self._init_optimizers()
        self._init_training_manager()
        self._init_checkpoint_manager()
        self._init_loggers()
        self._init_visualizer()
        
        # Initialize memory monitor for leak detection
        self.memory_monitor = GPUMemoryMonitor(device=self.device)
        logger.info("✓ GPU memory monitor initialized")

        # Resume from checkpoint if provided
        if resume_from:
            self._resume_from_checkpoint(resume_from)

    def _init_datasets(self):
        """
        Initialize datasets and dataloaders using DatasetFactory.

        DatasetFactory handles:
        - Preprocessor creation (based on required_models)
        - Augmentation pipeline creation (based on config)
        - TeacherDataset instantiation
        """
        # Get data and metadata from DataManager
        train_data = self.data_manager.get_split('train')
        val_data = self.data_manager.get_split('val')

        # Create training dataset with augmentation
        self.train_dataset = DatasetFactory.create_dataset(
            coco_data=train_data,
            image_path_resolver=self.data_manager.get_image_path,
            dataset_info=self.dataset_info,
            model_names=self.required_models,
            augmentation_config=self.config['augmentation'],
            is_training=True
        )

        # Create validation dataset without augmentation
        self.val_dataset = DatasetFactory.create_dataset(
            coco_data=val_data,
            image_path_resolver=self.data_manager.get_image_path,
            dataset_info=self.dataset_info,
            model_names=self.required_models,
            augmentation_config=None,  # No augmentation for validation
            is_training=False
        )

        # Create dataloaders
        batch_size = self.config.get('batch_size', 8)
        num_workers = self.config.get('num_workers', 4)

        self.train_loader = create_dataloader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        self.val_loader = create_dataloader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        logger.info("✓ Datasets initialized: %d train, %d val",
                    len(self.train_dataset), len(self.val_dataset))

    def _init_models(self):
        """Initialize teacher models based on dataset."""
        self.models = {}

        # Validate models config exists
        if 'models' not in self.config:
            raise ValueError(
                "Config missing 'models' section!\n"
                "This indicates a bug in config generation."
            )

        # Load Grounding DINO if dataset has boxes
        if 'grounding_dino' in self.required_models:
            model_config = self.config['models']['grounding_dino']

            logger.info("Loading Grounding DINO with LoRA...")

            # Checkpoint path is optional (sensible default exists)
            base_ckpt = model_config.get('model', {}).get(
                'base_checkpoint',
                'data/models/pretrained/groundingdino_swint_ogc.pth'
            )

            # LoRA config is REQUIRED for LoRA training
            if 'lora' not in model_config:
                raise ValueError(
                    "LoRA training requires 'models.grounding_dino.lora' config!\n"
                    "Expected keys: r, lora_alpha, target_modules, lora_dropout"
                )

            self.models['grounding_dino'] = load_grounding_dino_with_lora(
                base_checkpoint=base_ckpt,
                lora_config=model_config['lora'],
                freeze_backbone=model_config.get('freeze_backbone', True),
                freeze_bbox_embed=model_config.get('freeze_bbox_embed', False),
                bert_model_path=model_config.get('bert_model_path', None)
            ).to(self.device)

            logger.info(" Grounding DINO loaded")

        # Load SAM if dataset has masks
        if 'sam' in self.required_models:
            sam_config = self.config['models']['sam']

            logger.info("Loading SAM-HQ with LoRA...")

            model_section = sam_config.get('model', {})
            base_ckpt = model_section.get(
                'base_checkpoint',
                'data/models/pretrained/sam_vit_h_4b8939.pth'
            )
            model_type = model_section.get('model_type', 'vit_h')

            if 'lora' not in sam_config:
                raise ValueError(
                    "LoRA training requires 'models.sam.lora' config!\n"
                    "Expected keys: r, lora_alpha, target_modules, lora_dropout"
                )

            image_encoder_mode = sam_config.get('image_encoder_mode', 'lora')
            prompt_encoder_mode = sam_config.get('prompt_encoder_mode', 'frozen')
            mask_decoder_mode = sam_config.get('mask_decoder_mode', 'full')

            self.models['sam'] = load_sam_hq_with_lora(
                base_checkpoint=base_ckpt,
                model_type=model_type,
                lora_config=sam_config['lora'],
                image_encoder_mode=image_encoder_mode,
                prompt_encoder_mode=prompt_encoder_mode,
                mask_decoder_mode=mask_decoder_mode
            ).to(self.device)

            logger.info("SAM-HQ loaded (modes: encoder=%s, prompt=%s, decoder=%s)",
                       image_encoder_mode, prompt_encoder_mode, mask_decoder_mode)

        # Set models to training mode
        for model in self.models.values():
            model.train()

    def _init_losses(self):
        """Initialize loss functions based on loaded models."""
        self.losses = {}

        if 'grounding_dino' in self.models:
            # Use proper DETR-style criterion with Hungarian matching
            num_classes = self.dataset_info['num_classes']

            # Query num_decoder_layers from actual model architecture
            dino_model = self.models['grounding_dino']
            base_model = dino_model.model.model  # Unwrap PEFT wrapper
            num_decoder_layers = base_model.transformer.decoder.num_layers

            # Build criterion with auxiliary losses
            self.losses['detection'] = build_criterion(
                num_classes=num_classes,
                num_decoder_layers=num_decoder_layers,
                focal_alpha=0.25,
                focal_gamma=2.0
            )

            logger.info("✓ Grounding DINO criterion with Hungarian matching initialized")
            logger.info("  - Num classes: %d", num_classes)
            logger.info("  - Num decoder layers: %d (architectural constant)", num_decoder_layers)
            logger.info("  - Auxiliary losses: %d intermediate + 1 encoder", num_decoder_layers - 1)

        if 'sam' in self.models:
            self.losses['segmentation'] = SegmentationLoss().to(self.device)

    def _init_optimizers(self):
        """Initialize optimizers for each model."""
        self.optimizers = {}
        self.schedulers = {}

        # Shared training params (from flattened shared_config['training'])
        weight_decay = self.config.get('weight_decay', 1e-4)
        optimizer_type = self.config.get('optimizer', 'AdamW')
        warmup_epochs = self.config.get('warmup_epochs', 3)
        total_epochs = self.config.get('epochs', 50)

        for model_name, model in self.models.items():
            model_config = self.config['models'][model_name]
            lr = model_config.get('learning_rate', 1e-4)
            # Get trainable parameters
            trainable_params = [p for p in model.parameters() if p.requires_grad]

            # ===================================================================
            # Following prints are for debugging purposes
            # ===================================================================
            print("#" * 60)
            num_trainable = sum(p.numel() for p in trainable_params)
            print(f"Model: {model_name}")
            print(f"Trainable parameters: {num_trainable:,} ({num_trainable / 1e6:.2f}M)")
            print(f"Number of trainable tensors: {len(trainable_params)}")
            print("#" * 60)

            # Create optimizer
            if optimizer_type == 'AdamW':
                optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=lr,
                    weight_decay=weight_decay
                )
            elif optimizer_type == 'SGD':
                # Momentum is MODEL-SPECIFIC
                momentum = model_config.get('momentum', 0.9)
                optimizer = torch.optim.SGD(
                    trainable_params,
                    lr=lr,
                    momentum=momentum,
                    weight_decay=weight_decay
                )
            else:
                raise ValueError(f"Unknown optimizer: {optimizer_type}")

            self.optimizers[model_name] = optimizer

            # Create scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=max(1, total_epochs - warmup_epochs),
                T_mult=1
            )
            self.schedulers[model_name] = scheduler

            logger.info("Optimizer for %s: %s (lr=%.2e)", model_name, optimizer_type, lr)

    def _init_training_manager(self):
        """Initialize training managers for each model."""
        self.training_managers = {}

        for model_name, model in self.models.items():
            config_path = DEFAULT_CONFIGS_DIR / 'training_dynamics.yaml'

            manager = TrainingManager(
                model=model,
                optimizer=self.optimizers[model_name],
                config_path=str(config_path),
                scheduler=self.schedulers.get(model_name)
            )
            self.training_managers[model_name] = manager

            logger.info("✓ TrainingManager created for %s", model_name)

    def _init_checkpoint_manager(self):
        """Initialize checkpoint managers for each model."""
        self.checkpoint_managers = {}

        for model_name in self.models:
            output_dir = self.output_dir / 'teachers' / f'{model_name}_lora'
            config_path = DEFAULT_CONFIGS_DIR / 'checkpoint_config.yaml'

            # Dynamic monitor metric based on model name
            monitor_metric = f"val_{model_name}_total_loss"

            manager = CheckpointManager(
                output_dir=str(output_dir),
                config_path=str(config_path),
                monitor_metric=monitor_metric,
                mode='min'  # Loss should be minimized
            )
            self.checkpoint_managers[model_name] = manager

            logger.info("✓ CheckpointManager created for %s: %s", model_name, output_dir)
            logger.info("  Monitoring: %s (mode=min)", monitor_metric)

    def _init_loggers(self):
        """Initialize TensorBoard loggers."""
        self.tb_loggers = {}

        for model_name in self.models:
            log_dir = self.output_dir / 'logs' / model_name
            self.tb_loggers[model_name] = TensorBoardLogger(str(log_dir))
            logger.info("✓ TensorBoard logger: %s", log_dir)

    def _init_visualizer(self):
        """Initialize prediction visualizer for debugging."""
        save_predictions = self.config.get('evaluation', {}).get('save_predictions', False)

        if save_predictions:
            self.visualizer = PredictionVisualizer(
                output_dir=str(self.output_dir / 'predictions'),
                max_samples_per_epoch=8,
                enabled=True
            )
            logger.info("✓ Prediction visualizer enabled: %s", self.output_dir / 'predictions')
        else:
            self.visualizer = None

    def train(self):
        """
        Main training loop.
        
        Trains all loaded models (data-driven).
        
        Raises:
            TrainingCancelledException: If cancel_check returns True
        """
        epochs = self.config.get('epochs', 50)
        start_epoch = 0

        logger.info("=" * 60)
        logger.info("Starting Teacher Model Training")
        logger.info("=" * 60)
        log_config(logger, self.config, "Training Configuration")

        try:
            for epoch in range(start_epoch, epochs):
                # Check for cancellation at epoch start
                if self.cancel_check and self.cancel_check():
                    logger.info("Training cancelled by user request")
                    raise TrainingCancelledException("Training cancelled by user")

                logger.info("\nEpoch %s/%s", epoch + 1, epochs)
                logger.info("-" * 60)

                train_metrics = self.train_epoch(epoch)

                # Only validate at specified interval
                eval_interval = self.config.get('evaluation', {}).get('interval', 1)
                if (epoch + 1) % eval_interval == 0:
                    val_metrics = self.validate(epoch)
                else:
                    val_metrics = {}

                all_metrics = {**train_metrics, **val_metrics}

                for model_name, model in self.models.items():
                    self.checkpoint_managers[model_name].save_checkpoint(
                        epoch=epoch,
                        model=model,
                        optimizer=self.optimizers[model_name],
                        metrics=all_metrics,
                        scheduler=self.schedulers.get(model_name),
                        scaler=self.training_managers[model_name].scaler,
                        extra_info={'config': self.config}
                    )
                    # Check early stopping
                    if self.checkpoint_managers[model_name].should_stop:
                        logger.info("Early stopping triggered for %s", model_name)
                        break

                # Report progress after each epoch (outside inner for loop)
                if self.progress_callback:
                    progress_info = {
                        'current_epoch': epoch + 1,
                        'total_epochs': epochs,
                        'train_metrics': train_metrics,
                        'val_metrics': val_metrics,
                        'message': f"Completed epoch {epoch + 1}/{epochs}"
                    }
                    self.progress_callback(progress_info)

            logger.info("=" * 60)
            logger.info("Training Completed!")
            logger.info("=" * 60)

            # Save lightweight LoRA adapters for deployment
            self._save_lora_adapters_only()

            # Evaluate on test set and generate report
            self._evaluate_on_test_set()

            # Create downloadable export package
            self._create_export_package()

        finally:
            # Always close loggers
            for tb_logger in self.tb_loggers.values():
                tb_logger.close()

    def _save_lora_adapters_only(self):
        """
        Save only LoRA adapters for deployment (NOT full model).
        
        This produces ~5-10MB files instead of ~2GB checkpoints.
        Use these for inference; use full checkpoints only for resuming training.
        """
        for model_name, model in self.models.items():
            if hasattr(model, 'save_lora_adapters'):
                adapter_dir = self.output_dir / 'teachers' / f'{model_name}_lora_adapters'
                model.save_lora_adapters(str(adapter_dir))
                logger.info("✓ Saved LoRA adapters (~5MB) to: %s", adapter_dir)
            else:
                logger.warning("Model %s does not support save_lora_adapters", model_name)

    def _create_export_package(self):
        """
        Create downloadable export package with merged model and inference scripts.
        
        This creates a ZIP file in output_dir/exports/ containing:
        - merged_model.pth: Merged model weights (base + LoRA)
        - inference.py: Ready-to-run inference script
        - README.md: Usage instructions
        - requirements.txt: Python dependencies
        - class_names.txt: Classes the model was trained on
        """
        logger.info("=" * 60)
        logger.info("Creating Export Package")
        logger.info("=" * 60)

        class_names = list(self.dataset_info['class_mapping'].values())

        for model_name, model in self.models.items():
            if model_name != 'grounding_dino':
                logger.info("Skipping export for %s (only Grounding DINO supported)", model_name)
                continue

            try:
                # Gather training info for README
                model_config = self.config['models'][model_name]
                training_info = {
                    'epochs': self.config.get('epochs', 'N/A'),
                    'batch_size': self.config.get('batch_size', 'N/A'),
                    'learning_rate': model_config.get('learning_rate', 'N/A'),
                }

                # Try to get mAP from evaluation results
                eval_report_path = self.output_dir / 'evaluation' / f'{model_name}_report.json'
                if eval_report_path.exists():
                    import json
                    with open(eval_report_path) as f:
                        report = json.load(f)
                        training_info['mAP50'] = report.get('technical_metrics', {}).get('mAP50', 0)

                # Create export package
                zip_path = create_export_package(
                    model=model,
                    output_dir=self.output_dir,
                    class_names=class_names,
                    model_name=model_name,
                    training_info=training_info
                )

                logger.info("✓ Export package created: %s", zip_path)

            except Exception as e:
                logger.error("Failed to create export package for %s: %s", model_name, e)
                logger.exception(e)

        logger.info("=" * 60)
        logger.info("Export Package Creation Completed!")
        logger.info("=" * 60)

    def _evaluate_on_test_set(self):
        """
        Evaluate trained models on held-out test set.
        
        This runs after training completes and generates:
        - Technical metrics (mAP, IoU, etc.)
        - Simple metrics (overall score, detection rate)
        - Evaluation report with recommendations
        """
        # Check if test split exists
        try:
            test_data = self.data_manager.get_split('test')
        except ValueError:
            logger.warning("No test split available. Skipping test evaluation.")
            logger.warning("To enable test evaluation, provide split_config with 'test' ratio.")
            return

        if len(test_data.get('images', [])) == 0:
            logger.warning("Test split is empty. Skipping test evaluation.")
            return

        logger.info("=" * 60)
        logger.info("Running Test Set Evaluation")
        logger.info("=" * 60)

        # Create test dataset and dataloader
        test_dataset = DatasetFactory.create_dataset(
            coco_data=test_data,
            image_path_resolver=self.data_manager.get_image_path,
            dataset_info=self.dataset_info,
            model_names=self.required_models,
            augmentation_config=None,  # No augmentation for test
            is_training=False
        )

        batch_size = self.config.get('batch_size', 8)
        num_workers = self.config.get('num_workers', 4)

        test_loader = create_dataloader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        logger.info("Test set: %d images", len(test_dataset))

        # Initialize evaluator and report generator
        evaluator = ModelEvaluator(
            device=str(self.device),
            confidence_threshold=self.config.get('evaluation', {}).get('confidence_threshold', 0.3)
        )
        report_generator = ModelReportGenerator()

        # Get class names
        class_names = list(self.dataset_info['class_mapping'].values())

        # Evaluate each model
        all_reports = []

        for model_name, model in self.models.items():
            logger.info("Evaluating %s...", model_name)

            # Load best checkpoint before evaluation
            best_ckpt_path = self.checkpoint_managers[model_name].get_best_checkpoint_path()
            if best_ckpt_path.exists():
                logger.info("Loading best checkpoint: %s", best_ckpt_path)
                self.checkpoint_managers[model_name].load_checkpoint(
                    str(best_ckpt_path),
                    model=model,
                    load_optimizer=False,
                    load_rng_state=False  # Don't need RNG for evaluation
                )

            # Run evaluation based on model type
            if model_name == 'grounding_dino':
                eval_results = evaluator.evaluate_detection(
                    model=model,
                    dataloader=test_loader,
                    class_names=class_names,
                    dataset_info=self.dataset_info
                )
            elif model_name == 'sam':
                eval_results = evaluator.evaluate_segmentation(
                    model=model,
                    dataloader=test_loader,
                    class_names=class_names,
                    dataset_info=self.dataset_info
                )
            else:
                logger.warning("Unknown model type %s, skipping evaluation", model_name)
                continue

            # Generate report
            model_config = self.config['models'][model_name]
            report = report_generator.generate_report(
                evaluation_results=eval_results,
                model_name=model_name,
                test_set_size=len(test_dataset),
                extra_info={
                    'config': {
                        'epochs': self.config.get('epochs'),
                        'batch_size': self.config.get('batch_size'),
                        'learning_rate': model_config.get('learning_rate')
                    }
                }
            )

            # Save individual report
            report_path = self.output_dir / 'evaluation' / f'{model_name}_report.json'
            report_generator.save_report(report, str(report_path))

            # Print summary
            summary_text = report_generator.generate_summary_text(report)
            logger.info("\n%s", summary_text)

            all_reports.append(report)

        # Save combined report if multiple models
        if len(all_reports) > 1:
            combined_report = report_generator.combine_reports(all_reports)
            combined_path = self.output_dir / 'evaluation' / 'combined_report.json'
            report_generator.save_report(combined_report, str(combined_path))

        logger.info("=" * 60)
        logger.info("Test Evaluation Completed!")
        logger.info("Reports saved to: %s", self.output_dir / 'evaluation')
        logger.info("=" * 60)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dictionary of training metrics
            
        Raises:
            TrainingCancelledException: If cancel_check returns True
        """
        # Set models to train mode
        for model in self.models.values():
            model.train()

        # Update epoch in training managers
        for manager in self.training_managers.values():
            manager.set_epoch(epoch)

        # Metrics accumulator (dynamic keys)
        epoch_losses = {}
        total_steps = len(self.train_loader)
        
        # Memory monitoring
        self.memory_monitor.record('epoch_start')
        log_memory_snapshot(f"[Epoch {epoch+1}] Start - ")

        # Training loop
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch + 1}")
        for step, batch in enumerate(pbar):
            # Check for cancellation at each batch
            if self.cancel_check and self.cancel_check():
                logger.info("Training cancelled during epoch %d, step %d", epoch + 1, step)
                raise TrainingCancelledException("Training cancelled by user")
            
            # Memory check before batch
            if step % 50 == 0:  # Check every 50 steps
                self.memory_monitor.record('batch_start')

            batch_losses = self._train_batch(batch)
            
            # Memory check after batch
            if step % 50 == 0:
                self.memory_monitor.record('batch_end')
                
                # Check for leaks every 100 steps
                if step > 0 and step % 100 == 0:
                    if self.memory_monitor.check_leak('batch_end', threshold_gb=1.0):
                        logger.warning("Memory leak detected! Forcing garbage collection...")
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()

            # Accumulate losses (dynamically add new keys)
            for key, value in batch_losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(value)

            # Update progress bar - show main loss and components
            postfix = {}
            for key, values in epoch_losses.items():
                if values:
                    # Show only main losses and key components in progress bar
                    if 'total_loss' in key or key in ['grounding_dino_loss_ce', 'grounding_dino_loss_bbox', 'grounding_dino_loss_giou']:
                        postfix[key] = f"{values[-1]:.4f}"
            pbar.set_postfix(postfix)

            # Report step progress periodically (every 10% of epoch)
            if self.progress_callback and total_steps > 0:
                report_interval = max(1, total_steps // 10)
                if step % report_interval == 0:
                    avg_loss = sum(epoch_losses.get('grounding_dino_total_loss', [0])) / max(1, len(epoch_losses.get('grounding_dino_total_loss', [0])))
                    self.progress_callback({
                        'current_epoch': epoch,
                        'total_epochs': self.config.get('epochs', 50),
                        'current_step': step + 1,
                        'total_steps': total_steps,
                        'metrics': {'avg_loss': avg_loss},
                        'message': f"Epoch {epoch + 1}, Step {step + 1}/{total_steps}"
                    })

        # Memory monitoring at epoch end
        self.memory_monitor.record('epoch_end')
        log_memory_snapshot(f"[Epoch {epoch+1}] End - ")
        
        # Report memory statistics
        if (epoch + 1) % 5 == 0:  # Report every 5 epochs
            self.memory_monitor.report()

        # Compute average metrics
        train_metrics = {}
        for key, values in epoch_losses.items():
            if values:
                avg_value = sum(values) / len(values)
                train_metrics[f'train_{key}'] = avg_value

        # Log to TensorBoard
        for model_name, tb_logger in self.tb_loggers.items():
            model_metrics = {k.replace(f'{model_name}_', ''): v 
                           for k, v in train_metrics.items() 
                           if model_name in k}
            tb_logger.log_scalars(model_metrics, epoch, prefix='train')

        log_metrics(logger, train_metrics, epoch, prefix="Train")

        return train_metrics

    def _train_batch(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Train on a single batch.
        
        Args:
            batch: Batch from dataloader
        
        Returns:
            Dictionary of batch losses
        """
        batch_losses = {}

        if 'grounding_dino' in self.models:
            dino_losses = self._train_grounding_dino_batch(batch)
            # Add all GroundingDINO loss components with prefix
            for key, value in dino_losses.items():
                batch_losses[f'grounding_dino_{key}'] = value

        if 'sam' in self.models:
            sam_loss = self._train_sam_batch(batch)
            batch_losses['sam_loss'] = sam_loss

        return batch_losses

    def _train_grounding_dino_batch(self, batch: Dict[str, Any]) -> float:
        """Train Grounding DINO on a batch with auxiliary losses."""
        model = self.models['grounding_dino']
        manager = self.training_managers['grounding_dino']
        criterion = self.losses['detection']

        def compute_loss(batch):
            # Get preprocessed DINO data (already transformed by official transforms)
            dino_data = batch['preprocessed']['grounding_dino']
            images = dino_data['images'].to(self.device)    # NestedTensor with .tensors and .mask
            boxes = dino_data['boxes'].to(self.device)      # [B, max_objs, 4] normalized [cx,cy,w,h]
            labels = dino_data['labels'].to(self.device)    # [B, max_objs]

            # Get batch size from labels (simplest and most reliable)
            batch_size = labels.shape[0]

            # Sanity check: verify we have valid data
            if batch_size == 0:
                logger.error("Empty batch received!")
                return {'loss': torch.tensor(0.0, device=self.device, requires_grad=True)}

            # Debug: check for valid objects
            total_valid_objs = (labels != -1).sum().item()
            if total_valid_objs == 0:
                logger.warning("Batch has no valid objects! Skipping...")
                return {'loss': torch.tensor(0.0, device=self.device, requires_grad=True)}

            # Get class names from class mapping
            class_names = list(self.dataset_info['class_mapping'].values())

            # Forward pass - Grounding DINO will format as "class1 . class2 . class3"
            # The model should return auxiliary outputs for DETR-style training
            outputs = model(images, class_names=class_names)

            # Step 1: Build caption and character-level token spans using official utility
            # This formats the caption exactly as Grounding DINO expects: "class1 . class2 . class3"
            caption, cat2tokenspan = build_captions_and_token_span(
                class_names, 
                force_lowercase=False
            )

            # # Step 2: Tokenize using the model's tokenizer (same as model uses internally)
            tokenized = model.tokenizer(
                caption,
                padding="longest",
                return_tensors="pt"
            ).to(self.device)

            # Step 3: Build class_id -> token_span mapping
            # Map from class_id (integer) to character spans
            class_id_to_name = {i: name for i, name in enumerate(class_names)}
            token_span_per_class = []
            for class_id in range(len(class_names)):
                class_name = class_id_to_name[class_id]
                if class_name not in cat2tokenspan:
                    raise ValueError(
                        f"Class '{class_name}' not found in cat2tokenspan!\n"
                        f"Available classes: {list(cat2tokenspan.keys())}\n"
                        f"This indicates a mismatch between class_names and caption tokenization.\n"
                        f"Check if class names contain special characters or case mismatches."
                    )
                token_span_per_class.append(cat2tokenspan[class_name])

            # Step 4: Create positive map using official utility
            # This converts character spans to actual BERT token positions
            # Returns: [num_classes, max_text_len] with 1.0 at relevant token positions
            positive_map = create_positive_map_from_span(
                tokenized, 
                token_span_per_class,
                max_text_len=outputs['pred_logits'].shape[-1]  # 256
            ).to(self.device)

            # Prepare targets in DETR format: list of dicts (one per batch element)
            targets = []
            for b in range(batch_size):
                # Get valid objects for this batch element
                valid_mask = labels[b] != -1
                valid_labels = labels[b][valid_mask]  # [num_valid_objs]
                valid_boxes = boxes[b][valid_mask]    # [num_valid_objs, 4]

                # Sanity check boxes (should be normalized [0, 1])
                if len(valid_boxes) > 0:
                    if (valid_boxes < 0).any() or (valid_boxes > 1).any():
                        logger.warning(f"Batch {b}: boxes not normalized! Range: [{valid_boxes.min():.3f}, {valid_boxes.max():.3f}]")

                # Create token labels for each valid object using the positive map
                # token_labels[i, j] = positive_map[class_id, j]
                # Shape: [num_valid_objs, max_text_len]
                token_labels = torch.zeros(
                    len(valid_labels),
                    positive_map.shape[1],  # max_text_len (256)
                    dtype=torch.float32,
                    device=self.device
                )

                # Use category_id_to_index mapping to handle sparse COCO-style IDs
                cat_id_to_idx = self.dataset_info['category_id_to_index']
                for obj_idx, class_id in enumerate(valid_labels):
                    cat_id = int(class_id.item())  # Original category_id from annotation
                    if cat_id not in cat_id_to_idx:
                        raise ValueError(
                            f"Unknown category_id {cat_id} not in category_id_to_index mapping!\n"
                            f"Available category_ids: {list(cat_id_to_idx.keys())}\n"
                            f"This indicates corrupted annotations or dataset_info mismatch.\n"
                            f"Batch {b}, object {obj_idx}."
                        )
                    class_idx = cat_id_to_idx[cat_id]  # Convert to 0-based index
                    token_labels[obj_idx] = positive_map[class_idx]

                # Create target dict for this batch element
                targets.append({
                    'labels': valid_labels,        # [num_valid_objs] - class IDs
                    'boxes': valid_boxes,          # [num_valid_objs, 4] in normalized [cx, cy, w, h]
                    'token_labels': token_labels,  # [num_valid_objs, max_text_len] - proper token labels
                })

            # Compute loss with Hungarian matching and auxiliary losses
            # Returns dict with keys like:
            # - 'loss_ce', 'loss_bbox', 'loss_giou' (final layer)
            # - 'loss_ce_0', 'loss_bbox_0', 'loss_giou_0' (decoder layer 0)
            # - ... (decoder layers 1-4)
            # - 'loss_ce_enc', 'loss_bbox_enc', 'loss_giou_enc' (encoder)
            loss_dict = criterion(outputs, targets)

            # Debug: check for NaN in loss components
            for k, v in loss_dict.items():
                if torch.isnan(v).any():
                    logger.error(f"NaN detected in {k}: {v}")

            # Compute total weighted loss
            total_loss = sum(loss_dict[k] * criterion.weight_dict[k]
                           for k in loss_dict.keys()
                           if k in criterion.weight_dict)

            # Final NaN check
            if torch.isnan(total_loss):
                logger.error("NaN in total_loss!")
                logger.error(f"Loss components: {loss_dict}")
                logger.error(f"Valid objects: {total_valid_objs}")
                logger.error(f"Pred logits range: [{outputs['pred_logits'].min():.3f}, {outputs['pred_logits'].max():.3f}]")
                logger.error(f"Pred boxes range: [{outputs['pred_boxes'].min():.3f}, {outputs['pred_boxes'].max():.3f}]")

            # Return dict with total loss and components for logging
            result = {'loss': total_loss}
            result.update({k: v.detach() for k, v in loss_dict.items()})

            return result

        # Training step with gradient management
        loss_dict = manager.training_step(batch, compute_loss)

        # Return all loss components for logging (convert to Dict[str, float])
        result = {'total_loss': loss_dict['loss'].item()}

        # Add individual components (detached, already on CPU from result dict)
        for key, value in loss_dict.items():
            if key != 'loss':
                result[key] = value.item() if torch.is_tensor(value) else value

        return result

    def _train_sam_batch(self, batch: Dict[str, Any]) -> float:
        """Train SAM on a batch."""
        model = self.models['sam']
        manager = self.training_managers['sam']

        def compute_loss(batch):
            # Get preprocessed SAM data (already transformed by official transforms)
            sam_data = batch['preprocessed']['sam']
            images = sam_data['images'].to(self.device)     # [B, 3, 1024, 1024]
            boxes = sam_data['boxes'].to(self.device)       # [B, max_objs, 4] already in SAM space!
            masks = sam_data['masks'].to(self.device)       # [B, max_objs, 1024, 1024]
            labels = sam_data['labels'].to(self.device)     # [B, max_objs]

            # Create validity mask
            valid_mask = (labels != -1)  # [B, max_objs]

            # Forward pass (boxes already in correct xyxy format from SAM preprocessing)
            outputs = model(images, box_prompts=boxes)

            # Targets
            targets = {
                'masks': masks,
                'valid_mask': valid_mask
            }

            # Compute loss
            loss_dict = self.losses['segmentation'](outputs, targets)
            return loss_dict

        # Training step with gradient management
        loss_dict = manager.training_step(batch, compute_loss)

        return loss_dict['loss'].item()

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate models.
        
        Args:
            epoch: Current epoch
        
        Returns:
            Dictionary of validation metrics
        """
        # Set models to eval mode
        for model in self.models.values():
            model.eval()

        val_losses = {}  # Dynamic keys

        pbar = tqdm(self.val_loader, desc="Validation")
        visualized = False

        for batch in pbar:
            # Validate each model
            batch_losses = self._validate_batch(batch)

            # Visualize first batch only (for debugging)
            if self.visualizer and not visualized:
                self._visualize_batch(batch, epoch)
                visualized = True

            for key, value in batch_losses.items():
                if key not in val_losses:
                    val_losses[key] = []
                val_losses[key].append(value)

            # Update progress bar - show main loss and components
            postfix = {}
            for key, values in val_losses.items():
                if values:
                    # Show only main losses and key components
                    if 'total_loss' in key or key in ['grounding_dino_loss_ce', 'grounding_dino_loss_bbox', 'grounding_dino_loss_giou']:
                        postfix[key] = f"{values[-1]:.4f}"
            pbar.set_postfix(postfix)

        # Compute average metrics
        val_metrics = {}
        for key, values in val_losses.items():
            if values:
                avg_value = sum(values) / len(values)
                val_metrics[f'val_{key}'] = avg_value

        # Log to TensorBoard
        for model_name, tb_logger in self.tb_loggers.items():
            model_metrics = {k.replace(f'{model_name}_', ''): v 
                           for k, v in val_metrics.items() 
                           if model_name in k}
            tb_logger.log_scalars(model_metrics, epoch, prefix='val')

        log_metrics(logger, val_metrics, epoch, prefix="Val")

        # Set models back to train mode
        for model in self.models.values():
            model.train()

        return val_metrics

    def _validate_batch(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Validate on a single batch."""
        batch_losses = {}

        # Validate Grounding DINO if loaded
        if 'grounding_dino' in self.models:
            # Get preprocessed data (same structure as training)
            dino_data = batch['preprocessed']['grounding_dino']
            images = dino_data['images'].to(self.device)
            boxes = dino_data['boxes'].to(self.device)
            labels = dino_data['labels'].to(self.device)

            batch_size = labels.shape[0]
            class_names = list(self.dataset_info['class_mapping'].values())
            criterion = self.losses['detection']
            model = self.models['grounding_dino']

            outputs = model(images, class_names=class_names)

            # Use official Grounding DINO token mapping (same as training)
            caption, cat2tokenspan = build_captions_and_token_span(
                class_names,
                force_lowercase=False
            )

            tokenized = model.tokenizer(
                caption,
                padding="longest",
                return_tensors="pt"
            ).to(self.device)

            class_id_to_name = {i: name for i, name in enumerate(class_names)}
            token_span_per_class = []
            for class_id in range(len(class_names)):
                class_name = class_id_to_name[class_id]
                if class_name in cat2tokenspan:
                    token_span_per_class.append(cat2tokenspan[class_name])
                else:
                    token_span_per_class.append([])

            positive_map = create_positive_map_from_span(
                tokenized,
                token_span_per_class,
                max_text_len=outputs['pred_logits'].shape[-1]
            ).to(self.device)

            # Prepare targets in DETR format: list of dicts
            targets = []
            for b in range(batch_size):
                valid_mask = labels[b] != -1
                valid_labels = labels[b][valid_mask]
                valid_boxes = boxes[b][valid_mask]

                # Create token labels using official positive map
                token_labels = torch.zeros(
                    len(valid_labels),
                    positive_map.shape[1],
                    dtype=torch.float32,
                    device=self.device
                )

                # Use category_id_to_index mapping to handle sparse COCO-style IDs
                cat_id_to_idx = self.dataset_info['category_id_to_index']
                for obj_idx, class_id in enumerate(valid_labels):
                    cat_id = int(class_id.item())
                    if cat_id in cat_id_to_idx:
                        class_idx = cat_id_to_idx[cat_id]
                        token_labels[obj_idx] = positive_map[class_idx]

                targets.append({
                    'labels': valid_labels,
                    'boxes': valid_boxes,
                    'token_labels': token_labels,
                })

            # Compute loss with auxiliary losses
            loss_dict = criterion(outputs, targets)

            # Compute total weighted loss
            total_loss = sum(loss_dict[k] * criterion.weight_dict[k]
                           for k in loss_dict.keys()
                           if k in criterion.weight_dict)

            # Add all loss components with prefix
            batch_losses['grounding_dino_total_loss'] = total_loss.item()
            for key, value in loss_dict.items():
                batch_losses[f'grounding_dino_{key}'] = value.item() if torch.is_tensor(value) else value

        # Validate SAM if loaded
        if 'sam' in self.models:
            # Get preprocessed data (same structure as training)
            sam_data = batch['preprocessed']['sam']
            images = sam_data['images'].to(self.device)
            boxes = sam_data['boxes'].to(self.device)
            masks = sam_data['masks'].to(self.device)
            labels = sam_data['labels'].to(self.device)

            valid_mask = (labels != -1)

            outputs = self.models['sam'](images, box_prompts=boxes)

            targets = {
                'masks': masks,
                'valid_mask': valid_mask
            }

            loss_dict = self.losses['segmentation'](outputs, targets)
            batch_losses['sam_loss'] = loss_dict['loss'].item()

        return batch_losses

    def _visualize_batch(self, batch: Dict[str, Any], epoch: int):
        """
        Visualize predictions for debugging.
        
        Loads original images and draws GT vs predicted bounding boxes.
        """
        import numpy as np
        from PIL import Image

        if 'grounding_dino' not in self.models:
            return

        try:
            file_names = batch['file_names']

            # Get preprocessed data
            dino_data = batch['preprocessed']['grounding_dino']
            images_tensor = dino_data['images'].to(self.device)
            gt_boxes = dino_data['boxes'].to(self.device)  # [B, max_obj, 4] normalized cxcywh
            labels = dino_data['labels'].to(self.device)   # [B, max_obj]

            # Get predictions using model.predict() - handles token-to-class internally
            class_names = list(self.dataset_info['class_mapping'].values())
            model = self.models['grounding_dino']
            predictions = model.predict(images_tensor, class_names, confidence_threshold=0.3)

            batch_size = len(file_names)
            images_list = []
            predictions_list = {'boxes': [], 'labels': []}
            targets_list = {'boxes': [], 'labels': []}

            for b in range(min(batch_size, 8)):  # Max 8 samples
                # Load original image using path resolver
                img_path = Path(self.data_manager.get_image_path(file_names[b]))
                if not img_path.exists():
                    continue

                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img)
                img_w, img_h = img.size

                # Convert GT boxes: cxcywh normalized -> xyxy pixel
                valid_mask = labels[b] != -1
                gt_box = gt_boxes[b][valid_mask].cpu().numpy()  # [N, 4]
                gt_label_raw = labels[b][valid_mask].cpu().numpy() # [N]

                cat_id_to_idx = self.dataset_info['category_id_to_index']
                gt_label = []
                for cat_id in gt_label_raw:
                    cat_id_int = int(cat_id)
                    if cat_id_int not in cat_id_to_idx:
                        raise ValueError(
                            f"Unknown category_id {cat_id_int} in visualization!\n"
                            f"Available category_ids: {list(cat_id_to_idx.keys())}\n"
                            f"This indicates corrupted annotations or dataset_info mismatch."
                        )
                    gt_label.append(cat_id_to_idx[cat_id_int])
                gt_label = np.array(gt_label)

                if len(gt_box) > 0:
                    # cxcywh to xyxy
                    cx, cy, w, h = gt_box[:, 0], gt_box[:, 1], gt_box[:, 2], gt_box[:, 3]
                    gt_box_xyxy = np.stack([
                        (cx - w/2) * img_w,
                        (cy - h/2) * img_h,
                        (cx + w/2) * img_w,
                        (cy + h/2) * img_h
                    ], axis=1)
                else:
                    gt_box_xyxy = np.array([])
                    gt_label = np.array([])

                # Get predictions from model.predict() (already filtered by confidence)
                pred = predictions[b]
                pred_box = pred['boxes'].cpu().numpy()      # [N, 4] normalized cxcywh
                pred_score = pred['scores'].cpu().numpy()   # [N]
                pred_label = pred['labels'].cpu().numpy()   # [N]

                # Debug info
                logger.debug("Visualization sample %d: %d predictions, %d GT boxes",
                            b, len(pred_box), len(gt_box))

                if len(pred_box) > 0:
                    # cxcywh to xyxy (using original image size for visualization)
                    cx, cy, w, h = pred_box[:, 0], pred_box[:, 1], pred_box[:, 2], pred_box[:, 3]
                    pred_box_xyxy = np.stack([
                        (cx - w/2) * img_w,
                        (cy - h/2) * img_h,
                        (cx + w/2) * img_w,
                        (cy + h/2) * img_h
                    ], axis=1)
                else:
                    pred_box_xyxy = np.array([])
                    pred_label = np.array([])

                images_list.append(img_array)
                predictions_list['boxes'].append(pred_box_xyxy)
                predictions_list['labels'].append(pred_label)
                targets_list['boxes'].append(gt_box_xyxy)
                targets_list['labels'].append(gt_label)

            if images_list:
                self.visualizer.save_batch(
                    epoch=epoch,
                    images=images_list,
                    predictions=predictions_list,
                    targets=targets_list,
                    class_names=class_names,
                    image_ids=list(range(len(images_list)))
                )

        except Exception as e:
            logger.warning(f"Visualization failed: {e}")

    def _resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")

        for model_name, model in self.models.items():
            ckpt_manager = self.checkpoint_managers[model_name]
            checkpoint = ckpt_manager.load_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=self.optimizers[model_name],
                scheduler=self.schedulers.get(model_name)
            )
            if 'training_manager_state' in checkpoint:
                self.training_managers[model_name].load_state_dict(
                    checkpoint['training_manager_state']
                )
