"""
Trainer - Main orchestrator for teacher model training.

This module provides the main entry point for training teacher models.
It coordinates:
- Dataset/dataloader creation
- Model trainer creation (via factory pattern)
- Epoch loops (train/validate)
- Checkpoint coordination
- Progress reporting and cancellation
- Test evaluation and export
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Optional, Callable

import torch
from tqdm import tqdm

from ml_engine.data.loaders import create_dataloader
from ml_engine.data.manager import DataManager
from ml_engine.data.dataset_factory import DatasetFactory
from ml_engine.evaluation import PredictionVisualizer
from ml_engine.evaluation.evaluator import ModelEvaluator
from ml_engine.evaluation.report import ModelReportGenerator
from ml_engine.export import create_export_package
from core.constants import GROUNDING_DINO, SAM
from core.config import save_config
from core.log_utils import log_config, log_metrics

from .model_trainers.grounding_dino import GroundingDINOTrainer
from .model_trainers.sam import SAMTrainer
from .model_trainers.base import BaseModelTrainer

logger = logging.getLogger(__name__)


class TrainingCancelledException(Exception):
    """Raised when training is cancelled by user request."""
    pass


# Factory registry for model trainers
TRAINER_REGISTRY: Dict[str, type] = {
    GROUNDING_DINO: GroundingDINOTrainer,
    SAM: SAMTrainer,
}


class Trainer:
    """
    Main trainer for teacher models with data-driven model loading.
    
    This class coordinates training of multiple models (DINO, SAM) by:
    - Creating appropriate model trainers based on dataset requirements
    - Running training/validation epochs
    - Coordinating checkpointing across models
    - Running test evaluation and creating export packages
    
    Example:
        >>> from ml_engine.data.manager import DataManager
        >>> from ml_engine.training import Trainer
        >>> 
        >>> manager = DataManager(
        >>>     data_path='data/annotations.json',
        >>>     image_paths=[...],
        >>>     split_config={'train': 0.7, 'val': 0.2, 'test': 0.1}
        >>> )
        >>> 
        >>> trainer = Trainer(
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
        Initialize the Trainer.
        
        Args:
            data_manager: DataManager instance with train/val/test splits
            output_dir: Output directory for checkpoints and logs
            config: Training configuration
            resume_from: Optional checkpoint path to resume from
            progress_callback: Optional callback for progress reporting
            cancel_check: Optional function that returns True to cancel training
        """
        self.data_manager = data_manager
        self.output_dir = Path(output_dir)
        self.config = config
        self.resume_from = resume_from
        self.progress_callback = progress_callback
        self.cancel_check = cancel_check

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config for reproducibility
        config_path = self.output_dir / 'teacher_config.yaml'
        save_config(config, str(config_path))
        logger.info("Saved config to: %s", config_path)

        # Setup device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logger.info("Using device: %s", self.device)
        if torch.cuda.is_available():
            logger.info("GPU: %s", torch.cuda.get_device_name(0))

        # Get dataset info
        self.dataset_info = data_manager.get_dataset_info()
        self.required_models = data_manager.get_required_models()

        logger.info("Dataset: boxes=%s, masks=%s, classes=%d",
                   self.dataset_info['has_boxes'],
                   self.dataset_info['has_masks'],
                   self.dataset_info['num_classes'])
        logger.info("Required models: %s", self.required_models)

        # Initialize components
        self._init_datasets()
        self._init_trainers()
        self._init_visualizer()

        # Resume from checkpoint if provided
        if resume_from:
            self._resume_from_checkpoint(resume_from)

    def _init_datasets(self) -> None:
        """Initialize datasets and dataloaders."""
        train_data = self.data_manager.get_split('train')
        val_data = self.data_manager.get_split('val')
        
        # Check SAM single object sampling
        sam_config = self.config.get('models', {}).get('sam', {})
        sam_single_object_sampling = sam_config.get('training', {}).get('single_object_sampling', False)
        
        # Create datasets
        self.train_dataset = DatasetFactory.create_dataset(
            coco_data=train_data,
            image_path_resolver=self.data_manager.get_image_path,
            dataset_info=self.dataset_info,
            model_names=self.required_models,
            augmentation_config=self.config.get('augmentation'),
            is_training=True,
            sam_single_object_sampling=sam_single_object_sampling
        )
        
        self.val_dataset = DatasetFactory.create_dataset(
            coco_data=val_data,
            image_path_resolver=self.data_manager.get_image_path,
            dataset_info=self.dataset_info,
            model_names=self.required_models,
            augmentation_config=None,
            is_training=False,
            sam_single_object_sampling=False
        )
        
        # Create dataloaders
        batch_size = self.config.get('batch_size', 8)
        num_workers = self.config.get('num_workers', 4)
        
        self.train_loader = create_dataloader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        self.val_loader = create_dataloader(
            self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        logger.info("✓ Datasets: %d train, %d val", len(self.train_dataset), len(self.val_dataset))
    
    def _init_trainers(self) -> None:
        """Create model trainers based on required models."""
        self.trainers: Dict[str, BaseModelTrainer] = {}

        for model_name in self.required_models:
            if model_name not in TRAINER_REGISTRY:
                raise ValueError(f"Unknown model: {model_name}. Available: {list(TRAINER_REGISTRY.keys())}")

            trainer_cls = TRAINER_REGISTRY[model_name]
            model_config = self.config['models'][model_name]

            # Add shared config values
            model_config = {**model_config, 'epochs': self.config.get('epochs', 50)}

            self.trainers[model_name] = trainer_cls(
                config=model_config,
                device=self.device,
                output_dir=self.output_dir,
                dataset_info=self.dataset_info
            )

        logger.info("✓ Created %d model trainers", len(self.trainers))
    
    def _init_visualizer(self) -> None:
        """Initialize prediction visualizer."""
        save_predictions = self.config.get('evaluation', {}).get('save_predictions', False)
        if save_predictions:
            self.visualizer = PredictionVisualizer(
                output_dir=str(self.output_dir / 'predictions'),
                max_samples_per_epoch=8,
                enabled=True
            )
        else:
            self.visualizer = None
    
    def train(self) -> None:
        """
        Main training loop.
        
        Raises:
            TrainingCancelledException: If cancel_check returns True
        """
        epochs = self.config.get('epochs', 50)
        
        logger.info("=" * 60)
        logger.info("Starting Teacher Model Training")
        logger.info("=" * 60)
        log_config(logger, self.config, "Training Configuration")
        
        try:
            for epoch in range(epochs):
                # Check cancellation
                if self.cancel_check and self.cancel_check():
                    logger.info("Training cancelled by user")
                    raise TrainingCancelledException("Training cancelled")
                
                logger.info("\nEpoch %d/%d", epoch + 1, epochs)
                logger.info("-" * 60)
                
                # Train
                train_metrics = self._train_epoch(epoch)
                
                # Validate (at specified interval)
                eval_interval = self.config.get('evaluation', {}).get('interval', 1)
                if (epoch + 1) % eval_interval == 0:
                    val_metrics = self._validate_epoch(epoch)
                else:
                    val_metrics = {}
                
                # Merge metrics
                all_metrics = {**train_metrics, **val_metrics, 'epoch': epoch}
                
                # Save checkpoints and log
                for name, trainer in self.trainers.items():
                    trainer.save_checkpoint(epoch, all_metrics)
                    trainer.log_metrics(train_metrics, epoch, prefix='train')
                    if val_metrics:
                        trainer.log_metrics(val_metrics, epoch, prefix='val')
                    
                    if trainer.should_stop:
                        logger.info("Early stopping triggered for %s", name)
                        break
                
                # Report progress
                if self.progress_callback:
                    self.progress_callback({
                        'current_epoch': epoch + 1,
                        'total_epochs': epochs,
                        'train_metrics': train_metrics,
                        'val_metrics': val_metrics,
                        'message': f"Completed epoch {epoch + 1}/{epochs}"
                    })
            
            logger.info("=" * 60)
            logger.info("Training Completed!")
            logger.info("=" * 60)
            
            # Finalize
            self._save_adapters()
            self._evaluate_on_test_set()
            self._create_export_package()
        
        finally:
            # Cleanup
            for trainer in self.trainers.values():
                trainer.close()
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch across all models."""
        metrics_acc = defaultdict(list)
        total_steps = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Train {epoch + 1}")
        for step, batch in enumerate(pbar):
            # Check cancellation
            if self.cancel_check and self.cancel_check():
                raise TrainingCancelledException("Training cancelled")
            
            # Train each model
            for name, trainer in self.trainers.items():
                losses = trainer.train_batch(batch)
                for k, v in losses.items():
                    metrics_acc[f"{name}_{k}"].append(v)
            
            # Update progress bar
            postfix = {}
            for k, v in metrics_acc.items():
                if 'total_loss' in k:
                    postfix[k] = f"{v[-1]:.4f}"
            pbar.set_postfix(postfix)
            
            # Report step progress
            if self.progress_callback and total_steps > 0:
                report_interval = max(1, total_steps // 10)
                if step % report_interval == 0:
                    self.progress_callback({
                        'current_epoch': epoch,
                        'total_epochs': self.config.get('epochs', 50),
                        'current_step': step + 1,
                        'total_steps': total_steps,
                        'message': f"Epoch {epoch + 1}, Step {step + 1}/{total_steps}"
                    })
        
        # Average metrics
        train_metrics = {f'train_{k}': sum(v) / len(v) for k, v in metrics_acc.items()}
        log_metrics(logger, train_metrics, epoch, prefix="Train")
        
        return train_metrics
    
    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch across all models."""
        metrics_acc = defaultdict(list)
        
        pbar = tqdm(self.val_loader, desc="Validation")
        for batch in pbar:
            for name, trainer in self.trainers.items():
                losses = trainer.validate_batch(batch)
                for k, v in losses.items():
                    metrics_acc[f"{name}_{k}"].append(v)
            
            postfix = {}
            for k, v in metrics_acc.items():
                if 'total_loss' in k:
                    postfix[k] = f"{v[-1]:.4f}"
            pbar.set_postfix(postfix)
        
        val_metrics = {f'val_{k}': sum(v) / len(v) for k, v in metrics_acc.items()}
        log_metrics(logger, val_metrics, epoch, prefix="Val")
        
        return val_metrics
    
    def _save_adapters(self) -> None:
        """Save LoRA adapters for all models."""
        artifacts = {}
        for model_name, trainer in self.trainers.items():
            manifest_path = trainer.save_adapters()
            if manifest_path:
                rel_path = manifest_path.relative_to(self.output_dir)
                artifacts[model_name] = str(rel_path)

        from ml_engine.artifacts import BundleManifest
        bundle_manifest = BundleManifest(
            bundle_type="teacher_training_output",
            artifacts=artifacts,
            lineage={"job_id": None,},
            merged_checkpoints=None
        )
        bundle_manifest.save(self.output_dir / "bundle.manifest.json")
    
    def _evaluate_on_test_set(self) -> None:
        """Evaluate on held-out test set."""
        try:
            test_data = self.data_manager.get_split('test')
        except ValueError:
            logger.warning("No test split available. Skipping evaluation.")
            return
        
        if not test_data.get('images'):
            logger.warning("Test split is empty. Skipping evaluation.")
            return
        
        logger.info("=" * 60)
        logger.info("Running Test Set Evaluation")
        logger.info("=" * 60)
        
        # Create test dataloader
        test_dataset = DatasetFactory.create_dataset(
            coco_data=test_data,
            image_path_resolver=self.data_manager.get_image_path,
            dataset_info=self.dataset_info,
            model_names=self.required_models,
            augmentation_config=None,
            is_training=False
        )
        
        test_loader = create_dataloader(
            test_dataset,
            batch_size=self.config.get('batch_size', 8),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4)
        )
        
        logger.info("Test set: %d images", len(test_dataset))
        
        # Initialize evaluator
        evaluator = ModelEvaluator(
            device=str(self.device),
            confidence_threshold=self.config.get('evaluation', {}).get('confidence_threshold', 0.3)
        )
        report_generator = ModelReportGenerator()
        class_names = list(self.dataset_info['class_mapping'].values())
        
        all_reports = []
        
        for name, trainer in self.trainers.items():
            logger.info("Evaluating %s...", name)
            
            # Load best checkpoint
            best_path = trainer.get_checkpoint_manager().get_best_checkpoint_path()
            if best_path.exists():
                trainer.load_checkpoint(str(best_path))
            
            # Evaluate
            model = trainer.get_model()
            if name == GROUNDING_DINO:
                results = evaluator.evaluate_detection(
                    model=model, dataloader=test_loader,
                    class_names=class_names, dataset_info=self.dataset_info
                )
            elif name == SAM:
                results = evaluator.evaluate_segmentation(
                    model=model, dataloader=test_loader,
                    class_names=class_names, dataset_info=self.dataset_info
                )
            else:
                continue
            
            # Generate report
            report = report_generator.generate_report(
                evaluation_results=results,
                model_name=name,
                test_set_size=len(test_dataset),
                extra_info={'config': self.config['models'][name]}
            )
            
            report_path = self.output_dir / 'evaluation' / f'{name}_report.json'
            report_generator.save_report(report, str(report_path))
            
            summary = report_generator.generate_summary_text(report)
            logger.info("\n%s", summary)
            
            all_reports.append(report)
        
        if len(all_reports) > 1:
            combined = report_generator.combine_reports(all_reports)
            combined_path = self.output_dir / 'evaluation' / 'combined_report.json'
            report_generator.save_report(combined, str(combined_path))
        
        logger.info("Reports saved to: %s", self.output_dir / 'evaluation')
    
    def _create_export_package(self) -> None:
        """Create downloadable export packages."""
        logger.info("Creating export packages...")
        
        class_names = list(self.dataset_info['class_mapping'].values())
        
        for name, trainer in self.trainers.items():
            try:
                model_config = self.config['models'][name]
                training_info = {
                    'epochs': self.config.get('epochs'),
                    'batch_size': self.config.get('batch_size'),
                    'learning_rate': model_config.get('learning_rate')
                }

                report_path = self.output_dir / 'evaluation' / f'{name}_report.json'
                if report_path.exists():
                    import json
                    with open(report_path) as f:
                        report = json.load(f)
                        metrics = report.get('technical_metrics', {})
                        training_info['mAP50'] = metrics.get('mAP50', 0)
                        training_info['mIoU'] = metrics.get('mIoU', 0)

                zip_path = create_export_package(
                    model=trainer.get_model(),
                    output_dir=self.output_dir,
                    class_names=class_names,
                    model_name=name,
                    training_info=training_info
                )
                logger.info("✓ Export package: %s", zip_path)

            except Exception as e:
                logger.error("Failed to create export for %s: %s", name, e)
    
    def _resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Resume training from checkpoint."""
        logger.info("Resuming from: %s", checkpoint_path)
        for trainer in self.trainers.values():
            trainer.load_checkpoint(checkpoint_path)


# Backward compatibility alias
TeacherTrainer = Trainer
