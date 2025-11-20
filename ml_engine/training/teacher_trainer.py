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
from typing import Dict, Optional, Any
from pathlib import Path
import torch
from tqdm import tqdm

from ml_engine.data.loaders import create_dataloader
from ml_engine.data.preprocessing import create_preprocessor_from_models
from ml_engine.data.manager import DataManager
from ml_engine.models.teacher.grounding_dino_lora import load_grounding_dino_with_lora
from ml_engine.models.teacher.sam_lora import load_sam_with_lora
from ml_engine.training.losses import build_criterion, SegmentationLoss
from ml_engine.training.training_manager import TrainingManager
from ml_engine.training.checkpoint_manager import CheckpointManager
from core.logger import TensorBoardLogger, log_config, log_metrics
from core.constants import DEFAULT_CONFIGS_DIR
from augmentation import get_augmentation_registry

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
        >>> trainer = TeacherTrainer(
        >>>     train_data_path='data/raw/train.json',
        >>>     val_data_path='data/raw/val.json',
        >>>     image_dir='data/raw/images',
        >>>     output_dir='experiments/exp1',
        >>>     config=config
        >>> )
        >>> 
        >>> trainer.train()
    """

    def __init__(
        self,
        data_manager: DataManager,
        output_dir: str,
        config: Dict[str, Any],
        resume_from: Optional[str] = None
    ):
        """
        Initialize TeacherTrainer with DataManager.
        
        Args:
            data_manager: DataManager instance with train/val splits
            output_dir: Output directory for checkpoints and logs
            config: Training configuration
            resume_from: Optional checkpoint path to resume from
        
        Design philosophy:
        - User provides ONE dataset file
        - Platform handles splitting (train/val/test)
        - Trainer uses 'train' and 'val' splits from the manager
        
        Example:
            >>> # User provides single annotations.json
            >>> manager = DataManager(
            >>>     data_path='data/raw/annotations.json',
            >>>     image_dir='data/raw/images/',
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

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'teachers').mkdir(exist_ok=True)

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("Using device: %s", self.device)

        self.dataset_info = data_manager.get_dataset_info()

        logger.info("Dataset has boxes: %s", self.dataset_info['has_boxes'])
        logger.info("Dataset has masks: %s", self.dataset_info['has_masks'])
        logger.info("Num classes: %s", self.dataset_info['num_classes'])

        # Determine which models to load (DATA-DRIVEN!)
        self.required_models = data_manager.get_required_models()
        logger.info("Required teacher models: %s", self.required_models)

        # Initialize components
        self._init_augmentation()
        self._init_preprocessor()
        self._init_datasets()
        self._init_models()
        self._init_losses()
        self._init_optimizers()
        self._init_training_manager()
        self._init_checkpoint_manager()
        self._init_loggers()

        # Resume from checkpoint if provided
        if resume_from:
            self._resume_from_checkpoint(resume_from)

    def _init_augmentation(self):
        """Initialize augmentation pipeline."""
        aug_config = self.config['augmentation']

        if aug_config['enabled']:
            registry = get_augmentation_registry()

            self.augmentation_pipeline = registry.get_pipeline(
                characteristics=aug_config['characteristics'],
                environment=aug_config['environment'],
                intensity=aug_config['intensity']
            )
            logger.info("✓ Augmentation pipeline created")
        else:
            self.augmentation_pipeline = None
            logger.info("Augmentation disabled")

    def _init_preprocessor(self):
        """Initialize multi-model preprocessor."""
        self.preprocessor = create_preprocessor_from_models(
            model_names=self.required_models
        )
        logger.info("✓ Preprocessor initialized for: %s", self.required_models)

    def _init_datasets(self):
        """Initialize datasets and dataloaders using DataManager."""
        # Training dataset - use 'train' split from manager
        self.train_dataset = self.data_manager.create_pytorch_dataset(
            split='train',
            preprocessor=self.preprocessor,
            augmentation_pipeline=self.augmentation_pipeline
        )

        # Validation dataset - use 'val' split (no augmentation)
        self.val_dataset = self.data_manager.create_pytorch_dataset(
            split='val',  # Platform-created val split
            preprocessor=self.preprocessor,
            augmentation_pipeline=None  # No augmentation for validation
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

        # Load Grounding DINO if dataset has boxes
        if 'grounding_dino' in self.required_models:
            model_config = self.config.get('models').get('grounding_dino')
            logger.info("Loading Grounding DINO with LoRA...")
            base_ckpt = model_config.get('model').get('base_checkpoint',
                                        'data/models/pretrained/groundingdino_swint_ogc.pth')
            self.models['grounding_dino'] = load_grounding_dino_with_lora(
                base_checkpoint=base_ckpt,
                lora_config=model_config.get('lora'),
                freeze_backbone=model_config.get('freeze_backbone', True),
                freeze_bbox_embed=model_config.get('freeze_bbox_embed', False),
                bert_model_path=model_config.get('bert_model_path', None)
            ).to(self.device)

            logger.info(" Grounding DINO loaded")

        # Load SAM if dataset has masks
        if 'sam' in self.required_models:
            logger.info("Loading SAM with LoRA...")
            base_ckpt = self.config.get('sam_checkpoint',
                                        'data/models/pretrained/sam_vit_h_4b8939.pth')
            model_type = self.config.get('sam_model_type', 'vit_h')

            self.models['sam'] = load_sam_with_lora(
                base_checkpoint=base_ckpt,
                model_type=model_type,
                lora_config=self.config.get('sam_lora', {})
            ).to(self.device)

            logger.info("SAM loaded")

        # Set models to training mode
        for model in self.models.values():
            model.train()

    def _init_losses(self):
        """Initialize loss functions based on loaded models."""
        self.losses = {}

        if 'grounding_dino' in self.models:
            # Use proper DETR-style criterion with Hungarian matching
            num_classes = self.dataset_info['num_classes']
            num_decoder_layers = self.config.get('num_decoder_layers', 6)

            # Build criterion with auxiliary losses
            self.losses['detection'] = build_criterion(
                num_classes=num_classes,
                num_decoder_layers=num_decoder_layers,
                focal_alpha=0.25,
                focal_gamma=2.0
            )

            logger.info(f"✓ Grounding DINO criterion with Hungarian matching initialized")
            logger.info(f"  - Num classes: {num_classes}")
            logger.info(f"  - Num decoder layers: {num_decoder_layers}")
            logger.info(f"  - Auxiliary losses: {num_decoder_layers - 1} intermediate + 1 encoder")

        if 'sam' in self.models:
            self.losses['segmentation'] = SegmentationLoss().to(self.device)

    def _init_optimizers(self):
        """Initialize optimizers for each model."""
        self.optimizers = {}
        self.schedulers = {}
        
        # in default config, `learning_rate` as actually model dependent
        # but leave it as it is for now
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-4)
        optimizer_type = self.config.get('optimizer', 'AdamW')

        for model_name, model in self.models.items():
            # Get trainable parameters
            trainable_params = [p for p in model.parameters() if p.requires_grad]

            # Create optimizer
            if optimizer_type == 'AdamW':
                optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=lr,
                    weight_decay=weight_decay
                )
            elif optimizer_type == 'SGD':
                optimizer = torch.optim.SGD(
                    trainable_params,
                    lr=lr,
                    momentum=self.config.get('momentum', 0.9),
                    weight_decay=weight_decay
                )
            else:
                raise ValueError(f"Unknown optimizer: {optimizer_type}")

            self.optimizers[model_name] = optimizer

            # Create scheduler
            warmup_epochs = self.config.get('warmup_epochs', 3)
            total_epochs = self.config.get('epochs', 50)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=total_epochs - warmup_epochs,
                T_mult=1
            )
            self.schedulers[model_name] = scheduler

            logger.info(f"✓ Optimizer and scheduler created for {model_name}")
    
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

            logger.info(f"✓ TrainingManager created for {model_name}")
    
    def _init_checkpoint_manager(self):
        """Initialize checkpoint managers for each model."""
        self.checkpoint_managers = {}
        
        for model_name in self.models.keys():
            output_dir = self.output_dir / 'teachers' / f'{model_name}_lora'
            config_path = DEFAULT_CONFIGS_DIR / 'checkpoint_config.yaml'
            
            manager = CheckpointManager(
                output_dir=str(output_dir),
                config_path=str(config_path)
            )
            self.checkpoint_managers[model_name] = manager
            
            logger.info(f"✓ CheckpointManager created for {model_name}: {output_dir}")
    
    def _init_loggers(self):
        """Initialize TensorBoard loggers."""
        self.tb_loggers = {}
        
        for model_name in self.models.keys():
            log_dir = self.output_dir / 'logs' / model_name
            self.tb_loggers[model_name] = TensorBoardLogger(str(log_dir))
            logger.info(f"✓ TensorBoard logger: {log_dir}")
    
    def train(self):
        """
        Main training loop.
        
        Trains all loaded models (data-driven).
        """
        epochs = self.config.get('epochs', 50)
        start_epoch = 0
        
        logger.info("=" * 60)
        logger.info("Starting Teacher Model Training")
        logger.info("=" * 60)
        log_config(logger, self.config, "Training Configuration")
        
        for epoch in range(start_epoch, epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            logger.info("-" * 60)
            
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics, 'epoch': epoch}
            
            # Save checkpoints
            for model_name in self.models.keys():
                self.checkpoint_managers[model_name].save_checkpoint(
                    epoch=epoch,
                    model=self.models[model_name],
                    optimizer=self.optimizers[model_name],
                    metrics=all_metrics,
                    scheduler=self.schedulers.get(model_name),
                    scaler=self.training_managers[model_name].scaler,
                    extra_info={'config': self.config}
                )
                
                # Check early stopping
                if self.checkpoint_managers[model_name].should_stop:
                    logger.info(f"Early stopping triggered for {model_name}")
                    break
        
        logger.info("=" * 60)
        logger.info("Training Completed!")
        logger.info("=" * 60)
        
        # Close loggers
        for tb_logger in self.tb_loggers.values():
            tb_logger.close()
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dictionary of training metrics
        """
        # Set models to train mode
        for model in self.models.values():
            model.train()
        
        # Update epoch in training managers
        for manager in self.training_managers.values():
            manager.set_epoch(epoch)
        
        # Metrics accumulator
        epoch_losses = {f'{model_name}_loss': [] for model_name in self.models.keys()}
        
        # Training loop
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch + 1}")
        for batch in pbar:
            # Train each loaded model
            batch_losses = self._train_batch(batch)
            
            # Accumulate losses
            for key, value in batch_losses.items():
                if key in epoch_losses:
                    epoch_losses[key].append(value)
            
            # Update progress bar
            pbar.set_postfix({k: f"{v[-1]:.4f}" for k, v in epoch_losses.items() if v})
        
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
        
        # Train Grounding DINO if loaded
        if 'grounding_dino' in self.models:
            dino_loss = self._train_grounding_dino_batch(batch)
            batch_losses['grounding_dino_loss'] = dino_loss
        
        # Train SAM if loaded
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
            images = dino_data['images'].to(self.device)    # [B, 3, H, W] padded
            boxes = dino_data['boxes'].to(self.device)      # [B, max_objs, 4] already normalized!
            labels = dino_data['labels'].to(self.device)    # [B, max_objs]
            
            # Get batch size
            batch_size = images.shape['tensors.shape'][0]
            
            # Get class names from class mapping
            class_names = list(self.dataset_info['class_mapping'].values())
            
            # Forward pass - Grounding DINO will format as "class1 . class2 . class3"
            # The model should return auxiliary outputs for DETR-style training
            outputs = model(images, class_names=class_names)
            
            # Ensure model returns auxiliary outputs
            if 'aux_outputs' not in outputs:
                logger.warning("Model not returning auxiliary outputs! Training may be suboptimal.")
            
            # Prepare targets in DETR format: list of dicts (one per batch element)
            targets = []
            for b in range(batch_size):
                # Get valid objects for this batch element
                valid_mask = labels[b] != -1
                valid_labels = labels[b][valid_mask]  # [num_valid_objs]
                valid_boxes = boxes[b][valid_mask]    # [num_valid_objs, 4]
                
                # Create target dict for this batch element
                targets.append({
                    'labels': valid_labels,  # [num_valid_objs]
                    'boxes': valid_boxes,    # [num_valid_objs, 4] in [cx, cy, w, h] format
                })
            
            # Compute loss with Hungarian matching and auxiliary losses
            # Returns dict with keys like:
            # - 'loss_ce', 'loss_bbox', 'loss_giou' (final layer)
            # - 'loss_ce_0', 'loss_bbox_0', 'loss_giou_0' (decoder layer 0)
            # - ... (decoder layers 1-4)
            # - 'loss_ce_enc', 'loss_bbox_enc', 'loss_giou_enc' (encoder)
            loss_dict = criterion(outputs, targets)
            
            # Compute total weighted loss
            total_loss = sum(loss_dict[k] * criterion.weight_dict[k] 
                           for k in loss_dict.keys() 
                           if k in criterion.weight_dict)
            
            # Return dict with total loss and components for logging
            result = {'loss': total_loss}
            result.update({k: v.detach() for k, v in loss_dict.items()})
            
            return result
        
        # Training step with gradient management
        loss_dict = manager.training_step(batch, compute_loss)
        
        return loss_dict['loss'].item()
    
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
        
        val_losses = {f'{model_name}_loss': [] for model_name in self.models.keys()}
        
        pbar = tqdm(self.val_loader, desc=f"Val Epoch {epoch + 1}")
        for batch in pbar:
            # Validate each model
            batch_losses = self._validate_batch(batch)
            
            for key, value in batch_losses.items():
                if key in val_losses:
                    val_losses[key].append(value)
            
            pbar.set_postfix({k: f"{v[-1]:.4f}" for k, v in val_losses.items() if v})
        
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
            
            batch_size = images.shape[0]
            class_names = list(self.dataset_info['class_mapping'].values())
            criterion = self.losses['detection']
            
            outputs = self.models['grounding_dino'](images, class_names=class_names)
            
            # Prepare targets in DETR format: list of dicts
            targets = []
            for b in range(batch_size):
                valid_mask = labels[b] != -1
                valid_labels = labels[b][valid_mask]
                valid_boxes = boxes[b][valid_mask]
                
                targets.append({
                    'labels': valid_labels,
                    'boxes': valid_boxes,
                })
            
            # Compute loss with auxiliary losses
            loss_dict = criterion(outputs, targets)
            
            # Compute total weighted loss
            total_loss = sum(loss_dict[k] * criterion.weight_dict[k] 
                           for k in loss_dict.keys() 
                           if k in criterion.weight_dict)
            
            batch_losses['grounding_dino_loss'] = total_loss.item()
        
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
    
    def _resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        
        for model_name in self.models.keys():
            ckpt_manager = self.checkpoint_managers[model_name]
            checkpoint = ckpt_manager.load_checkpoint(
                checkpoint_path,
                model=self.models[model_name],
                optimizer=self.optimizers[model_name],
                scheduler=self.schedulers.get(model_name)
            )
            
            # Update training manager state
            if 'training_manager_state' in checkpoint:
                self.training_managers[model_name].load_state_dict(
                    checkpoint['training_manager_state']
                )

