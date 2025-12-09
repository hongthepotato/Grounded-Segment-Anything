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
from ml_engine.models.teacher.grounding_dino_lora import load_grounding_dino_with_lora
from ml_engine.models.teacher.sam_lora import load_sam_with_lora
from ml_engine.training.losses import build_criterion, SegmentationLoss
from ml_engine.training.training_manager import TrainingManager
from ml_engine.training.checkpoint_manager import CheckpointManager
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

        # Job manager integration - callbacks for progress reporting and cancellation
        self.progress_callback = progress_callback
        self.cancel_check = cancel_check

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
            image_dir=str(self.data_manager.image_dir),
            dataset_info=self.dataset_info,
            model_names=self.required_models,
            augmentation_config=self.config['augmentation'],
            is_training=True
        )

        # Create validation dataset without augmentation
        self.val_dataset = DatasetFactory.create_dataset(
            coco_data=val_data,
            image_dir=str(self.data_manager.image_dir),
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

            logger.info("✓ Grounding DINO criterion with Hungarian matching initialized")
            logger.info("  - Num classes: %s", num_classes)
            logger.info("  - Num decoder layers: %s", num_decoder_layers)
            logger.info("  - Auxiliary losses: %s intermediate + 1 encoder", num_decoder_layers - 1)

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

            logger.info("✓ Optimizer and scheduler created for %s", model_name)

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

        finally:
            # Always close loggers
            for tb_logger in self.tb_loggers.values():
                tb_logger.close()

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

        # Training loop
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch + 1}")
        for step, batch in enumerate(pbar):
            # Check for cancellation at each batch
            if self.cancel_check and self.cancel_check():
                logger.info("Training cancelled during epoch %d, step %d", epoch + 1, step)
                raise TrainingCancelledException("Training cancelled by user")

            batch_losses = self._train_batch(batch)

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

            # # Diagnose valid token count
            # print("=" * 80)
            # print("TOKEN DIAGNOSTICS")
            # print("=" * 80)

            # # Get valid token mask from first query (all queries share same text encoding)
            # first_query_logits = outputs['pred_logits'][0, 0, :]  # [num_text_tokens]
            # valid_token_mask = ~torch.isinf(first_query_logits)
            # num_valid_tokens = valid_token_mask.sum().item()
            # total_tokens = first_query_logits.shape[0]

            # print(f"Text caption: '{' . '.join(class_names)}'")
            # print(f"Number of classes: {len(class_names)}")
            # print(f"Total token positions (with padding): {total_tokens}")
            # print(f"Valid tokens (non -inf): {num_valid_tokens}")
            # print(f"Padding tokens (-inf): {total_tokens - num_valid_tokens}")
            # print(f"Valid token positions: {valid_token_mask.nonzero().squeeze().tolist()}")

            # # Show first few logits values
            # print(f"\nFirst 20 logit values (from first query):")
            # for i in range(min(20, total_tokens)):
            #     val = first_query_logits[i].item()
            #     status = "VALID" if not torch.isinf(first_query_logits[i]) else "-INF"
            #     print(f"  Token {i:3d}: {val:8.4f}  [{status}]")

            # # Show what token ranges each class should occupy
            # if num_valid_tokens > 0:
            #     tokens_per_class = max(1, num_valid_tokens // len(class_names))
            #     print(f"\nRecommended token mapping (equally distributed):")
            #     for class_id, class_name in enumerate(class_names):
            #         start_idx = class_id * tokens_per_class
            #         end_idx = min(start_idx + tokens_per_class, num_valid_tokens)
            #         print(f"  Class {class_id} '{class_name}': tokens [{start_idx}:{end_idx}] (total: {end_idx - start_idx})")

            # print("=" * 80)

            # Ensure model returns auxiliary outputs
            # if 'aux_outputs' not in outputs:
            #     logger.warning("Model not returning auxiliary outputs! Training may be suboptimal.")

            # # Debug: check model outputs
            # if torch.isnan(outputs['pred_logits']).any():
            #     logger.error("NaN detected in pred_logits!")
            # if torch.isnan(outputs['pred_boxes']).any():
            #     logger.error("NaN detected in pred_boxes!")

            # ============================================================
            # OFFICIAL TOKEN MAPPING: Use Grounding DINO's built-in utilities
            # ============================================================

            # Step 1: Build caption and character-level token spans using official utility
            # This formats the caption exactly as Grounding DINO expects: "class1 . class2 . class3"
            caption, cat2tokenspan = build_captions_and_token_span(
                class_names, 
                force_lowercase=False
            )

            # print(f"\nOFFICIAL TOKEN MAPPING:")
            # print(f"  Caption: '{caption}'")
            # print(f"  Character spans per category:")
            # for cat_name, spans in cat2tokenspan.items():
            #     print(f"    '{cat_name}': {spans}")

            # # Step 2: Tokenize using the model's tokenizer (same as model uses internally)
            tokenized = model.tokenizer(
                caption,
                padding="longest",
                return_tensors="pt"
            ).to(self.device)

            # print(f"  Tokenized input_ids: {tokenized['input_ids'][0].tolist()}")
            # print(f"  Tokenized length: {tokenized['input_ids'].shape[1]}")

            # Step 3: Build class_id -> token_span mapping
            # Map from class_id (integer) to character spans
            class_id_to_name = {i: name for i, name in enumerate(class_names)}
            token_span_per_class = []
            for class_id in range(len(class_names)):
                class_name = class_id_to_name[class_id]
                if class_name in cat2tokenspan:
                    token_span_per_class.append(cat2tokenspan[class_name])
                else:
                    logger.warning(f"Class '{class_name}' not found in cat2tokenspan!")
                    token_span_per_class.append([])  # Empty span as fallback

            # Step 4: Create positive map using official utility
            # This converts character spans to actual BERT token positions
            # Returns: [num_classes, max_text_len] with 1.0 at relevant token positions
            positive_map = create_positive_map_from_span(
                tokenized, 
                token_span_per_class,
                max_text_len=outputs['pred_logits'].shape[-1]  # 256
            ).to(self.device)

            # print(f"  Positive map shape: {positive_map.shape}")
            # for class_id, class_name in enumerate(class_names):
            #     active_tokens = (positive_map[class_id] > 0).nonzero().squeeze()
            #     print(f"  Class {class_id} '{class_name}' → tokens {active_tokens.tolist()}")
            # print("=" * 80)

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
                    if cat_id in cat_id_to_idx:
                        class_idx = cat_id_to_idx[cat_id]  # Convert to 0-based index
                        token_labels[obj_idx] = positive_map[class_idx]
                    else:
                        logger.warning(f"Unknown category_id {cat_id} not in category_id_to_index mapping")

                # Create target dict for this batch element
                targets.append({
                    'labels': valid_labels,        # [num_valid_objs] - class IDs
                    'boxes': valid_boxes,          # [num_valid_objs, 4] in normalized [cx, cy, w, h]
                    'token_labels': token_labels,  # [num_valid_objs, max_text_len] - proper token labels
                })

            # DEBUG: Print target structure before loss computation
            # print("\n" + "=" * 80)
            # print("TARGET STRUCTURE DEBUG")
            # print("=" * 80)
            # print(f"Number of targets (batch elements): {len(targets)}")
            # for batch_idx, target in enumerate(targets):
            #     print(f"\nBatch {batch_idx}:")
            #     print(f"  - num_objects: {len(target['labels'])}")
            #     print(f"  - labels: {target['labels']}")
            #     print(f"  - boxes shape: {target['boxes'].shape}")
            #     print(f"  - token_labels shape: {target['token_labels'].shape}")
            #     print(f"  - token_labels nonzero count: {(target['token_labels'] > 0).sum().item()}")

            #     # Show token_labels for first object
            #     if len(target['labels']) > 0:
            #         first_obj_tokens = target['token_labels'][0]
            #         nonzero_positions = (first_obj_tokens > 0).nonzero().squeeze()
            #         print(f"  - First object token_labels nonzero at: {nonzero_positions.tolist() if nonzero_positions.numel() > 0 else 'none'}")
            # print("=" * 80)

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
        
        pbar = tqdm(self.val_loader, desc=f"Val Epoch {epoch + 1}")
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
            # Get batch info
            file_names = batch['file_names']
            image_sizes = batch['image_sizes']  # List of (width, height)
            
            # Get preprocessed data
            dino_data = batch['preprocessed']['grounding_dino']
            images_tensor = dino_data['images'].to(self.device)
            gt_boxes = dino_data['boxes'].to(self.device)  # [B, max_obj, 4] normalized cxcywh
            labels = dino_data['labels'].to(self.device)   # [B, max_obj]
            
            # Get predictions
            class_names = list(self.dataset_info['class_mapping'].values())
            model = self.models['grounding_dino']
            outputs = model(images_tensor, class_names=class_names)
            
            pred_boxes = outputs['pred_boxes']  # [B, num_queries, 4] normalized cxcywh
            pred_logits = outputs['pred_logits']  # [B, num_queries, num_tokens]
            
            batch_size = len(file_names)
            images_list = []
            predictions_list = {'boxes': [], 'labels': []}
            targets_list = {'boxes': [], 'labels': []}
            
            for b in range(min(batch_size, 8)):  # Max 8 samples
                # Load original image
                img_path = self.data_manager.image_dir / file_names[b]
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
                gt_label = np.array([cat_id_to_idx.get(int(cat_id), -1) for cat_id in gt_label_raw])
                
                # Debug: print raw normalized GT boxes
                print(f"\nGround Truth boxes (raw normalized cxcywh):")
                for i, box in enumerate(gt_box[:5]):
                    print(f"  GT Box {i}: cxcywh={box}, cat_id={gt_label_raw[i]}, class_idx={gt_label[i]}")
                
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
                
                # Convert pred boxes: take top-k by confidence
                # Note: pred_logits is [num_queries, num_tokens], not [num_queries, num_classes]
                # Need to map token probabilities to class probabilities
                pred_box = pred_boxes[b].cpu().numpy()  # [num_queries, 4]
                pred_logit = pred_logits[b].sigmoid()  # [num_queries, num_tokens]
                
                # Build positive_map for token->class mapping
                from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span
                caption, cat2tokenspan = build_captions_and_token_span(class_names, force_lowercase=False)
                tokenized = model.tokenizer(caption, padding="longest", return_tensors="pt").to(self.device)
                
                token_span_per_class = []
                for class_id in range(len(class_names)):
                    class_name = class_names[class_id]
                    if class_name in cat2tokenspan:
                        token_span_per_class.append(cat2tokenspan[class_name])
                    else:
                        token_span_per_class.append([])
                
                positive_map = create_positive_map_from_span(
                    tokenized, token_span_per_class, max_text_len=pred_logit.shape[-1]
                ).to(self.device)  # [num_classes, num_tokens]
                
                # For each query, compute class score as MEAN of token probs for that class
                # This matches mmdetection's convert_grounding_to_cls_scores approach
                num_classes = positive_map.shape[0]
                class_probs = torch.zeros(pred_logit.shape[0], num_classes, device=self.device)
                for c in range(num_classes):
                    token_mask = positive_map[c] > 0  # which tokens belong to class c
                    if token_mask.sum() > 0:
                        class_probs[:, c] = pred_logit[:, token_mask].mean(dim=-1)
                
                pred_score = class_probs.max(dim=-1)[0].cpu().numpy()  # [num_queries]
                pred_class = class_probs.max(dim=-1)[1].cpu().numpy()  # [num_queries] - actual class IDs
                
                # Get metadata for size comparison
                metadata = dino_data['metadata'][b] if 'metadata' in dino_data else {}
                final_size = metadata.get('final_size', 'N/A')
                original_size = metadata.get('original_size', 'N/A')
                
                # Debug: print prediction info
                print(f"\n{'='*60}")
                print(f"[DEBUG] Visualization - Sample {b}")
                print(f"{'='*60}")
                print(f"Image: {file_names[b]}")
                print(f"Original image size (from file): {img_w}x{img_h}")
                print(f"Metadata original_size (H,W): {original_size}")
                print(f"Metadata final_size (H,W): {final_size}")
                print(f"Class names: {class_names}")
                print(f"Num queries: {pred_box.shape[0]}, Num classes: {num_classes}")
                print(f"\nTop 5 predictions (before threshold):")
                top5_indices = np.argsort(pred_score)[-5:][::-1]
                for rank, idx in enumerate(top5_indices):
                    cls_id = pred_class[idx]
                    cls_name = class_names[cls_id] if cls_id < len(class_names) else f'unk_{cls_id}'
                    box = pred_box[idx]  # normalized cxcywh
                    print(f"  #{rank+1}: class={cls_id}({cls_name}), score={pred_score[idx]:.4f}, box_cxcywh={box}")
                
                # Filter by confidence threshold
                conf_thresh = 0.3
                keep = pred_score > conf_thresh
                num_kept = keep.sum()
                print(f"Confidence threshold: {conf_thresh}")
                print(f"Boxes kept: {num_kept} / {len(pred_score)}")
                
                pred_box = pred_box[keep]
                pred_label = pred_class[keep]
                pred_score_kept = pred_score[keep]
                
                if len(pred_box) > 0:
                    print(f"\nKept boxes (normalized cxcywh):")
                    for i, (box, score, lbl) in enumerate(zip(pred_box[:5], pred_score_kept[:5], pred_label[:5])):
                        print(f"  Box {i}: cxcywh={box}, score={score:.4f}, class={lbl}")
                    
                    # cxcywh to xyxy
                    cx, cy, w, h = pred_box[:, 0], pred_box[:, 1], pred_box[:, 2], pred_box[:, 3]
                    pred_box_xyxy = np.stack([
                        (cx - w/2) * img_w,
                        (cy - h/2) * img_h,
                        (cx + w/2) * img_w,
                        (cy + h/2) * img_h
                    ], axis=1)
                    
                    print(f"\nConverted boxes (pixel xyxy):")
                    for i, box in enumerate(pred_box_xyxy[:5]):
                        print(f"  Box {i}: xyxy={box}")
                else:
                    pred_box_xyxy = np.array([])
                    pred_label = np.array([])
                    print("No boxes passed threshold!")
                
                # Also print GT for comparison
                print(f"\nGround Truth boxes: {len(gt_box_xyxy)}")
                if len(gt_box_xyxy) > 0:
                    for i, (box, lbl, raw_id) in enumerate(zip(gt_box_xyxy[:5], gt_label[:5], gt_label_raw[:5])):
                        cls_name = class_name[lbl] if 0 <= lbl < len(class_name) else f'unk_{lbl}'
                        print(f"  GT Box {i}: xyxy={box}, cat_id={raw_id}, class_idx={lbl}({cls_name})")
                print(f"{'='*60}\n")
                
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
