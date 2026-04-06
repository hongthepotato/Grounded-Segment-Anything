"""
Grounding DINO Trainer for object detection with LoRA.

This module provides the trainer for fine-tuning Grounding DINO
using LoRA (Low-Rank Adaptation) for memory-efficient training.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ml_engine.models.teacher.grounding_dino_lora import load_grounding_dino_with_lora
from ml_engine.training.config_types import GroundingDINOConfig, LoopConfig
from ml_engine.training.losses import build_criterion
from ml_engine.training.dino_utils import build_positive_map, build_detr_targets
from .base import BaseModelTrainer

logger = logging.getLogger(__name__)


class GroundingDINOTrainer(BaseModelTrainer):
    """
    Trainer for Grounding DINO with LoRA.
    
    Handles:
    - Model loading with LoRA adapters
    - Hungarian matching loss computation
    - Token-level classification with positive maps
    - Auxiliary losses from all decoder layers
    
    Example:
        >>> trainer = GroundingDINOTrainer(
        >>>     config=dino_config,
        >>>     device=torch.device('cuda'),
        >>>     output_dir=Path('experiments/exp1'),
        >>>     dataset_info=dataset_info
        >>> )
        >>> loss = trainer.train_batch(batch)
    """
    
    model_name = "grounding_dino"
    
    def __init__(
        self,
        config: GroundingDINOConfig,
        loop: LoopConfig,
        device: torch.device,
        output_dir: Path,
        dataset_info: Dict[str, Any]
    ):
        """
        Initialize the Grounding DINO trainer.

        Args:
            config: Typed Grounding DINO model configuration
            loop: Shared training loop config (epochs, optimizer settings)
            device: Device to train on
            output_dir: Root output directory
            dataset_info: Dataset metadata with class_mapping, category_id_to_index
        """
        # Store class names before calling super().__init__
        self.class_names = list(dataset_info['class_mapping'].values())
        self.category_id_to_index = dataset_info['category_id_to_index']

        # Cache for positive map (computed once)
        self._positive_map_cache: Optional[torch.Tensor] = None
        self._positive_map_max_len: Optional[int] = None

        super().__init__(config, loop, device, output_dir, dataset_info)
    
    def _load_model(self) -> nn.Module:
        """Load Grounding DINO with LoRA adapters."""
        model = load_grounding_dino_with_lora(
            base_checkpoint=self.config.base_checkpoint,
            lora_config=self.config.lora.to_peft_dict(),
            freeze_backbone=self.config.freeze_backbone,
            freeze_bbox_embed=self.config.freeze_bbox_embed,
            bert_model_path=self.config.bert_model_path,
        )
        return model
    
    def _create_criterion(self) -> nn.Module:
        """Create Hungarian matching criterion with auxiliary losses."""
        num_classes = self.dataset_info['num_classes']
        
        # Get number of decoder layers from model architecture
        base_model = self.model.model.model  # Unwrap PEFT wrapper
        num_decoder_layers = base_model.transformer.decoder.num_layers
        
        criterion = build_criterion(
            num_classes=num_classes,
            num_decoder_layers=num_decoder_layers,
            focal_alpha=0.25,
            focal_gamma=2.0
        )
        
        logger.info(f"  Criterion: Hungarian matching with {num_decoder_layers} decoder layers")
        logger.info(f"  Num classes: {num_classes}")
        
        return criterion
    
    def _get_positive_map(self, max_text_len: int) -> torch.Tensor:
        """Get cached positive map or build a new one."""
        if self._positive_map_cache is None or self._positive_map_max_len != max_text_len:
            self._positive_map_cache = build_positive_map(
                tokenizer=self.model.tokenizer,
                class_names=self.class_names,
                max_text_len=max_text_len,
                device=self.device
            )
            self._positive_map_max_len = max_text_len
        return self._positive_map_cache
    
    def compute_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Compute loss for a batch with Hungarian matching.
        
        Args:
            batch: Batch with preprocessed data containing:
                - preprocessed['grounding_dino']['images']: NestedTensor
                - preprocessed['grounding_dino']['boxes']: [B, max_objs, 4]
                - preprocessed['grounding_dino']['labels']: [B, max_objs]
        
        Returns:
            Dict with 'loss' and individual loss components
        """
        # Get preprocessed data
        dino_data = batch['preprocessed']['grounding_dino']
        images = dino_data['images'].to(self.device)
        boxes = dino_data['boxes'].to(self.device)
        labels = dino_data['labels'].to(self.device)
        
        batch_size = labels.shape[0]
        
        # Check for valid data
        if batch_size == 0:
            logger.error("Empty batch received!")
            return {'loss': torch.tensor(0.0, device=self.device, requires_grad=True)}
        
        total_valid_objs = (labels != -1).sum().item()
        if total_valid_objs == 0:
            logger.warning("Batch has no valid objects! Skipping...")
            return {'loss': torch.tensor(0.0, device=self.device, requires_grad=True)}
        
        # Forward pass
        outputs = self.model(images, class_names=self.class_names)
        
        # Get or build positive map (cached)
        max_text_len = outputs['pred_logits'].shape[-1]
        positive_map = self._get_positive_map(max_text_len)
        
        # Build targets in DETR format
        targets = build_detr_targets(
            boxes=boxes,
            labels=labels,
            positive_map=positive_map,
            category_id_to_index=self.category_id_to_index,
            device=self.device
        )
        
        # Compute loss with Hungarian matching
        loss_dict = self.criterion(outputs, targets)
        
        # Check for NaN
        for k, v in loss_dict.items():
            if torch.isnan(v).any():
                logger.error(f"NaN detected in {k}: {v}")
        
        # Compute total weighted loss
        total_loss = sum(
            loss_dict[k] * self.criterion.weight_dict[k]
            for k in loss_dict.keys()
            if k in self.criterion.weight_dict
        )
        
        if torch.isnan(total_loss):
            logger.error("NaN in total_loss!")
            logger.error(f"Loss components: {loss_dict}")
            logger.error(f"Valid objects: {total_valid_objs}")
        
        # Return dict with total loss and components
        result = {'loss': total_loss, 'total_loss': total_loss.detach()}
        result.update({k: v.detach() for k, v in loss_dict.items()})
        
        return result
    
    def get_predictions(
        self,
        batch: Dict[str, Any],
        confidence_threshold: float = 0.3
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Get model predictions for visualization/evaluation.
        
        Args:
            batch: Batch with preprocessed data
            confidence_threshold: Minimum confidence for predictions
        
        Returns:
            List of prediction dicts with boxes, scores, labels
        """
        self.model.eval()
        
        dino_data = batch['preprocessed']['grounding_dino']
        images = dino_data['images'].to(self.device)
        
        with torch.no_grad():
            predictions = self.model.predict(
                images,
                self.class_names,
                confidence_threshold=confidence_threshold
            )
        
        return predictions
