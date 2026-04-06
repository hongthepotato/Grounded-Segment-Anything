"""
SAM Trainer for instance segmentation with LoRA.

This module provides the trainer for fine-tuning SAM-HQ
using LoRA (Low-Rank Adaptation) for memory-efficient training.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

from ml_engine.models.teacher.sam_lora import load_sam_hq_with_lora
from ml_engine.training.config_types import LoopConfig, SAMConfig
from ml_engine.training.losses import SegmentationLoss
from .base import BaseModelTrainer

logger = logging.getLogger(__name__)


class SAMTrainer(BaseModelTrainer):
    """
    Trainer for SAM-HQ with LoRA.
    
    Handles:
    - Model loading with LoRA adapters
    - Segmentation loss (focal + dice + IoU)
    - Box-prompted mask prediction
    
    Example:
        >>> trainer = SAMTrainer(
        >>>     config=sam_config,
        >>>     device=torch.device('cuda'),
        >>>     output_dir=Path('experiments/exp1'),
        >>>     dataset_info=dataset_info
        >>> )
        >>> loss = trainer.train_batch(batch)
    """
    
    model_name = "sam"
    
    def __init__(
        self,
        config: SAMConfig,
        loop: LoopConfig,
        device: torch.device,
        output_dir: Path,
        dataset_info: Dict[str, Any]
    ):
        """
        Initialize the SAM trainer.

        Args:
            config: Typed SAM model configuration
            loop: Shared training loop config (epochs, optimizer settings)
            device: Device to train on
            output_dir: Root output directory
            dataset_info: Dataset metadata
        """
        super().__init__(config, loop, device, output_dir, dataset_info)

    def _load_model(self) -> nn.Module:
        """Load SAM-HQ with LoRA adapters."""
        model = load_sam_hq_with_lora(
            base_checkpoint=self.config.base_checkpoint,
            model_type=self.config.model_type,
            lora_config=self.config.lora.to_peft_dict(),
            image_encoder_mode=self.config.image_encoder_mode,
            prompt_encoder_mode=self.config.prompt_encoder_mode,
            mask_decoder_mode=self.config.mask_decoder_mode,
        )
        logger.info("  Modes: encoder=%s, prompt=%s, decoder=%s",
                    self.config.image_encoder_mode,
                    self.config.prompt_encoder_mode,
                    self.config.mask_decoder_mode)
        return model
    
    def _create_criterion(self) -> nn.Module:
        """Create segmentation loss (focal + dice + IoU)."""
        criterion = SegmentationLoss()
        logger.info("  Criterion: SegmentationLoss (focal + dice + IoU)")
        return criterion
    
    def compute_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Compute segmentation loss for a batch.
        
        Args:
            batch: Batch with preprocessed data containing:
                - preprocessed['sam']['images']: [B, 3, 1024, 1024]
                - preprocessed['sam']['boxes']: [B, max_objs, 4] in SAM xyxy format
                - preprocessed['sam']['masks']: [B, max_objs, 256, 256]
                - preprocessed['sam']['labels']: [B, max_objs]
        
        Returns:
            Dict with 'loss' and individual loss components
        """
        # Get preprocessed data
        sam_data = batch['preprocessed']['sam']
        images = sam_data['images'].to(self.device)
        boxes = sam_data['boxes'].to(self.device)
        masks = sam_data['masks'].to(self.device)
        labels = sam_data['labels'].to(self.device)
        
        # Create validity mask
        valid_mask = (labels != -1)
        
        # Check for valid data
        if not valid_mask.any():
            logger.warning("Batch has no valid objects! Skipping...")
            return {'loss': torch.tensor(0.0, device=self.device, requires_grad=True)}
        
        # Forward pass (boxes already in correct xyxy format from SAM preprocessing)
        outputs = self.model(images, box_prompts=boxes)
        
        # Prepare targets
        targets = {
            'masks': masks,
            'valid_mask': valid_mask
        }
        
        # Compute loss
        loss_dict = self.criterion(outputs, targets)
        
        # Add total_loss for consistency with DINO trainer
        loss_dict['total_loss'] = loss_dict['loss'].detach()
        
        return loss_dict
