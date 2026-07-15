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
        job_id: str,
        config: Dict[str, Any],
        device: torch.device,
        output_dir: Path,
        dataset_info: Dict[str, Any],
    ):
        """
        Initialize the SAM trainer.

        Args:
            job_id: Lineage id forwarded from the parent Trainer (TODO #16).
            config: Model configuration with keys:
                - model.base_checkpoint: Path to pretrained weights
                - model.model_type: Model type (vit_h, vit_l, vit_b)
                - lora: LoRA configuration
                - image_encoder_mode: 'lora' or 'frozen'
                - prompt_encoder_mode: 'frozen' or 'full'
                - mask_decoder_mode: 'full' or 'frozen'
                - learning_rate: Learning rate
            device: Device to train on
            output_dir: Root output directory
            dataset_info: Dataset metadata
        """
        super().__init__(job_id, config, device, output_dir, dataset_info)

    def _load_model(self) -> nn.Module:
        """Load SAM-HQ with LoRA adapters."""
        model_section = self.config.get("model", {})
        base_ckpt = model_section.get("base_checkpoint", "data/models/pretrained/sam_vit_h_4b8939.pth")
        model_type = model_section.get("model_type", "vit_h")

        # LoRA config is required
        if "lora" not in self.config:
            raise ValueError(
                "LoRA training requires 'lora' config!\n"
                "Expected keys: r, lora_alpha, target_modules, lora_dropout"
            )

        # Get training modes
        image_encoder_mode = self.config.get("image_encoder_mode", "lora")
        prompt_encoder_mode = self.config.get("prompt_encoder_mode", "frozen")
        mask_decoder_mode = self.config.get("mask_decoder_mode", "full")

        model = load_sam_hq_with_lora(
            base_checkpoint=base_ckpt,
            model_type=model_type,
            lora_config=self.config["lora"],
            image_encoder_mode=image_encoder_mode,
            prompt_encoder_mode=prompt_encoder_mode,
            mask_decoder_mode=mask_decoder_mode,
        )

        logger.info(
            "  Modes: encoder=%s, prompt=%s, decoder=%s",
            image_encoder_mode,
            prompt_encoder_mode,
            mask_decoder_mode,
        )

        return model

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create AdamW with differential LR: mask decoder at a lower rate than LoRA adapters.

        mask_decoder_mode='full' means all mask-decoder weights are trainable, so they
        need a lower LR to avoid overwriting pretrained representations. LoRA adapters
        start from zero and can tolerate the full base LR.
        """
        lr = self.config.get("learning_rate", 1e-4)
        weight_decay = self.config.get("weight_decay", 1e-4)
        multiplier = self.config.get("mask_decoder_lr_multiplier", 1.0)
        mask_decoder_lr = lr * multiplier

        mask_decoder_params = []
        lora_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "mask_decoder" in name:
                mask_decoder_params.append(param)
            else:
                lora_params.append(param)

        param_groups = []
        if lora_params:
            param_groups.append({"params": lora_params, "lr": lr, "name": "lora"})
        if mask_decoder_params:
            param_groups.append(
                {"params": mask_decoder_params, "lr": mask_decoder_lr, "name": "mask_decoder"}
            )

        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        logger.info(
            "  Optimizer: AdamW — lora %d params (lr=%s), mask_decoder %d params (lr=%s)",
            len(lora_params),
            lr,
            len(mask_decoder_params),
            mask_decoder_lr,
        )
        return optimizer

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
        sam_data = batch["preprocessed"]["sam"]
        images = sam_data["images"].to(self.device)
        boxes = sam_data["boxes"].to(self.device)
        masks = sam_data["masks"].to(self.device)
        labels = sam_data["labels"].to(self.device)

        # Create validity mask: collate_fn always puts valid objects first,
        # so valid_mask is True at indices 0..n_valid-1 per image.
        valid_mask = labels != -1  # [B, max_objs]

        # Check for valid data
        if not valid_mask.any():
            logger.warning("Batch has no valid objects! Skipping...")
            return {"loss": torch.tensor(0.0, device=self.device, requires_grad=True)}

        # Trim to max_valid: avoids running the mask decoder on padding entries.
        # With max_objs=10 and 3 real objects, this cuts 7 wasted decoder calls per image.
        max_valid = int(valid_mask.sum(dim=1).max().item())
        max_valid = max(max_valid, 1)
        boxes = boxes[:, :max_valid, :]  # [B, max_valid, 4]
        masks = masks[:, :max_valid, :]  # [B, max_valid, H, W]
        valid_mask = valid_mask[:, :max_valid]  # [B, max_valid]

        # Forward pass (boxes already in correct xyxy format from SAM preprocessing)
        outputs = self.model(images, box_prompts=boxes)

        # Prepare targets
        targets = {"masks": masks, "valid_mask": valid_mask}

        # Compute loss
        loss_dict = self.criterion(outputs, targets)

        # Add total_loss for consistency with DINO trainer
        loss_dict["total_loss"] = loss_dict["loss"].detach()

        return loss_dict
