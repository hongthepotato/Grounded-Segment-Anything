"""
Training Manager for gradient handling and mixed precision.

This module provides:
- Automatic Mixed Precision (AMP) training
- Gradient clipping
- Gradient accumulation
- Batch Normalization freezing for LoRA
"""

import logging
from typing import Dict, Callable, Optional

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import yaml

logger = logging.getLogger(__name__)


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base. Returns a new dict."""
    result = dict(base)
    for k, v in overrides.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


class TrainingManager:
    """
    Manages training dynamics: AMP, gradient clipping, accumulation.
    
    Example:
        >>> manager = TrainingManager(
        >>>     model=model,
        >>>     optimizer=optimizer,
        >>>     config_path='configs/defaults/training_dynamics.yaml'
        >>> )
        >>> 
        >>> for batch in dataloader:
        >>>     loss_dict = manager.training_step(batch, compute_loss_fn)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config_path: str,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        config_overrides: Optional[Dict] = None,
    ):
        """
        Args:
            model: The model being trained
            optimizer: The optimizer
            config_path: Path to training dynamics config YAML
            scheduler: Optional learning rate scheduler
            config_overrides: Dict of training_dynamics overrides from experiment loop.
                Deep-merged over the YAML values after loading. Keys match the YAML
                structure under the top-level 'training_dynamics' key, e.g.
                {'gradient_clipping': {'max_norm': 0.5}, 'mixed_precision': {'enabled': False}}.
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Deep-merge experiment-loop overrides over YAML defaults
        if config_overrides:
            td_overrides = config_overrides.get('training_dynamics', config_overrides)
            config = _deep_merge(config.get('training_dynamics', config), td_overrides)
        else:
            config = config.get('training_dynamics', config)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Mixed precision (use default scaler params)
        self.use_amp = config.get('mixed_precision', {}).get('enabled', False)
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            logger.info("AMP enabled")
        
        # Gradient clipping
        clip_cfg = config.get('gradient_clipping', {})
        self.clip_enabled = clip_cfg.get('enabled', False)
        self.clip_max_norm = clip_cfg.get('max_norm', 1.0)
        self.clip_norm_type = clip_cfg.get('norm_type', 2.0)
        if self.clip_enabled:
            logger.info("Gradient clipping enabled (max_norm=%s)", self.clip_max_norm)
        
        # Freeze BatchNorm for LoRA training
        norm_cfg = config.get('normalization', {})
        if norm_cfg.get('freeze_bn_teacher', False):
            self._freeze_batch_norm()
        
        # Step counter for gradient accumulation
        self.global_step = 0
    
    def _freeze_batch_norm(self) -> None:
        """Freeze BatchNorm layers for LoRA training."""
        logger.info("Freezing BatchNorm layers for LoRA training")
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()
                module.track_running_stats = False
                for param in module.parameters():
                    param.requires_grad = False
        logger.info("BatchNorm frozen")
    
    def training_step(
        self,
        batch: Dict,
        compute_loss_fn: Callable,
        accumulation_steps: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Execute one training step with AMP and gradient clipping.
        
        Args:
            batch: Input batch
            compute_loss_fn: Function that computes loss given batch
            accumulation_steps: Number of steps to accumulate gradients
        
        Returns:
            Dict with loss and metrics
        """
        # Zero gradients at start of accumulation cycle
        if self.global_step % accumulation_steps == 0:
            self.optimizer.zero_grad(set_to_none=True)
        
        if self.use_amp:
            with autocast(device_type='cuda'):
                loss_dict = compute_loss_fn(batch)
                loss = loss_dict['loss']
            
            # Scale loss for accumulation
            scaled_loss = loss / accumulation_steps
            self.scaler.scale(scaled_loss).backward()
            
            # Update weights after accumulation
            if (self.global_step + 1) % accumulation_steps == 0:
                if self.clip_enabled:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.clip_max_norm,
                        norm_type=self.clip_norm_type
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                if self.scheduler is not None:
                    self.scheduler.step()
        else:
            loss_dict = compute_loss_fn(batch)
            loss = loss_dict['loss']
            
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()
            
            if (self.global_step + 1) % accumulation_steps == 0:
                if self.clip_enabled:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.clip_max_norm,
                        norm_type=self.clip_norm_type
                    )
                
                self.optimizer.step()
                
                if self.scheduler is not None:
                    self.scheduler.step()
        
        self.global_step += 1
        loss_dict['lr'] = self.optimizer.param_groups[0]['lr']
        
        return loss_dict
