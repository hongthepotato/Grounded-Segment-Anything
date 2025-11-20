"""
Training Manager for gradient handling, mixed precision, and training dynamics.

This module provides a centralized manager for:
- Automatic Mixed Precision (AMP) training
- Gradient clipping
- Batch Normalization freezing
- Training statistics tracking
"""

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from typing import Dict, Callable, Optional
import yaml
import logging

logger = logging.getLogger(__name__)


class TrainingManager:
    """
    Manages all training dynamics: gradient clipping, mixed precision, etc.
    
    Why centralized?
    - Consistent training behavior across teacher/student
    - Easy to modify hyperparameters
    - Prevents common training failures
    - Config-driven (no hardcoding!)
    
    Example:
        >>> manager = TrainingManager(
        >>>     model=model,
        >>>     optimizer=optimizer,
        >>>     config_path='configs/defaults/training_dynamics.yaml'
        >>> )
        >>> 
        >>> for batch in dataloader:
        >>>     loss_dict = manager.training_step(
        >>>         batch=batch,
        >>>         compute_loss_fn=my_loss_function
        >>>     )
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config_path: str,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        """
        Args:
            model: The model being trained
            optimizer: The optimizer
            config_path: Path to training dynamics config
            scheduler: Optional learning rate scheduler
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # Setup mixed precision
        self.use_amp = self.config.get('mixed_precision', {}).get('enabled', False)
        if self.use_amp:
            amp_config = self.config['mixed_precision']
            self.scaler = GradScaler(
                init_scale=amp_config.get('init_scale', 65536),
                growth_factor=amp_config.get('growth_factor', 2.0),
                backoff_factor=amp_config.get('backoff_factor', 0.5),
                growth_interval=amp_config.get('growth_interval', 2000),
            )
            logger.info("✓ Automatic Mixed Precision (AMP) enabled")
        else:
            self.scaler = None
            logger.info("AMP disabled - using FP32 training")
        
        # Setup gradient clipping
        self.clip_cfg = self.config.get('gradient_clipping', {})
        self.clip_enabled = self.clip_cfg.get('enabled', False)
        if self.clip_enabled:
            logger.info(f"✓ Gradient clipping enabled (max_norm={self.clip_cfg.get('max_norm')})")
        
        # Setup BN behavior
        self._configure_batch_norm()
        
        # Training statistics
        self.global_step = 0
        self.epoch = 0
    
    def _configure_batch_norm(self):
        """Configure BatchNorm layers based on config."""
        norm_cfg = self.config.get('normalization', {})
        
        # For teacher fine-tuning with LoRA
        if norm_cfg.get('freeze_bn_teacher', False):
            logger.info("Freezing BatchNorm layers for LoRA training")
            for module in self.model.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.eval()  # Keep in eval mode
                    if not norm_cfg.get('track_running_stats_teacher', False):
                        module.track_running_stats = False
                    # Freeze parameters
                    for param in module.parameters():
                        param.requires_grad = False
            logger.info("✓ BatchNorm frozen")
    
    def training_step(
        self,
        batch: Dict,
        compute_loss_fn: Callable,
        accumulation_steps: int = 1
    ) -> Dict[str, float]:
        """
        Execute one training step with proper gradient handling.
        
        Args:
            batch: Input batch
            compute_loss_fn: Function that computes loss given batch
            accumulation_steps: Number of steps to accumulate gradients
        
        Returns:
            Dict with loss and metrics
        
        Example:
            >>> def my_loss_fn(batch):
            >>>     outputs = model(batch['images'])
            >>>     loss = criterion(outputs, batch['targets'])
            >>>     return {'loss': loss, 'acc': compute_acc(outputs, batch['targets'])}
            >>> 
            >>> loss_dict = manager.training_step(batch, my_loss_fn)
        """
        # Zero gradients at the start of accumulation cycle
        if self.global_step % accumulation_steps == 0:
            self.optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with automatic mixed precision
        if self.use_amp:
            with autocast():
                loss_dict = compute_loss_fn(batch)
                loss = loss_dict['loss']
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / accumulation_steps
            
            # Backward pass with gradient scaling
            self.scaler.scale(scaled_loss).backward()
            
            # Update weights after accumulation
            if (self.global_step + 1) % accumulation_steps == 0:
                # Gradient clipping (if enabled)
                if self.clip_enabled:
                    self.scaler.unscale_(self.optimizer)  # Unscale before clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.clip_cfg['max_norm'],
                        norm_type=self.clip_cfg.get('norm_type', 2.0),
                        error_if_nonfinite=self.clip_cfg.get('error_if_nonfinite', True)
                    )
                    loss_dict['grad_norm'] = grad_norm.item()
                
                # Optimizer step with gradient scaling
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()
        
        else:
            # Standard training without AMP
            loss_dict = compute_loss_fn(batch)
            loss = loss_dict['loss']
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()
            
            # Update weights after accumulation
            if (self.global_step + 1) % accumulation_steps == 0:
                # Gradient clipping (if enabled)
                if self.clip_enabled:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.clip_cfg['max_norm'],
                        norm_type=self.clip_cfg.get('norm_type', 2.0)
                    )
                    loss_dict['grad_norm'] = grad_norm.item()
                
                self.optimizer.step()
                
                # Scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()
        
        self.global_step += 1
        
        # Add learning rate to metrics
        loss_dict['lr'] = self.optimizer.param_groups[0]['lr']
        
        return loss_dict
    
    def get_grad_statistics(self) -> Dict[str, float]:
        """
        Get gradient statistics for monitoring.
        
        Returns:
            Dictionary with gradient statistics
        """
        stats = {
            'grad_norm': [],
            'grad_max': [],
            'grad_mean': [],
        }
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()
                stats['grad_norm'].append(grad.norm().item())
                stats['grad_max'].append(grad.abs().max().item())
                stats['grad_mean'].append(grad.abs().mean().item())
        
        if stats['grad_norm']:
            return {
                'total_grad_norm': sum(stats['grad_norm']),
                'max_grad_value': max(stats['grad_max']),
                'mean_grad_value': sum(stats['grad_mean']) / len(stats['grad_mean']),
            }
        return {}
    
    def set_epoch(self, epoch: int) -> None:
        """Set current epoch number."""
        self.epoch = epoch
    
    def get_state_dict(self) -> Dict:
        """
        Get state dictionary for checkpointing.
        
        Returns:
            State dictionary containing optimizer, scheduler, scaler states
        """
        state = {
            'optimizer': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch
        }
        
        if self.scheduler is not None:
            state['scheduler'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()
        
        return state
    
    def load_state_dict(self, state: Dict) -> None:
        """
        Load state dictionary from checkpoint.
        
        Args:
            state: State dictionary
        """
        self.optimizer.load_state_dict(state['optimizer'])
        self.global_step = state.get('global_step', 0)
        self.epoch = state.get('epoch', 0)
        
        if self.scheduler is not None and 'scheduler' in state:
            self.scheduler.load_state_dict(state['scheduler'])
        
        if self.scaler is not None and 'scaler' in state:
            self.scaler.load_state_dict(state['scaler'])
        
        logger.info(f"✓ Loaded training state (epoch={self.epoch}, step={self.global_step})")
