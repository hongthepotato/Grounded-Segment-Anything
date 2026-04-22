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
        config_overrides: Optional[Dict] = None,
    ):
        """
        Args:
            model: The model being trained
            optimizer: The optimizer
            config_path: Path to training dynamics config YAML
            config_overrides: Dict of training_dynamics overrides from experiment loop.
                Deep-merged over the YAML values after loading. Keys match the YAML
                structure under the top-level 'training_dynamics' key, e.g.
                {'gradient_clipping': {'max_norm': 0.5}, 'mixed_precision': {'enabled': False}}.

        Note: LR scheduler stepping is intentionally NOT handled here. TrainingManager
        is a per-batch primitive (AMP, clipping, accumulation). Epoch-level concerns
        like scheduler stepping belong in BaseModelTrainer / Trainer.
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

        # Mixed precision — read dtype from config; gate GradScaler on float16.
        # bfloat16 shares FP32's exponent range, so gradients don't underflow;
        # loss scaling is unnecessary (and unsupported — GradScaler is FP16-only).
        amp_cfg = config.get('mixed_precision', {})
        self.use_amp = amp_cfg.get('enabled', False)
        _dtype_aliases = {
            'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16,
            'float16': torch.float16, 'fp16': torch.float16, 'half': torch.float16,
        }
        dtype_str = str(amp_cfg.get('dtype', 'bfloat16')).lower()
        if dtype_str not in _dtype_aliases:
            raise ValueError(
                f"mixed_precision.dtype must be one of {sorted(_dtype_aliases)}, "
                f"got {dtype_str!r}"
            )
        self.amp_dtype = _dtype_aliases[dtype_str]

        if self.use_amp and self.amp_dtype == torch.float16:
            self.scaler = GradScaler(
                init_scale=amp_cfg.get('init_scale', 65536),
                growth_factor=amp_cfg.get('growth_factor', 2.0),
                backoff_factor=amp_cfg.get('backoff_factor', 0.5),
                growth_interval=amp_cfg.get('growth_interval', 2000),
            )
            logger.info("AMP enabled (dtype=float16, init_scale=%s)",
                        amp_cfg.get('init_scale', 65536))
        elif self.use_amp:
            self.scaler = None
            logger.info("AMP enabled (dtype=%s, no GradScaler)", dtype_str)
        else:
            self.scaler = None

        # Gradient clipping
        clip_cfg = config.get('gradient_clipping', {})
        self.clip_enabled = clip_cfg.get('enabled', False)
        self.clip_max_norm = clip_cfg.get('max_norm', 1.0)
        self.clip_norm_type = clip_cfg.get('norm_type', 2.0)
        self.clip_error_if_nonfinite = clip_cfg.get('error_if_nonfinite', False)
        if self.clip_enabled:
            logger.info(
                "Gradient clipping enabled (max_norm=%s, error_if_nonfinite=%s)",
                self.clip_max_norm, self.clip_error_if_nonfinite
            )

        # Gradient accumulation
        accum_cfg = config.get('gradient_accumulation', {})
        self.accumulation_steps = max(1, accum_cfg.get('steps', 1))
        if self.accumulation_steps > 1:
            logger.info("Gradient accumulation: %d steps (effective batch ×%d)",
                        self.accumulation_steps, self.accumulation_steps)

        # Freeze BatchNorm for LoRA training.
        # _freeze_bn is stored so training_step() can re-apply eval() after
        # each model.train() call (which would otherwise undo the freeze).
        norm_cfg = config.get('normalization', {})
        self._freeze_bn = norm_cfg.get('freeze_bn_teacher', False)
        if self._freeze_bn:
            self._freeze_batch_norm()

        # Step counter for gradient accumulation
        self.global_step = 0

    def _freeze_batch_norm(self) -> None:
        """
        Freeze BatchNorm layers for LoRA training.

        Sets BN to eval mode so it uses the pre-trained running statistics
        (running_mean / running_var) rather than batch statistics. This keeps
        normalization stable with the small effective batch sizes typical of
        LoRA runs. track_running_stats is intentionally left True — setting it
        to False would make BN compute batch stats even in eval mode.
        """
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def training_step(
        self,
        batch: Dict,
        compute_loss_fn: Callable,
    ) -> Dict[str, torch.Tensor]:
        """
        Execute one training step with AMP and gradient clipping.
        
        Args:
            batch: Input batch
            compute_loss_fn: Function that computes loss given batch
        
        Returns:
            Dict with loss and metrics
        """
        # Set training mode here (not in BaseModelTrainer) so that the BN
        # re-freeze happens atomically: model.train() sets all submodules to
        # train mode, then _freeze_batch_norm() immediately overrides BN back
        # to eval. Without this, model.train() in base.py would undo the freeze
        # set during __init__.
        self.model.train()
        if self._freeze_bn:
            self._freeze_batch_norm()

        # Zero gradients at start of accumulation cycle
        if self.global_step % self.accumulation_steps == 0:
            self.optimizer.zero_grad(set_to_none=True)

        if self.use_amp:
            device_type = next(self.model.parameters()).device.type
            with autocast(device_type=device_type, dtype=self.amp_dtype):
                loss_dict = compute_loss_fn(batch)
                loss = loss_dict['loss']

            # Scale loss for accumulation
            scaled_loss = loss / self.accumulation_steps
            if self.scaler is not None:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            # Update weights after accumulation
            if (self.global_step + 1) % self.accumulation_steps == 0:
                if self.clip_enabled:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.clip_max_norm,
                        norm_type=self.clip_norm_type,
                        error_if_nonfinite=self.clip_error_if_nonfinite,
                    )

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
        else:
            loss_dict = compute_loss_fn(batch)
            loss = loss_dict['loss']

            scaled_loss = loss / self.accumulation_steps
            scaled_loss.backward()

            if (self.global_step + 1) % self.accumulation_steps == 0:
                if self.clip_enabled:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.clip_max_norm,
                        norm_type=self.clip_norm_type,
                        error_if_nonfinite=self.clip_error_if_nonfinite,
                    )

                self.optimizer.step()

        self.global_step += 1
        loss_dict['lr'] = self.optimizer.param_groups[0]['lr']

        return loss_dict
