"""
Checkpoint Manager for model saving, loading, and early stopping.

This module provides:
- Automatic checkpoint saving at intervals
- Best model tracking based on validation metrics
- Early stopping to prevent overfitting
- Full state restoration for resuming training
- Automatic cleanup of old checkpoints
- Trainable-only saving for LoRA (saves disk space)
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints with best model tracking and early stopping.

    Example:
        >>> manager = CheckpointManager(
        >>>     output_dir='experiments/exp1/grounding_dino',
        >>>     config_path='configs/defaults/checkpoint_config.yaml',
        >>>     monitor_metric='val_grounding_dino_total_loss',
        >>>     mode='min'
        >>> )
        >>>
        >>> for epoch in range(epochs):
        >>>     metrics = train_and_validate()
        >>>     manager.save_checkpoint(epoch, model, optimizer, metrics)
        >>>     if manager.should_stop:
        >>>         break
    """

    def __init__(self, output_dir: str, config_path: str, monitor_metric: str, mode: str = "min"):
        """
        Args:
            output_dir: Directory to save checkpoints
            config_path: Path to checkpoint config
            monitor_metric: Metric to monitor for best model
            mode: 'min' or 'max' for best model selection
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.config = config.get("checkpointing", config)

        self.monitor_metric = monitor_metric
        self.mode = mode
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = -1

        # Early stopping
        early_cfg = self.config.get("early_stopping", {})
        self.early_stopping_enabled = early_cfg.get("enabled", False)
        self.patience = early_cfg.get("patience", 15)
        self.patience_counter = 0
        self.should_stop = False

        # Checkpoint management
        self.save_interval = self.config.get("save_interval", 5)
        self.max_keep = self.config.get("max_keep_checkpoints", 5)
        self.save_trainable_only = self.config.get("save_trainable_only", False)
        self.checkpoint_history: List[Path] = []

        logger.info(f"CheckpointManager: {output_dir}")
        logger.info(f"  Monitoring: {monitor_metric} (mode={mode})")

    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, float],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        extra_info: Optional[Dict] = None,
    ) -> Optional[Path]:
        """
        Save checkpoint with full state.

        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer state
            metrics: All metrics (must include monitor_metric)
            scheduler: Optional scheduler state
            scaler: Optional AMP scaler state
            extra_info: Additional info to save

        Returns:
            Path to saved checkpoint (or None if not saved)
        """
        # Prepare model state
        if self.save_trainable_only:
            model_state = {
                name: param.data for name, param in model.named_parameters() if param.requires_grad
            }
        else:
            model_state = model.state_dict()

        # Build checkpoint dict
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "timestamp": datetime.now().isoformat(),
            "trainable_only": self.save_trainable_only,
        }

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()

        # Save RNG state for reproducibility
        checkpoint["rng_state"] = {
            "python": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        if extra_info:
            checkpoint["extra_info"] = extra_info

        # Check if best
        is_best = self._is_best(metrics)
        saved_path = None

        # Save periodic checkpoint
        if epoch % self.save_interval == 0:
            path = self.output_dir / f"epoch_{epoch:04d}.pth"
            torch.save(checkpoint, path)
            self.checkpoint_history.append(path)
            saved_path = path
            logger.info(f"✓ Saved checkpoint: {path.name}")

        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / "best.pth"
            torch.save(checkpoint, best_path)
            saved_path = best_path
            logger.info(f"✨ New best! {self.monitor_metric}={metrics.get(self.monitor_metric, 'N/A'):.4f}")

        # Always save last checkpoint
        last_path = self.output_dir / "last.pth"
        torch.save(checkpoint, last_path)

        # Cleanup old checkpoints
        self._cleanup()

        # Check early stopping
        self._check_early_stopping(metrics)

        return saved_path

    def _is_best(self, metrics: Dict[str, float]) -> bool:
        """Check if current metrics are best so far."""
        if self.monitor_metric not in metrics:
            return False

        current = metrics[self.monitor_metric]
        min_delta = 0.001

        if self.mode == "max":
            is_better = current > (self.best_metric + min_delta)
        else:
            is_better = current < (self.best_metric - min_delta)

        if is_better:
            self.best_metric = current
            self.best_epoch = metrics.get("epoch", -1)
            self.patience_counter = 0
            return True

        return False

    def _check_early_stopping(self, metrics: Dict[str, float]) -> None:
        """Check if training should stop early."""
        if not self.early_stopping_enabled:
            return

        if self.monitor_metric not in metrics:
            return

        # If we didn't improve, increment patience counter
        current = metrics[self.monitor_metric]
        if self.mode == "max":
            improved = current > self.best_metric
        else:
            improved = current < self.best_metric

        if not improved:
            self.patience_counter += 1
            logger.info(f"No improvement for {self.patience_counter}/{self.patience} epochs")

            if self.patience_counter >= self.patience:
                self.should_stop = True
                logger.info("⚠️ Early stopping triggered!")

    def _cleanup(self) -> None:
        """Remove old checkpoints to save disk space."""
        if len(self.checkpoint_history) > self.max_keep:
            to_remove = self.checkpoint_history[: -self.max_keep]
            self.checkpoint_history = self.checkpoint_history[-self.max_keep :]

            for path in to_remove:
                if path.exists() and path.name not in ["best.pth", "last.pth"]:
                    path.unlink()

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        load_optimizer: bool = True,
        load_rng_state: bool = True,
    ) -> Dict[str, Any]:
        """
        Load checkpoint and restore state.

        Args:
            checkpoint_path: Path to checkpoint (or 'best', 'last')
            model: Model to load into
            optimizer: Optional optimizer to restore
            scheduler: Optional scheduler to restore
            scaler: Optional scaler to restore
            load_optimizer: Whether to load optimizer state
            load_rng_state: Whether to restore RNG state

        Returns:
            Checkpoint dict with metadata
        """
        # Resolve special names
        if checkpoint_path == "best":
            checkpoint_path = self.output_dir / "best.pth"
        elif checkpoint_path == "last":
            checkpoint_path = self.output_dir / "last.pth"
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint: {checkpoint_path}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model (handle trainable-only)
        trainable_only = checkpoint.get("trainable_only", False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=not trainable_only)
        logger.info(f"✓ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")

        # Load optimizer
        if load_optimizer and optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("✓ Optimizer loaded")

        # Load scheduler
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.info("✓ Scheduler loaded")

        # Load scaler
        if scaler and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            logger.info("✓ Scaler loaded")

        # Load RNG state
        if load_rng_state and "rng_state" in checkpoint:
            try:
                rng = checkpoint["rng_state"]["python"]
                if not isinstance(rng, torch.ByteTensor):
                    rng = rng.to(torch.uint8)
                torch.set_rng_state(rng)

                if torch.cuda.is_available() and checkpoint["rng_state"]["cuda"]:
                    cuda_rng = checkpoint["rng_state"]["cuda"]
                    if cuda_rng and not isinstance(cuda_rng[0], torch.ByteTensor):
                        cuda_rng = [s.to(torch.uint8) for s in cuda_rng]
                    torch.cuda.set_rng_state_all(cuda_rng)
                logger.info("✓ RNG state restored")
            except Exception as e:
                logger.warning(f"Failed to restore RNG: {e}")

        # Restore tracking state
        self.best_metric = checkpoint.get("best_metric", self.best_metric)
        self.best_epoch = checkpoint.get("best_epoch", -1)

        return checkpoint

    def get_best_checkpoint_path(self) -> Path:
        """Get path to best checkpoint."""
        return self.output_dir / "best.pth"

    def get_last_checkpoint_path(self) -> Path:
        """Get path to last checkpoint."""
        return self.output_dir / "last.pth"
