"""
Checkpoint Manager for model saving, loading, and early stopping.

This module provides:
- Automatic checkpoint saving at intervals
- Best model tracking based on validation metrics
- Early stopping to prevent overfitting
- Full state restoration for resuming training
- Automatic cleanup of old checkpoints
"""

import torch
import torch.nn as nn
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, List, Any
import yaml
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints: saving, loading, best model selection, early stopping.
    
    Key features:
    - Config-driven (no hardcoded paths/metrics)
    - Automatic cleanup of old checkpoints
    - Best model tracking across experiments
    - Early stopping support
    - Full reproducibility (saves RNG states)
    
    Example:
        >>> manager = CheckpointManager(
        >>>     output_dir='experiments/exp1/teachers/grounding_dino_lora',
        >>>     config_path='configs/defaults/checkpoint_config.yaml'
        >>> )
        >>> 
        >>> # During training
        >>> for epoch in range(epochs):
        >>>     train_metrics = train_epoch()
        >>>     val_metrics = validate()
        >>>     metrics = {**train_metrics, **val_metrics, 'epoch': epoch}
        >>>     
        >>>     manager.save_checkpoint(
        >>>         epoch=epoch,
        >>>         model=model,
        >>>         optimizer=optimizer,
        >>>         metrics=metrics
        >>>     )
        >>>     
        >>>     if manager.should_stop:
        >>>         break
    """
    
    def __init__(self, output_dir: str, config_path: str):
        """
        Args:
            output_dir: Directory to save checkpoints
            config_path: Path to checkpoint config
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.config = config.get('checkpointing', config)
        
        # Best model tracking
        self.monitor_metric = self.config.get('monitor_metric', 'mAP50')
        self.mode = self.config.get('mode', 'max')
        self.min_delta = self.config.get('min_delta', 0.001)
        
        self.best_metric = float('-inf') if self.mode == 'max' else float('inf')
        self.best_epoch = -1
        
        # Early stopping
        self.early_stop_cfg = self.config.get('early_stopping', {})
        self.patience_counter = 0
        self.should_stop = False
        
        # Checkpoint history
        self.checkpoint_history: List[Path] = []
        
        logger.info(f"CheckpointManager initialized: output_dir={output_dir}")
        logger.info(f"Monitoring metric: {self.monitor_metric} (mode={self.mode})")
    
    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, float],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        extra_info: Optional[Dict] = None
    ) -> Optional[Path]:
        """
        Save a checkpoint with all necessary information.
        
        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer state
            metrics: Dictionary of metrics (must include monitor_metric)
            scheduler: Optional LR scheduler state
            scaler: Optional AMP scaler state
            extra_info: Additional info to save (e.g., config, args)
        
        Returns:
            Path to saved checkpoint (or None if not saved this epoch)
        """
        # Prepare checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add optional components
        if scheduler is not None and self.config.get('save_scheduler', True):
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if scaler is not None and self.config.get('save_scaler', True):
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        
        if self.config.get('save_rng_state', True):
            checkpoint['rng_state'] = {
                'python': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
        
        if extra_info:
            checkpoint['extra_info'] = extra_info
        
        # Determine if this is the best model
        is_best = self._is_best(metrics)
        save_paths = []
        
        # Save regular checkpoint (every N epochs)
        save_interval = self.config.get('save_interval', 5)
        if epoch % save_interval == 0:
            # Format checkpoint name
            checkpoint_format = self.config.get('checkpoint_format', 'epoch_{epoch:04d}.pth')
            try:
                checkpoint_name = checkpoint_format.format(epoch=epoch, **metrics)
            except (KeyError, ValueError):
                # Fallback if format string has invalid keys
                checkpoint_name = f'epoch_{epoch:04d}.pth'
            
            checkpoint_path = self.output_dir / checkpoint_name
            torch.save(checkpoint, checkpoint_path)
            save_paths.append(checkpoint_path)
            self.checkpoint_history.append(checkpoint_path)
            logger.info(f"✓ Saved checkpoint: {checkpoint_path.name}")
        
        # Save best checkpoint
        if is_best and self.config.get('save_best', True):
            best_name = self.config.get('best_checkpoint_name', 'best.pth')
            best_path = self.output_dir / best_name
            torch.save(checkpoint, best_path)
            save_paths.append(best_path)
            logger.info(f"✨ New best model! {self.monitor_metric}={metrics.get(self.monitor_metric, 'N/A'):.4f}")
            logger.info(f"✓ Saved best checkpoint: {best_path.name}")
        
        # Save last checkpoint (overwrite)
        if self.config.get('save_last', True):
            last_name = self.config.get('last_checkpoint_name', 'last.pth')
            last_path = self.output_dir / last_name
            torch.save(checkpoint, last_path)
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
        
        # Check early stopping
        self._check_early_stopping(metrics)
        
        return save_paths[0] if save_paths else None
    
    def _is_best(self, metrics: Dict[str, float]) -> bool:
        """Check if current metrics are the best so far."""
        if self.monitor_metric not in metrics:
            logger.warning(f"Monitor metric '{self.monitor_metric}' not in metrics: {list(metrics.keys())}")
            return False
        
        current_metric = metrics[self.monitor_metric]
        
        if self.mode == 'max':
            is_better = current_metric > (self.best_metric + self.min_delta)
        else:
            is_better = current_metric < (self.best_metric - self.min_delta)
        
        if is_better:
            self.best_metric = current_metric
            self.best_epoch = metrics.get('epoch', -1)
            self.patience_counter = 0  # Reset early stopping counter
            return True
        
        return False
    
    def _check_early_stopping(self, metrics: Dict[str, float]) -> None:
        """Check if training should stop early."""
        if not self.early_stop_cfg.get('enabled', False):
            return
        
        if self.monitor_metric not in metrics:
            return
        
        current_metric = metrics[self.monitor_metric]
        min_delta = self.early_stop_cfg.get('min_delta', 0.001)
        
        # Check if improved
        if self.mode == 'max':
            improved = current_metric > (self.best_metric + min_delta)
        else:
            improved = current_metric < (self.best_metric - min_delta)
        
        if not improved:
            self.patience_counter += 1
            patience = self.early_stop_cfg.get('patience', 15)
            logger.info(f"No improvement for {self.patience_counter}/{patience} epochs")
            
            if self.patience_counter >= patience:
                self.should_stop = True
                logger.info(f"⚠️  Early stopping triggered!")
                logger.info(f"Best {self.monitor_metric}={self.best_metric:.4f} at epoch {self.best_epoch}")
    
    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints to save disk space."""
        max_keep = self.config.get('max_keep_checkpoints', 5)
        
        if len(self.checkpoint_history) > max_keep:
            # Keep only the most recent checkpoints
            to_remove = self.checkpoint_history[:-max_keep]
            self.checkpoint_history = self.checkpoint_history[-max_keep:]
            
            # Protected checkpoint names (don't delete these)
            protected = [
                self.config.get('best_checkpoint_name', 'best.pth'),
                self.config.get('last_checkpoint_name', 'last.pth')
            ]
            
            for checkpoint_path in to_remove:
                if checkpoint_path.exists() and checkpoint_path.name not in protected:
                    checkpoint_path.unlink()
                    logger.debug(f"Removed old checkpoint: {checkpoint_path.name}")
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        load_optimizer: bool = True,
        strict: bool = True,
        map_location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint and restore training state.
        
        Args:
            checkpoint_path: Path to checkpoint file (or 'best', 'last')
            model: Model to load weights into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            scaler: Optional scaler to load state into
            load_optimizer: Whether to load optimizer state
            strict: Strict mode for model loading
            map_location: Device to map tensors to
        
        Returns:
            Checkpoint dictionary with metadata
        
        Example:
            >>> # Load best checkpoint
            >>> checkpoint = manager.load_checkpoint(
            >>>     'best',
            >>>     model=model,
            >>>     optimizer=optimizer
            >>> )
            >>> start_epoch = checkpoint['epoch'] + 1
        """
        # Resolve special checkpoint names
        if checkpoint_path in ['best', 'last']:
            if checkpoint_path == 'best':
                checkpoint_path = self.output_dir / self.config.get('best_checkpoint_name', 'best.pth')
            else:
                checkpoint_path = self.output_dir / self.config.get('last_checkpoint_name', 'last.pth')
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        # Load checkpoint
        if map_location is None:
            map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Load model
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        logger.info(f"✓ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        
        # Load optimizer
        if load_optimizer and optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("✓ Optimizer state loaded")
        
        # Load scheduler
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("✓ Scheduler state loaded")
        
        # Load scaler
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logger.info("✓ AMP scaler state loaded")
        
        # Load RNG state for reproducibility
        if 'rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state']['python'])
            if torch.cuda.is_available() and checkpoint['rng_state']['cuda'] is not None:
                torch.cuda.set_rng_state_all(checkpoint['rng_state']['cuda'])
            logger.info("✓ RNG states restored")
        
        # Restore best metric tracking
        self.best_metric = checkpoint.get('best_metric', self.best_metric)
        self.best_epoch = checkpoint.get('best_epoch', -1)
        
        logger.info(f"✓ Checkpoint loaded successfully")
        if self.best_epoch >= 0:
            logger.info(f"  Best metric so far: {self.best_metric:.4f} at epoch {self.best_epoch}")
        
        return checkpoint
    
    def get_best_checkpoint_path(self) -> Path:
        """Get path to best checkpoint."""
        return self.output_dir / self.config.get('best_checkpoint_name', 'best.pth')
    
    def get_last_checkpoint_path(self) -> Path:
        """Get path to last checkpoint."""
        return self.output_dir / self.config.get('last_checkpoint_name', 'last.pth')
    
    def has_checkpoint(self, name: str = 'last') -> bool:
        """Check if a checkpoint exists."""
        if name == 'best':
            path = self.get_best_checkpoint_path()
        elif name == 'last':
            path = self.get_last_checkpoint_path()
        else:
            path = self.output_dir / name
        return path.exists()
    
    def reset_early_stopping(self) -> None:
        """Reset early stopping counter."""
        self.patience_counter = 0
        self.should_stop = False
        logger.info("Early stopping counter reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of checkpoint manager."""
        return {
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'patience_counter': self.patience_counter,
            'should_stop': self.should_stop,
            'num_checkpoints': len(self.checkpoint_history),
            'monitor_metric': self.monitor_metric,
            'mode': self.mode
        }
