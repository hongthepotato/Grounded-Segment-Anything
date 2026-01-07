"""
Logging utilities for the platform.

This module provides:
- TensorBoardLogger: Wrapper for TensorBoard logging
- log_config(): Pretty-print configuration dictionaries
- log_metrics(): Format and log training metrics
"""

import logging
import warnings
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def log_config(logger: logging.Logger, config: dict, title: str = "Configuration") -> None:
    """
    Log configuration dictionary in a readable format.
    
    Args:
        logger: Logger instance
        config: Configuration dictionary
        title: Title for the log section
    
    Example:
        >>> log_config(logger, config, title="Training Configuration")
    """
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)

    def log_dict(d: dict, indent: int = 0):
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info("  " * indent + f"{key}:")
                log_dict(value, indent + 1)
            else:
                logger.info("  " * indent + f"{key}: {value}")

    log_dict(config)
    logger.info("=" * 60)


def log_metrics(
    logger: logging.Logger,
    metrics: dict,
    epoch: Optional[int] = None,
    prefix: str = ""
) -> None:
    """
    Log metrics in a consistent format.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metric names to values
        epoch: Optional epoch number
        prefix: Optional prefix for metric names
    
    Example:
        >>> metrics = {'loss': 0.5, 'mAP50': 0.85}
        >>> log_metrics(logger, metrics, epoch=10, prefix="val")
    """
    if epoch is not None:
        msg = f"Epoch {epoch}"
    else:
        msg = "Metrics"

    if prefix:
        msg = f"{prefix} - {msg}"

    metric_strs = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                   for k, v in metrics.items()]
    msg += " | " + " | ".join(metric_strs)

    logger.info(msg)


class TensorBoardLogger:
    """
    Wrapper for TensorBoard logging.
    
    Example:
        >>> tb_logger = TensorBoardLogger('experiments/exp1/logs')
        >>> tb_logger.log_scalar('loss', 0.5, step=100)
        >>> tb_logger.log_scalars({'train_loss': 0.5, 'val_loss': 0.6}, step=100)
    """
    
    def __init__(self, log_dir: str):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.enabled = True
        except ImportError:
            logging.warning("TensorBoard not available. Install with: pip install tensorboard")
            self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value."""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, metrics: dict, step: int, prefix: str = "") -> None:
        """Log multiple scalar values."""
        if self.enabled:
            for key, value in metrics.items():
                tag = f"{prefix}/{key}" if prefix else key
                self.writer.add_scalar(tag, value, step)
    
    def log_image(self, tag: str, image, step: int) -> None:
        """Log an image."""
        if self.enabled:
            self.writer.add_image(tag, image, step)
    
    def log_histogram(self, tag: str, values, step: int) -> None:
        """Log a histogram."""
        if self.enabled:
            self.writer.add_histogram(tag, values, step)
    
    def close(self) -> None:
        """Close the writer."""
        if self.enabled:
            self.writer.close()
