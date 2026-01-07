"""
Logging utilities for the platform.

This module provides:
- TensorBoardLogger: Wrapper for TensorBoard logging
"""

import logging


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
