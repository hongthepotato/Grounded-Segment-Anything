"""
Logging configuration and utilities.

This module sets up consistent logging across the platform.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = 'grounded_sam',
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with console and file handlers.
    
    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level (default: INFO)
        format_string: Optional custom format string
    
    Returns:
        Configured logger instance
    
    Example:
        >>> logger = setup_logger(
        >>>     name='training',
        >>>     log_file='logs/train.log',
        >>>     level=logging.DEBUG
        >>> )
        >>> logger.info("Training started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Default format
    if format_string is None:
        format_string = '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
    
    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger by name.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    
    Example:
        >>> logger = get_logger('training')
        >>> logger.info("Message")
    """
    return logging.getLogger(name)


def create_training_logger(exp_dir: Path, name: str = 'training') -> logging.Logger:
    """
    Create a logger specifically for training with file output.
    
    Args:
        exp_dir: Experiment directory
        name: Logger name
    
    Returns:
        Configured training logger
    
    Example:
        >>> logger = create_training_logger(Path('experiments/exp1'))
        >>> logger.info("Epoch 1/100")
    """
    log_dir = exp_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{name}_{timestamp}.log'
    
    return setup_logger(
        name=name,
        log_file=str(log_file),
        level=logging.INFO
    )


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


# Global logger instance
_default_logger = None


def get_default_logger() -> logging.Logger:
    """Get or create the default platform logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logger('grounded_sam')
    return _default_logger


