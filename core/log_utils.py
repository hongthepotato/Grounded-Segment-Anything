"""
Logging utilities for the platform.
"""

import logging
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
