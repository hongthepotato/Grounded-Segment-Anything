"""
Centralized logging configuration for the platform.

This module provides a unified logging setup that:
- Configures root logger once at application entry points
- Supports both text (development) and JSON (production) output formats
- Provides job-specific loggers that persist to experiment directories
- Is backward compatible with existing `logging.getLogger(__name__)` pattern

Usage:
    # At entry points (API startup, CLI main, subprocess entry):
    from core.logging_config import configure_logging
    configure_logging()
    
    # In any module:
    from core.logging_config import get_logger
    logger = get_logger(__name__)
    
    # For training subprocesses (logs saved to experiment dir):
    from core.logging_config import get_job_logger
    logger = get_job_logger(job_id, output_dir)

Environment Variables:
    LOG_LEVEL: DEBUG, INFO, WARNING, ERROR (default: INFO)
    LOG_FORMAT: text, json (default: text)
    LOG_FILE: Optional path to global log file
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.logging_formatters import TextFormatter, JSONFormatter, ColoredTextFormatter
from core.constants import (
    DEFAULT_LOG_LEVEL,
    LOG_FORMAT_TEXT,
    LOG_FORMAT_JSON,
    DATE_FORMAT
)

# Track if logging has been configured
_configured = False


def configure_logging(
    level: Optional[str] = None,
    format_type: Optional[str] = None,
    log_file: Optional[str] = None,
    force: bool = False
) -> None:
    """
    Configure the root logger for the application.
    
    Should be called once at application entry points:
    - API startup (lifespan)
    - CLI script main()
    - Subprocess entry point
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). 
               Defaults to LOG_LEVEL env var or "INFO".
        format_type: Output format ("text" or "json").
                    Defaults to LOG_FORMAT env var or "text".
        log_file: Optional path to log file. 
                 Defaults to LOG_FILE env var.
        force: If True, reconfigure even if already configured.
               Useful for testing.
    
    Example:
        # Basic usage (reads from environment):
        configure_logging()
        
        # Override specific settings:
        configure_logging(level="DEBUG", format_type="json")
        
        # With file output:
        configure_logging(log_file="logs/app.log")
    """
    global _configured

    if _configured and not force:
        return

    # Get configuration from environment or parameters
    level = level or os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL)
    format_type = format_type or os.environ.get("LOG_FORMAT", LOG_FORMAT_TEXT)
    log_file = log_file or os.environ.get("LOG_FILE")

    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers = []

    # Create formatter based on format type
    if format_type.lower() == LOG_FORMAT_JSON:
        formatter = JSONFormatter()
    else:
        # Use colored formatter for console if TTY detected
        if sys.stdout.isatty():
            console_formatter = ColoredTextFormatter()
        else:
            console_formatter = TextFormatter()
        # Always use plain text for file (will create later if needed)
        file_formatter = TextFormatter()

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    if format_type.lower() == LOG_FORMAT_JSON:
        console_handler.setFormatter(formatter)
    else:
        console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        if format_type.lower() == LOG_FORMAT_JSON:
            file_handler.setFormatter(formatter)
        else:
            file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    _configured = True

    # Log configuration (only at DEBUG level to avoid noise)
    root_logger.debug(
        "Logging configured: level=%s, format=%s, file=%s",
        level, format_type, log_file or "(none)"
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance by name.
    
    This is a thin wrapper around logging.getLogger() that ensures
    the logging system is configured. Safe to call before or after
    configure_logging().
    
    Args:
        name: Logger name (typically __name__ from the calling module)
    
    Returns:
        logging.Logger instance
    
    Example:
        from core.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("Hello, world!")
    """
    return logging.getLogger(name)


def get_job_logger(
    job_id: str,
    output_dir: str,
    name: str = "training"
) -> logging.Logger:
    """
    Get a logger for a specific training job that persists to the experiment directory.
    
    Job logs are written ONLY to the experiment directory, not to console/worker logs.
    This ensures all job-related artifacts (configs, checkpoints, logs) are co-located.
    
    Creates a logger that writes to:
    - File: {output_dir}/logs/{name}_{timestamp}.log
    
    Note: Console output is intentionally excluded to avoid polluting worker system logs.
    Use `tail -f {output_dir}/logs/*.log` to monitor training in real-time.
    
    Args:
        job_id: Unique job identifier (used in log messages)
        output_dir: Experiment output directory (required)
        name: Logger name (default: "training")
    
    Returns:
        Configured logger with file handler
    
    Example:
        # In subprocess_runner.py:
        logger = get_job_logger(job_id, output_dir)
        logger.info("Training started")
        # Logs saved to: {output_dir}/logs/training_20250106_143022.log
    """
    # Get format type from environment
    format_type = os.environ.get("LOG_FORMAT", LOG_FORMAT_TEXT)
    level = os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL)
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create unique logger name for this job
    logger_name = f"job.{job_id[:8]}.{name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(numeric_level)

    # Remove existing handlers (in case of reuse)
    logger.handlers = []

    # Don't propagate to root logger (avoid duplicate logs in worker logs)
    logger.propagate = False

    # Create log directory in experiment folder
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime(DATE_FORMAT)
    log_file = log_dir / f"{name}_{timestamp}.log"

    # Create formatter with job context
    if format_type.lower() == LOG_FORMAT_JSON:
        formatter = JSONFormatter(extra_fields={"job_id": job_id})
    else:
        formatter = TextFormatter(fmt=LOG_FORMAT_TEXT)

    # File handler - ONLY write to experiment directory
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Job logger initialized: %s", log_file)

    return logger


def reset_logging() -> None:
    """
    Reset logging configuration.
    
    Useful for testing to ensure clean state between tests.
    """
    global _configured

    # Clear all handlers from root logger
    root_logger = logging.getLogger()
    root_logger.handlers = []

    _configured = False
