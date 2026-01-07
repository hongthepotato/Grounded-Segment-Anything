"""
Custom logging formatters for the platform.

Provides:
- TextFormatter: Human-readable format for development
- JSONFormatter: Structured JSON format for production log aggregation

Usage:
    from core.logging_formatters import TextFormatter, JSONFormatter
    
    handler.setFormatter(TextFormatter())
    # or
    handler.setFormatter(JSONFormatter())
"""

import json
import sys
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional

from core.constants import LOG_FORMAT_STRING, DATE_FORMAT_STRING


class TextFormatter(logging.Formatter):
    """
    Standard text formatter with consistent styling.
    
    Format: [timestamp] [name] [level] message
    
    Example output:
        [2025-01-06 14:30:22] [ml_engine.training] [INFO] Training started
        [2025-01-06 14:30:23] [ml_engine.training] [ERROR] Failed to load model
    
    Args:
        fmt: Optional custom format string
        datefmt: Optional custom date format
    """

    DEFAULT_FORMAT = LOG_FORMAT_STRING
    DEFAULT_DATE_FORMAT = DATE_FORMAT_STRING

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None
    ):
        fmt = fmt or self.DEFAULT_FORMAT
        datefmt = datefmt or self.DEFAULT_DATE_FORMAT
        super().__init__(fmt=fmt, datefmt=datefmt)


class JSONFormatter(logging.Formatter):
    """
    Structured JSON formatter for production log aggregation.
    
    Outputs each log record as a single JSON line, suitable for:
    - ELK Stack (Elasticsearch, Logstash, Kibana)
    - AWS CloudWatch
    - Grafana Loki
    - Google Cloud Logging
    
    Output fields:
        - timestamp: ISO 8601 format
        - level: Log level name (INFO, ERROR, etc.)
        - logger: Logger name
        - message: Log message
        - module: Source module
        - function: Source function
        - line: Source line number
        - exception: Exception info (if present)
        - ... any extra fields passed to constructor
    
    Example output:
        {"timestamp": "2025-01-06T14:30:22.123456", "level": "INFO", 
         "logger": "ml_engine.training", "message": "Training started", ...}
    
    Args:
        extra_fields: Optional dict of extra fields to include in every log
    """

    def __init__(self, extra_fields: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.extra_fields = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        # Base fields
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields from constructor
        log_data.update(self.extra_fields)

        # Add extra fields from record (e.g., logger.info("msg", extra={...}))
        if hasattr(record, "__dict__"):
            # Standard fields to exclude
            standard_fields = {
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message", "asctime"
            }
            for key, value in record.__dict__.items():
                if key not in standard_fields and not key.startswith("_"):
                    # Try to serialize, skip if not serializable
                    try:
                        json.dumps(value)
                        log_data[key] = value
                    except (TypeError, ValueError):
                        log_data[key] = str(value)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }

        # Serialize to JSON (single line, no pretty printing)
        return json.dumps(log_data, default=str, ensure_ascii=False)


class ColoredTextFormatter(TextFormatter):
    """
    Text formatter with ANSI color codes for terminal output.
    
    Colors:
        - DEBUG: Cyan
        - INFO: Green
        - WARNING: Yellow
        - ERROR: Red
        - CRITICAL: Red (bold)
    
    Note: Colors are only applied if output is a terminal (TTY).
    Falls back to plain TextFormatter if not a TTY.
    """

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[1;31m" # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format with colors if output is a terminal."""
        formatted = super().format(record)

        # Only colorize if we're writing to a terminal (TTY)
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            color = self.COLORS.get(record.levelname, "")
            if color:
                return f"{color}{formatted}{self.RESET}"

        return formatted
