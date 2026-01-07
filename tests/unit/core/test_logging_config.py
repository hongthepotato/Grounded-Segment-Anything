"""
Unit tests for core.logging_config module.

Tests centralized logging configuration functions.
"""

import logging
import os
from unittest.mock import patch
import pytest
from core.logging_formatters import JSONFormatter
from core.logging_formatters import TextFormatter

from core.logging_config import (
    configure_logging,
    get_logger,
    get_job_logger,
    reset_logging,
)


@pytest.fixture(autouse=True)
def cleanup_logging():
    """Reset logging configuration before and after each test."""
    reset_logging()
    yield
    reset_logging()


@pytest.mark.unit
class TestConfigureLogging:
    """Test configure_logging function."""

    def test_configure_logging_default(self):
        """Test configure_logging with default parameters."""
        configure_logging()
        root_logger = logging.getLogger()

        assert root_logger.level == logging.INFO
        assert len(root_logger.handlers) == 1  # Console only

    def test_configure_logging_custom_level(self):
        """Test configure_logging with custom log level."""
        configure_logging(level="DEBUG")

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_configure_logging_from_environment(self):
        """Test configure_logging reads from environment variables."""
        with patch.dict(os.environ, {"LOG_LEVEL": "WARNING"}):
            configure_logging()

        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING

    def test_configure_logging_with_file(self, temp_dir):
        """Test configure_logging with file output."""
        log_file = temp_dir / "test.log"

        configure_logging(log_file=str(log_file))

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 2

        root_logger.info("Test message")

        for handler in root_logger.handlers:
            handler.close()

        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content

    def test_configure_logging_idempotent(self):
        """Test that configure_logging is idempotent (only configures once)."""
        configure_logging()
        handler_count_1 = len(logging.getLogger().handlers)

        configure_logging()
        handler_count_2 = len(logging.getLogger().handlers)

        assert handler_count_1 == handler_count_2

    def test_configure_logging_force_reconfigure(self):
        """Test force parameter allows reconfiguration."""
        configure_logging(level="INFO")
        assert logging.getLogger().level == logging.INFO

        configure_logging(level="DEBUG", force=False)
        assert logging.getLogger().level == logging.INFO

        configure_logging(level="DEBUG", force=True)
        assert logging.getLogger().level == logging.DEBUG

    def test_configure_logging_json_format(self):
        """Test configure_logging with JSON format."""
        configure_logging(format_type="json")

        root_logger = logging.getLogger()
        handler = root_logger.handlers[0]

        assert isinstance(handler.formatter, JSONFormatter)

    def test_configure_logging_text_format(self):
        """Test configure_logging with text format."""
        configure_logging(format_type="text")

        root_logger = logging.getLogger()
        handler = root_logger.handlers[0]

        assert isinstance(handler.formatter, TextFormatter)


@pytest.mark.unit
class TestGetLogger:
    """Test get_logger function."""

    def test_get_logger_returns_logger(self):
        """Test get_logger returns a logger instance."""
        logger = get_logger("test_module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_get_logger_same_instance(self):
        """Test get_logger returns same instance for same name."""
        logger1 = get_logger("test_module")
        logger2 = get_logger("test_module")

        assert logger1 is logger2

    def test_get_logger_different_instances(self):
        """Test get_logger returns different instances for different names."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        assert logger1 is not logger2


@pytest.mark.unit
class TestGetJobLogger:
    """Test get_job_logger function."""

    def test_get_job_logger_creates_file(self, temp_dir):
        """Test get_job_logger creates log file in output directory."""
        job_id = "test-job-12345678"

        logger = get_job_logger(job_id, str(temp_dir))
        logger.info("Test message")

        for handler in logger.handlers:
            handler.close()

        log_dir = temp_dir / "logs"
        assert log_dir.exists()
        log_files = list(log_dir.glob("training_*.log"))
        assert len(log_files) == 1

    def test_get_job_logger_file_content(self, temp_dir):
        """Test get_job_logger writes correct content to file."""
        job_id = "test-job-12345678"

        logger = get_job_logger(job_id, str(temp_dir))
        logger.info("Test message from job")

        for handler in logger.handlers:
            handler.close()

        log_dir = temp_dir / "logs"
        log_file = list(log_dir.glob("training_*.log"))[0]
        content = log_file.read_text()

        assert "Test message from job" in content
        assert job_id[:8] in content  # Job ID prefix should be in log

    def test_get_job_logger_custom_name(self, temp_dir):
        """Test get_job_logger with custom logger name."""
        job_id = "test-job-12345678"

        logger = get_job_logger(job_id, str(temp_dir), name="evaluation")
        logger.info("Evaluation message")

        # Close handlers to flush
        for handler in logger.handlers:
            handler.close()

        log_dir = temp_dir / "logs"
        log_files = list(log_dir.glob("evaluation_*.log"))
        assert len(log_files) == 1

    def test_get_job_logger_only_file_handler(self, temp_dir):
        """Test get_job_logger only has file handler (no console)."""
        job_id = "test-job-12345678"

        logger = get_job_logger(job_id, str(temp_dir))

        # Should only have file handler (console removed to avoid polluting worker logs)
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.FileHandler)

    def test_get_job_logger_does_not_propagate(self, temp_dir):
        """Test get_job_logger does not propagate to root logger."""
        job_id = "test-job-12345678"

        logger = get_job_logger(job_id, str(temp_dir))

        assert logger.propagate is False

    def test_get_job_logger_json_format(self, temp_dir):
        """Test get_job_logger respects LOG_FORMAT environment variable."""
        job_id = "test-job-12345678"

        with patch.dict(os.environ, {"LOG_FORMAT": "json"}):
            logger = get_job_logger(job_id, str(temp_dir))
            logger.info("JSON test message")

            # Close handlers to flush
            for handler in logger.handlers:
                handler.close()

        log_dir = temp_dir / "logs"
        log_file = list(log_dir.glob("training_*.log"))[0]
        content = log_file.read_text()

        # File contains multiple JSON lines (newline-delimited JSON)
        # Parse each line separately
        import json
        lines = content.strip().split('\n')
        assert len(lines) >= 1

        # Find the line with our test message
        test_log_entry = None
        for line in lines:
            if line.strip():
                entry = json.loads(line)
                if entry["message"] == "JSON test message":
                    test_log_entry = entry
                    break

        assert test_log_entry is not None
        assert test_log_entry["message"] == "JSON test message"
        assert test_log_entry["job_id"] == job_id


@pytest.mark.unit
class TestResetLogging:
    """Test reset_logging function."""

    def test_reset_logging_clears_handlers(self):
        """Test reset_logging clears all handlers."""
        configure_logging()
        assert len(logging.getLogger().handlers) > 0

        reset_logging()
        assert len(logging.getLogger().handlers) == 0

    def test_reset_logging_allows_reconfigure(self):
        """Test reset_logging allows re-running configure_logging."""
        configure_logging(level="INFO")
        reset_logging()
        configure_logging(level="DEBUG")

        assert logging.getLogger().level == logging.DEBUG
