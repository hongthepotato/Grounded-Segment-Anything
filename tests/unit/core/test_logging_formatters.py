"""
Unit tests for core.logging_formatters module.

Tests custom logging formatters for text and JSON output.
"""

import json
import logging
import pytest

from core.logging_formatters import TextFormatter, JSONFormatter, ColoredTextFormatter


@pytest.fixture
def log_record():
    """Create a sample log record for testing."""
    record = logging.LogRecord(
        name="test.module",
        level=logging.INFO,
        pathname="/path/to/file.py",
        lineno=42,
        msg="Test message: %s",
        args=("value",),
        exc_info=None
    )
    return record


@pytest.fixture
def log_record_with_exception():
    """Create a log record with exception info."""
    try:
        raise ValueError("Test error")
    except ValueError:
        import sys
        exc_info = sys.exc_info()
    
    record = logging.LogRecord(
        name="test.module",
        level=logging.ERROR,
        pathname="/path/to/file.py",
        lineno=42,
        msg="Error occurred",
        args=(),
        exc_info=exc_info
    )
    return record


@pytest.mark.unit
class TestTextFormatter:
    """Test TextFormatter class."""

    def test_format_basic_message(self, log_record):
        """Test formatting a basic log message."""
        formatter = TextFormatter()
        output = formatter.format(log_record)

        assert "test.module" in output
        assert "INFO" in output
        assert "Test message: value" in output

    def test_format_includes_timestamp(self, log_record):
        """Test that output includes timestamp."""
        formatter = TextFormatter()
        output = formatter.format(log_record)

        # Default format includes [timestamp]
        assert "[" in output and "]" in output

    def test_custom_format_string(self, log_record):
        """Test using custom format string."""
        formatter = TextFormatter(fmt="%(levelname)s: %(message)s")
        output = formatter.format(log_record)

        assert output == "INFO: Test message: value"

    def test_custom_date_format(self, log_record):
        """Test using custom date format."""
        formatter = TextFormatter(datefmt="%H:%M:%S")
        output = formatter.format(log_record)

        # Should not have full date, just time
        assert "-" not in output.split("]")[0]  # Date format usually has dashes

    def test_default_format(self):
        """Test default format constant."""
        assert "[%(asctime)s]" in TextFormatter.DEFAULT_FORMAT
        assert "%(name)s" in TextFormatter.DEFAULT_FORMAT
        assert "%(levelname)s" in TextFormatter.DEFAULT_FORMAT
        assert "%(message)s" in TextFormatter.DEFAULT_FORMAT


@pytest.mark.unit
class TestJSONFormatter:
    """Test JSONFormatter class."""

    def test_format_produces_valid_json(self, log_record):
        """Test that output is valid JSON."""
        formatter = JSONFormatter()
        output = formatter.format(log_record)

        # Should not raise
        data = json.loads(output)
        assert isinstance(data, dict)

    def test_format_includes_required_fields(self, log_record):
        """Test that output includes all required fields."""
        formatter = JSONFormatter()
        output = formatter.format(log_record)
        data = json.loads(output)

        assert "timestamp" in data
        assert "level" in data
        assert "logger" in data
        assert "message" in data
        assert "module" in data
        assert "function" in data
        assert "line" in data

    def test_format_correct_values(self, log_record):
        """Test that field values are correct."""
        formatter = JSONFormatter()
        output = formatter.format(log_record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["logger"] == "test.module"
        assert data["message"] == "Test message: value"
        assert data["line"] == 42

    def test_format_with_extra_fields(self, log_record):
        """Test formatter with extra fields from constructor."""
        formatter = JSONFormatter(extra_fields={"job_id": "test-123", "version": "1.0"})
        output = formatter.format(log_record)
        data = json.loads(output)

        assert data["job_id"] == "test-123"
        assert data["version"] == "1.0"

    def test_format_with_exception(self, log_record_with_exception):
        """Test formatting a log record with exception info."""
        formatter = JSONFormatter()
        output = formatter.format(log_record_with_exception)
        data = json.loads(output)

        assert "exception" in data
        assert data["exception"]["type"] == "ValueError"
        assert data["exception"]["message"] == "Test error"
        assert isinstance(data["exception"]["traceback"], list)

    def test_format_single_line(self, log_record):
        """Test that output is a single line (no pretty printing)."""
        formatter = JSONFormatter()
        output = formatter.format(log_record)

        # Should be single line
        assert "\n" not in output.strip()

    def test_format_with_record_extras(self, log_record):
        """Test that extra fields from record are included."""
        log_record.custom_field = "custom_value"
        log_record.request_id = "req-456"

        formatter = JSONFormatter()
        output = formatter.format(log_record)
        data = json.loads(output)

        assert data["custom_field"] == "custom_value"
        assert data["request_id"] == "req-456"

    def test_format_non_serializable_extra(self, log_record):
        """Test handling of non-JSON-serializable extra fields."""
        class NonSerializable:
            def __str__(self):
                return "NonSerializable object"

        log_record.weird_object = NonSerializable()

        formatter = JSONFormatter()
        # Should not raise
        output = formatter.format(log_record)
        data = json.loads(output)

        # Should be converted to string
        assert data["weird_object"] == "NonSerializable object"

    def test_format_unicode(self, log_record):
        """Test handling of Unicode characters."""
        log_record.msg = "Unicode test: 你好 🎉"
        log_record.args = ()

        formatter = JSONFormatter()
        output = formatter.format(log_record)
        data = json.loads(output)

        assert "你好" in data["message"]
        assert "🎉" in data["message"]


@pytest.mark.unit
class TestColoredTextFormatter:
    """Test ColoredTextFormatter class."""

    def test_inherits_from_text_formatter(self):
        """Test that ColoredTextFormatter inherits from TextFormatter."""
        assert issubclass(ColoredTextFormatter, TextFormatter)

    def test_has_color_codes(self):
        """Test that color codes are defined for all levels."""
        assert "DEBUG" in ColoredTextFormatter.COLORS
        assert "INFO" in ColoredTextFormatter.COLORS
        assert "WARNING" in ColoredTextFormatter.COLORS
        assert "ERROR" in ColoredTextFormatter.COLORS
        assert "CRITICAL" in ColoredTextFormatter.COLORS

    def test_has_reset_code(self):
        """Test that reset code is defined."""
        assert ColoredTextFormatter.RESET == "\033[0m"

    def test_format_includes_color_codes(self, log_record):
        """Test that output includes ANSI color codes when TTY is detected."""
        from unittest.mock import patch

        formatter = ColoredTextFormatter()

        # Mock sys.stdout.isatty() to simulate TTY environment
        with patch('sys.stdout.isatty', return_value=True):
            output = formatter.format(log_record)

        # Should have color code and reset
        assert "\033[" in output
        assert ColoredTextFormatter.RESET in output

    def test_format_correct_color_for_level(self, log_record):
        """Test that correct color is used for log level when TTY is detected."""
        from unittest.mock import patch

        formatter = ColoredTextFormatter()

        # Mock sys.stdout.isatty() to simulate TTY environment
        with patch('sys.stdout.isatty', return_value=True):
            # INFO level
            log_record.levelname = "INFO"
            output = formatter.format(log_record)
            assert ColoredTextFormatter.COLORS["INFO"] in output

            # ERROR level
            log_record.levelname = "ERROR"
            output = formatter.format(log_record)
            assert ColoredTextFormatter.COLORS["ERROR"] in output

    def test_format_no_color_when_not_tty(self, log_record):
        """Test that no color codes are added when not a TTY."""
        from unittest.mock import patch

        formatter = ColoredTextFormatter()

        # Mock sys.stdout.isatty() to simulate non-TTY environment (e.g., file redirect)
        with patch('sys.stdout.isatty', return_value=False):
            output = formatter.format(log_record)

        # Should NOT have color codes
        assert "\033[" not in output
        assert ColoredTextFormatter.RESET not in output
        # But should still have the message
        assert "Test message" in output


@pytest.mark.unit
class TestFormatterIntegration:
    """Integration tests for formatters with logging system."""

    def test_text_formatter_with_handler(self, capfd):
        """Test TextFormatter integrated with StreamHandler."""
        logger = logging.getLogger("test_text_formatter")
        logger.setLevel(logging.INFO)
        logger.handlers = []

        import sys
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(TextFormatter())
        logger.addHandler(handler)

        logger.info("Integration test message")

        captured = capfd.readouterr()
        assert "Integration test message" in captured.out
        assert "INFO" in captured.out

    def test_json_formatter_with_handler(self, capfd):
        """Test JSONFormatter integrated with StreamHandler."""
        logger = logging.getLogger("test_json_formatter")
        logger.setLevel(logging.INFO)
        logger.handlers = []

        import sys
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

        logger.info("JSON integration test")

        captured = capfd.readouterr()

        # Should be valid JSON
        data = json.loads(captured.out.strip())
        assert data["message"] == "JSON integration test"
