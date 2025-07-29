"""Tests for utility functions."""

import logging
import tempfile
from pathlib import Path

from src.utils import (
    calculate_percentage_change,
    ensure_directory,
    format_number,
    setup_logging,
    validate_config,
)


class TestUtils:
    """Test utility functions."""

    def test_setup_logging(self):
        """Test logging setup."""
        logger = setup_logging("test_logger", logging.DEBUG)
        assert logger.name == "test_logger"
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) > 0

    def test_validate_config(self):
        """Test config validation."""
        # Valid config
        assert validate_config({"key": "value"}) is True
        assert validate_config({}) is True

        # Invalid config
        assert validate_config("not a dict") is False
        assert validate_config(None) is False

    def test_ensure_directory(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "test" / "nested" / "dir"
            result = ensure_directory(test_path)

            assert result.exists()
            assert result.is_dir()

    def test_format_number(self):
        """Test number formatting."""
        assert format_number(1234.5678, 2) == "1,234.57"
        assert format_number(1000000, 0) == "1,000,000"
        assert format_number(0.12345, 4) == "0.1235"

    def test_calculate_percentage_change(self):
        """Test percentage change calculation."""
        assert calculate_percentage_change(100, 110) == 10.0
        assert calculate_percentage_change(100, 90) == -10.0
        assert calculate_percentage_change(0, 100) == 0.0  # Handle zero case
        assert calculate_percentage_change(50, 100) == 100.0
