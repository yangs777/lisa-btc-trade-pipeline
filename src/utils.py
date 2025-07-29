"""Utility functions for the trading system."""

import logging
import sys
from pathlib import Path
from typing import Any


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up logging configuration.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def validate_config(config: dict[str, Any]) -> bool:
    """Validate configuration dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(config, dict):
        return False

    # Add specific validation rules as needed
    return True


def ensure_directory(path: Path) -> Path:
    """Ensure directory exists.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_number(value: float, decimals: int = 2) -> str:
    """Format number for display.

    Args:
        value: Number to format
        decimals: Decimal places

    Returns:
        Formatted string
    """
    return f"{value:,.{decimals}f}"


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values.

    Args:
        old_value: Previous value
        new_value: Current value

    Returns:
        Percentage change
    """
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100
