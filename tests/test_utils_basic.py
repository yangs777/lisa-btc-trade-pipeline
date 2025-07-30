"""Tests for utils module."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_setup_logger():
    """Test logger setup."""
    from src.utils import setup_logger
    logger = setup_logger("test")
    assert logger is not None
    assert logger.name == "test"


def test_get_project_root():
    """Test project root detection."""
    from src.utils import get_project_root
    root = get_project_root()
    assert root is not None
    assert isinstance(root, Path)
    assert root.exists()


def test_load_config():
    """Test config loading."""
    from src.utils import load_config
    with patch("builtins.open", create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = '{"test": "value"}'
        config = load_config("test.yaml")
        assert config == {"test": "value"}
