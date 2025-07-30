"""Test coverage for utils module."""

import pytest
import logging
import sys
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import tempfile
import shutil

from src.utils import (
    setup_logging,
    validate_config,
    ensure_directory,
    format_number,
    calculate_percentage_change,
    get_project_root,
    load_yaml_config,
    validate_data
)


class TestLoggingUtils:
    """Test logging utilities."""
    
    def test_setup_logging_default(self) -> None:
        """Test default logging setup."""
        logger = setup_logging("test_module")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0
        
        # Clean up handler
        logger.handlers.clear()
    
    def test_setup_logging_custom_level(self) -> None:
        """Test logging with custom level."""
        logger = setup_logging("test_debug", level=logging.DEBUG)
        
        assert logger.level == logging.DEBUG
        
        # Clean up handler
        logger.handlers.clear()
    
    def test_setup_logging_no_duplicate_handlers(self) -> None:
        """Test that multiple calls don't create duplicate handlers."""
        logger = setup_logging("test_duplicate")
        initial_handlers = len(logger.handlers)
        
        # Call again
        logger = setup_logging("test_duplicate")
        assert len(logger.handlers) == initial_handlers
        
        # Clean up handler
        logger.handlers.clear()
    
    def test_setup_logging_formatter(self) -> None:
        """Test logging formatter."""
        logger = setup_logging("test_formatter")
        
        # Check handler has formatter
        assert len(logger.handlers) > 0
        handler = logger.handlers[0]
        assert handler.formatter is not None
        
        # Test format string
        format_str = handler.formatter._fmt
        assert "%(asctime)s" in format_str
        assert "%(name)s" in format_str
        assert "%(levelname)s" in format_str
        assert "%(message)s" in format_str
        
        # Clean up handler
        logger.handlers.clear()


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_validate_config_valid_dict(self) -> None:
        """Test validation of valid config dict."""
        config = {"key": "value", "number": 42}
        assert validate_config(config) == True
    
    def test_validate_config_empty_dict(self) -> None:
        """Test validation of empty dict."""
        config = {}
        assert validate_config(config) == True  # Empty dict is valid
    
    def test_validate_config_not_dict(self) -> None:
        """Test validation of non-dict types."""
        assert validate_config("not a dict") == False
        assert validate_config(123) == False
        assert validate_config([1, 2, 3]) == False
        assert validate_config(None) == False
    
    def test_validate_config_nested_dict(self) -> None:
        """Test validation of nested dict."""
        config = {
            "level1": {
                "level2": {
                    "key": "value"
                }
            }
        }
        assert validate_config(config) == True


class TestFileSystemUtils:
    """Test file system utilities."""
    
    def test_ensure_directory_new(self) -> None:
        """Test creating new directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "new" / "nested" / "dir"
            assert not test_path.exists()
            
            result = ensure_directory(test_path)
            
            assert test_path.exists()
            assert test_path.is_dir()
            assert result == test_path
    
    def test_ensure_directory_existing(self) -> None:
        """Test ensure_directory with existing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir)
            assert test_path.exists()
            
            result = ensure_directory(test_path)
            
            assert test_path.exists()
            assert result == test_path
    
    def test_ensure_directory_string_path(self) -> None:
        """Test ensure_directory with string path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = f"{tmpdir}/string_path"
            
            result = ensure_directory(test_path)
            
            assert Path(test_path).exists()
            assert isinstance(result, Path)
    
    def test_get_project_root(self) -> None:
        """Test getting project root."""
        root = get_project_root()
        
        assert isinstance(root, Path)
        assert root.exists()
        
        # Should be two levels up from src/utils.py
        # Project root should have src directory
        assert (root / "src").exists()
        assert (root / "src" / "utils.py").exists()


class TestNumberFormatting:
    """Test number formatting utilities."""
    
    def test_format_number_default(self) -> None:
        """Test default number formatting."""
        assert format_number(1234.567) == "1,234.57"
        assert format_number(1000000) == "1,000,000.00"
        assert format_number(0.123) == "0.12"
    
    def test_format_number_custom_decimals(self) -> None:
        """Test formatting with custom decimal places."""
        assert format_number(1234.5678, decimals=3) == "1,234.568"
        assert format_number(1234.5678, decimals=0) == "1,235"
        assert format_number(1234.5678, decimals=4) == "1,234.5678"
    
    def test_format_number_negative(self) -> None:
        """Test formatting negative numbers."""
        assert format_number(-1234.56) == "-1,234.56"
        assert format_number(-1000000.123) == "-1,000,000.12"
    
    def test_format_number_zero(self) -> None:
        """Test formatting zero."""
        assert format_number(0) == "0.00"
        assert format_number(0, decimals=5) == "0.00000"


class TestCalculations:
    """Test calculation utilities."""
    
    def test_calculate_percentage_change_positive(self) -> None:
        """Test positive percentage change."""
        result = calculate_percentage_change(100, 150)
        assert result == 50.0
        
        result = calculate_percentage_change(50, 55)
        assert result == 10.0
    
    def test_calculate_percentage_change_negative(self) -> None:
        """Test negative percentage change."""
        result = calculate_percentage_change(100, 75)
        assert result == -25.0
        
        result = calculate_percentage_change(200, 100)
        assert result == -50.0
    
    def test_calculate_percentage_change_zero_old_value(self) -> None:
        """Test percentage change with zero old value."""
        result = calculate_percentage_change(0, 100)
        assert result == 0.0  # Defined as 0 to avoid division by zero
    
    def test_calculate_percentage_change_same_values(self) -> None:
        """Test percentage change with same values."""
        result = calculate_percentage_change(100, 100)
        assert result == 0.0
    
    def test_calculate_percentage_change_fractional(self) -> None:
        """Test percentage change with fractional values."""
        result = calculate_percentage_change(0.1, 0.15)
        assert abs(result - 50.0) < 0.0001
        
        result = calculate_percentage_change(1.5, 1.2)
        assert abs(result - (-20.0)) < 0.0001


class TestYamlConfig:
    """Test YAML configuration loading."""
    
    def test_load_yaml_config_valid(self) -> None:
        """Test loading valid YAML config."""
        yaml_content = """
api:
  key: test_key
  secret: test_secret
trading:
  symbol: BTCUSDT
  interval: 1h
"""
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            config = load_yaml_config("config.yaml")
            
            assert config["api"]["key"] == "test_key"
            assert config["api"]["secret"] == "test_secret"
            assert config["trading"]["symbol"] == "BTCUSDT"
            assert config["trading"]["interval"] == "1h"
    
    def test_load_yaml_config_empty(self) -> None:
        """Test loading empty YAML file."""
        with patch("builtins.open", mock_open(read_data="")):
            config = load_yaml_config("empty.yaml")
            assert config == {}
    
    def test_load_yaml_config_none(self) -> None:
        """Test loading YAML that returns None."""
        with patch("builtins.open", mock_open(read_data="# Just a comment")):
            config = load_yaml_config("comment.yaml")
            assert config == {}
    
    def test_load_yaml_config_file_not_found(self) -> None:
        """Test loading non-existent YAML file."""
        with pytest.raises(FileNotFoundError):
            load_yaml_config("missing.yaml")
    
    def test_load_yaml_config_invalid_yaml(self) -> None:
        """Test loading invalid YAML."""
        yaml_content = """
invalid: yaml: content: here
    bad indentation
"""
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with pytest.raises(Exception):  # yaml.YAMLError
                load_yaml_config("invalid.yaml")


class TestDataValidation:
    """Test data validation utilities."""
    
    def test_validate_data_none(self) -> None:
        """Test validation of None."""
        assert validate_data(None) == False
    
    def test_validate_data_empty_list(self) -> None:
        """Test validation of empty list."""
        assert validate_data([]) == False
    
    def test_validate_data_empty_dict(self) -> None:
        """Test validation of empty dict."""
        assert validate_data({}) == False
    
    def test_validate_data_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_data("") == False
    
    def test_validate_data_valid_list(self) -> None:
        """Test validation of non-empty list."""
        assert validate_data([1, 2, 3]) == True
        assert validate_data(["a", "b"]) == True
    
    def test_validate_data_valid_dict(self) -> None:
        """Test validation of non-empty dict."""
        assert validate_data({"key": "value"}) == True
        assert validate_data({1: "one", 2: "two"}) == True
    
    def test_validate_data_valid_string(self) -> None:
        """Test validation of non-empty string."""
        assert validate_data("test") == True
        assert validate_data("a") == True
    
    def test_validate_data_number(self) -> None:
        """Test validation of numbers."""
        assert validate_data(42) == True
        assert validate_data(0) == True
        assert validate_data(-1) == True
        assert validate_data(3.14) == True
    
    def test_validate_data_boolean(self) -> None:
        """Test validation of boolean values."""
        assert validate_data(True) == True
        assert validate_data(False) == True
    
    def test_validate_data_custom_object(self) -> None:
        """Test validation of custom objects."""
        class CustomObj:
            def __len__(self):
                return 5
        
        assert validate_data(CustomObj()) == True
        
        class EmptyObj:
            def __len__(self):
                return 0
        
        assert validate_data(EmptyObj()) == False
EOF < /dev/null
