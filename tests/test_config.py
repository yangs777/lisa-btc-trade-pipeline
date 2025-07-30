"""Test coverage for configuration module."""

import pytest
import os
from unittest.mock import patch
from typing import Dict, Any
from src.config import load_config, get_env_var, validate_config


class TestConfiguration:
    """Test configuration functions."""
    
    def test_load_config_defaults(self) -> None:
        """Test loading default configuration."""
        config = load_config()
        
        assert "project_id" in config
        assert "bucket_name" in config
        assert "credentials_path" in config
        assert "symbol" in config
        assert "depth_levels" in config
        assert "buffer_size" in config
        assert "upload_workers" in config
        assert "cleanup_after_upload" in config
        
        # Check default values
        assert config["symbol"] == "btcusdt"
        assert config["depth_levels"] == 20
        assert config["buffer_size"] == 1000
        assert config["upload_workers"] == 2
        assert config["cleanup_after_upload"] == True
    
    def test_load_config_with_env_vars(self) -> None:
        """Test loading config with environment variables."""
        env_vars = {
            "GCP_PROJECT_ID": "test-project",
            "GCS_BUCKET": "test-bucket",
            "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/creds.json"
        }
        
        with patch.dict(os.environ, env_vars):
            config = load_config()
            
            assert config["project_id"] == "test-project"
            assert config["bucket_name"] == "test-bucket"
            assert config["credentials_path"] == "/path/to/creds.json"
    
    def test_get_env_var_exists(self) -> None:
        """Test getting existing environment variable."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            value = get_env_var("TEST_VAR")
            assert value == "test_value"
    
    def test_get_env_var_not_exists(self) -> None:
        """Test getting non-existent environment variable."""
        value = get_env_var("NON_EXISTENT_VAR")
        assert value is None
    
    def test_get_env_var_with_default(self) -> None:
        """Test getting environment variable with default."""
        value = get_env_var("NON_EXISTENT_VAR", "default_value")
        assert value == "default_value"
        
        # Test that existing var overrides default
        with patch.dict(os.environ, {"EXISTING_VAR": "actual_value"}):
            value = get_env_var("EXISTING_VAR", "default_value")
            assert value == "actual_value"
    
    def test_validate_config_valid(self) -> None:
        """Test validating valid configuration."""
        config = {
            "project_id": "valid-project",
            "bucket_name": "valid-bucket",
            "credentials_path": "/path/to/creds.json",
            "symbol": "btcusdt",
            "depth_levels": 20,
            "buffer_size": 1000,
            "upload_workers": 2,
            "cleanup_after_upload": True
        }
        
        # Should return True for valid config
        assert validate_config(config) == True
    
    def test_validate_config_missing_keys(self) -> None:
        """Test validating config with missing keys."""
        config = {
            "project_id": "valid-project",
            # Missing other required keys
        }
        
        # Should return False for incomplete config
        assert validate_config(config) == False
    
    def test_validate_config_invalid_types(self) -> None:
        """Test validating config with invalid types."""
        config = {
            "project_id": "valid-project",
            "bucket_name": "valid-bucket",
            "credentials_path": "/path/to/creds.json",
            "symbol": "btcusdt",
            "depth_levels": "twenty",  # Should be int
            "buffer_size": 1000,
            "upload_workers": 2,
            "cleanup_after_upload": True
        }
        
        # Should return False for invalid types
        assert validate_config(config) == False
    
    def test_validate_config_invalid_values(self) -> None:
        """Test validating config with invalid values."""
        config = {
            "project_id": "",  # Empty string
            "bucket_name": "valid-bucket",
            "credentials_path": "/path/to/creds.json",
            "symbol": "btcusdt",
            "depth_levels": -5,  # Negative value
            "buffer_size": 0,  # Zero buffer
            "upload_workers": 0,  # Zero workers
            "cleanup_after_upload": True
        }
        
        # Should return False for invalid values
        assert validate_config(config) == False
    
    def test_config_consistency(self) -> None:
        """Test that config remains consistent across calls."""
        config1 = load_config()
        config2 = load_config()
        
        # Should return same values
        assert config1 == config2
        
        # Test that modifying returned dict doesn't affect subsequent calls
        config1["symbol"] = "modified"
        config3 = load_config()
        assert config3["symbol"] == "btcusdt"  # Should still be original value
    
    def test_config_paths(self) -> None:
        """Test that config paths are properly set."""
        from src.config import PROJECT_ROOT, DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR
        
        assert PROJECT_ROOT.exists()
        assert DATA_DIR == PROJECT_ROOT / "data"
        assert RAW_DATA_DIR == DATA_DIR / "raw"
        assert PROCESSED_DATA_DIR == DATA_DIR / "processed"
        
        # Check that directories were created
        assert RAW_DATA_DIR.exists()
        assert PROCESSED_DATA_DIR.exists()
    
    def test_config_constants(self) -> None:
        """Test configuration constants."""
        from src.config import (
            BINANCE_SYMBOL,
            BINANCE_DEPTH_LEVELS,
            BINANCE_BUFFER_SIZE,
            UPLOAD_WORKERS,
            CLEANUP_AFTER_UPLOAD
        )
        
        assert BINANCE_SYMBOL == "btcusdt"
        assert BINANCE_DEPTH_LEVELS == 20
        assert BINANCE_BUFFER_SIZE == 1000
        assert UPLOAD_WORKERS == 2
        assert CLEANUP_AFTER_UPLOAD == True
        
        # Check types
        assert isinstance(BINANCE_SYMBOL, str)
        assert isinstance(BINANCE_DEPTH_LEVELS, int)
        assert isinstance(BINANCE_BUFFER_SIZE, int)
        assert isinstance(UPLOAD_WORKERS, int)
        assert isinstance(CLEANUP_AFTER_UPLOAD, bool)