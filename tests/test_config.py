"""Test coverage for configuration module."""

import pytest
import os
from unittest.mock import patch, mock_open
from src.config import load_config, get_env_var, validate_config


class TestConfig:
    """Test configuration functions."""
    
    def test_get_env_var_exists(self) -> None:
        """Test getting existing environment variable."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            value = get_env_var("TEST_VAR")
            assert value == "test_value"
    
    def test_get_env_var_with_default(self) -> None:
        """Test getting env var with default."""
        with patch.dict(os.environ, {}, clear=True):
            value = get_env_var("MISSING_VAR", "default_value")
            assert value == "default_value"
    
    def test_get_env_var_none(self) -> None:
        """Test getting non-existent env var without default."""
        with patch.dict(os.environ, {}, clear=True):
            value = get_env_var("MISSING_VAR")
            assert value is None
    
    def test_load_config_from_env(self) -> None:
        """Test loading configuration from environment."""
        env_vars = {
            "BINANCE_API_KEY": "test_key",
            "BINANCE_API_SECRET": "test_secret",
            "MODEL_PATH": "models/test.pkl",
            "LOG_LEVEL": "DEBUG",
            "RISK_MAX_POSITION_SIZE": "0.5",
            "TRADING_SYMBOL": "ETHUSDT"
        }
        
        with patch.dict(os.environ, env_vars):
            config = load_config()
            
            assert config["binance"]["api_key"] == "test_key"
            assert config["binance"]["api_secret"] == "test_secret"
            assert config["model"]["path"] == "models/test.pkl"
            assert config["logging"]["level"] == "DEBUG"
            assert config["risk"]["max_position_size"] == 0.5
            assert config["trading"]["symbol"] == "ETHUSDT"
    
    def test_load_config_defaults(self) -> None:
        """Test loading configuration with defaults."""
        with patch.dict(os.environ, {}, clear=True):
            config = load_config()
            
            # Check defaults
            assert config["binance"]["api_key"] == ""
            assert config["binance"]["api_secret"] == ""
            assert config["model"]["path"] == "models/tau_sac_btc.pkl"
            assert config["logging"]["level"] == "INFO"
            assert config["risk"]["max_position_size"] == 0.1
            assert config["trading"]["symbol"] == "BTCUSDT"
    
    def test_load_config_from_file(self) -> None:
        """Test loading configuration from file."""
        config_content = """{
            "binance": {
                "api_key": "file_key",
                "api_secret": "file_secret"
            },
            "model": {
                "path": "models/file_model.pkl"
            },
            "trading": {
                "symbol": "BTCUSDT",
                "interval": "5m"
            }
        }"""
        
        with patch("builtins.open", mock_open(read_data=config_content)):
            with patch("os.path.exists", return_value=True):
                config = load_config("config.json")
                
                assert config["binance"]["api_key"] == "file_key"
                assert config["binance"]["api_secret"] == "file_secret"
                assert config["model"]["path"] == "models/file_model.pkl"
                assert config["trading"]["interval"] == "5m"
    
    def test_load_config_file_not_found(self) -> None:
        """Test loading config when file doesn't exist."""
        with patch("os.path.exists", return_value=False):
            config = load_config("missing.json")
            
            # Should return defaults
            assert config["model"]["path"] == "models/tau_sac_btc.pkl"
    
    def test_load_config_invalid_json(self) -> None:
        """Test loading config with invalid JSON."""
        with patch("builtins.open", mock_open(read_data="invalid json")):
            with patch("os.path.exists", return_value=True):
                config = load_config("invalid.json")
                
                # Should return defaults on error
                assert config["model"]["path"] == "models/tau_sac_btc.pkl"
    
    def test_validate_config_valid(self) -> None:
        """Test validating valid configuration."""
        config = {
            "binance": {
                "api_key": "key",
                "api_secret": "secret"
            },
            "model": {
                "path": "model.pkl"
            },
            "risk": {
                "max_position_size": 0.1,
                "max_drawdown": 0.2
            },
            "trading": {
                "symbol": "BTCUSDT",
                "interval": "1h"
            }
        }
        
        # Should not raise any exceptions
        validate_config(config)
    
    def test_validate_config_missing_keys(self) -> None:
        """Test validating config with missing keys."""
        config = {
            "binance": {
                "api_key": "key"
                # Missing api_secret
            }
        }
        
        with pytest.raises(KeyError):
            validate_config(config)
    
    def test_validate_config_invalid_values(self) -> None:
        """Test validating config with invalid values."""
        config = {
            "binance": {
                "api_key": "",  # Empty key
                "api_secret": "secret"
            },
            "model": {
                "path": "model.pkl"
            },
            "risk": {
                "max_position_size": 1.5,  # Too large
                "max_drawdown": 0.2
            }
        }
        
        with pytest.raises(ValueError):
            validate_config(config)
    
    def test_load_config_with_env_override(self) -> None:
        """Test that environment variables override file config."""
        config_content = """{
            "binance": {
                "api_key": "file_key"
            }
        }"""
        
        env_vars = {
            "BINANCE_API_KEY": "env_key"  # Should override file
        }
        
        with patch("builtins.open", mock_open(read_data=config_content)):
            with patch("os.path.exists", return_value=True):
                with patch.dict(os.environ, env_vars):
                    config = load_config("config.json")
                    
                    # Environment should take precedence
                    assert config["binance"]["api_key"] == "env_key"
    
    def test_config_type_conversions(self) -> None:
        """Test type conversions in config loading."""
        env_vars = {
            "RISK_MAX_POSITION_SIZE": "0.25",  # String to float
            "RISK_MAX_DRAWDOWN": "0.15",
            "TRADING_LEVERAGE": "3",  # String to int
            "ENABLE_PAPER_TRADING": "true",  # String to bool
            "LOG_LEVEL": "DEBUG"  # String stays string
        }
        
        with patch.dict(os.environ, env_vars):
            config = load_config()
            
            assert isinstance(config["risk"]["max_position_size"], float)
            assert config["risk"]["max_position_size"] == 0.25
            assert isinstance(config["risk"]["max_drawdown"], float)
            assert config["risk"]["max_drawdown"] == 0.15
            
            # Check if trading config exists and has expected types
            if "leverage" in config.get("trading", {}):
                assert isinstance(config["trading"]["leverage"], int)
            if "paper_trading" in config.get("trading", {}):
                assert isinstance(config["trading"]["paper_trading"], bool)