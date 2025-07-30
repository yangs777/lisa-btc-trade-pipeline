"""Integration tests for the trading pipeline."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_end_to_end_pipeline():
    """Test end-to-end pipeline execution."""
    # Create sample data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='h'),
        'open': np.random.uniform(40000, 50000, 1000),
        'high': np.random.uniform(40100, 50100, 1000),
        'low': np.random.uniform(39900, 49900, 1000),
        'close': np.random.uniform(40000, 50000, 1000),
        'volume': np.random.uniform(100, 200, 1000)
    })
    
    # Test data validation
    from src.data_processing.validator import DataValidator
    validator = DataValidator()
    assert validator.validate(data) is True
    
    # Test feature engineering
    from src.feature_engineering.engineer import FeatureEngineer
    engineer = FeatureEngineer({})
    features = engineer.transform(data)
    assert features is not None
    assert len(features) > 0
    
    # Test risk management
    from src.risk_management import RiskManager
    risk_config = {
        "max_position_size": 0.1,
        "max_drawdown": 0.2,
        "risk_per_trade": 0.02
    }
    risk_manager = RiskManager(risk_config)
    
    position_size = risk_manager.calculate_position_size(
        capital=10000,
        price=45000,
        stop_loss_pct=0.02
    )
    assert position_size > 0
    assert position_size <= 1000  # 10% of capital


def test_config_loading_integration():
    """Test configuration loading and usage."""
    from src.utils import load_config
    
    # Mock config file
    mock_config = {
        "trading": {
            "initial_capital": 10000,
            "commission": 0.001
        },
        "risk": {
            "max_position_size": 0.1,
            "max_drawdown": 0.2
        },
        "features": {
            "indicators": ["sma", "rsi", "macd"]
        }
    }
    
    with patch("src.utils.load_config") as mock_load:
        mock_load.return_value = mock_config
        
        config = load_config("config.yaml")
        
        assert config["trading"]["initial_capital"] == 10000
        assert config["risk"]["max_position_size"] == 0.1
        assert "sma" in config["features"]["indicators"]
