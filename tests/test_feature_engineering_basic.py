"""Tests for feature_engineering module."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_feature_engineer():
    """Test feature engineering."""
    from src.feature_engineering.engineer import FeatureEngineer
    
    config = {
        "indicators": ["sma", "rsi", "macd"],
        "lookback_periods": [5, 10, 20]
    }
    
    engineer = FeatureEngineer(config)
    
    # Create sample data
    df = pd.DataFrame({
        'open': np.random.uniform(100, 110, 100),
        'high': np.random.uniform(101, 111, 100),
        'low': np.random.uniform(99, 109, 100),
        'close': np.random.uniform(100, 110, 100),
        'volume': np.random.uniform(1000, 2000, 100)
    })
    
    # Generate features
    features = engineer.transform(df)
    
    assert features is not None
    assert len(features) > 0
    assert features.shape[1] > df.shape[1]  # More columns after feature engineering


def test_indicator_registry():
    """Test indicator registry."""
    from src.feature_engineering.registry import IndicatorRegistry
    
    registry = IndicatorRegistry()
    
    # Test registration
    def custom_indicator(df):
        return df['close'].rolling(5).mean()
    
    registry.register("custom_sma", custom_indicator)
    
    # Test retrieval
    indicator_func = registry.get("custom_sma")
    assert indicator_func is not None
    
    # Test listing
    indicators = registry.list()
    assert "custom_sma" in indicators
