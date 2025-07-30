"""Tests for data_processing module."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_data_validator():
    """Test data validation."""
    from src.data_processing.validator import DataValidator
    
    validator = DataValidator()
    
    # Test valid data
    df = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [101, 102, 103],
        'low': [99, 100, 101],
        'close': [100.5, 101.5, 102.5],
        'volume': [1000, 1100, 1200]
    })
    
    is_valid = validator.validate(df)
    assert is_valid is True
    
    # Test invalid data (negative prices)
    df_invalid = pd.DataFrame({
        'open': [100, -101, 102],
        'high': [101, 102, 103],
        'low': [99, 100, 101],
        'close': [100.5, 101.5, 102.5],
        'volume': [1000, 1100, 1200]
    })
    
    is_valid = validator.validate(df_invalid)
    assert is_valid is False


def test_data_preprocessing():
    """Test data preprocessing."""
    from src.data_processing.daily_preprocessor import DailyPreprocessor
    
    preprocessor = DailyPreprocessor()
    
    # Create sample data
    df = pd.DataFrame({
        'open': [100, 101, 102, 103],
        'high': [101, 102, 103, 104],
        'low': [99, 100, 101, 102],
        'close': [100.5, 101.5, 102.5, 103.5],
        'volume': [1000, 1100, 1200, 1300]
    })
    
    # Process data
    processed = preprocessor.process(df)
    
    assert processed is not None
    assert len(processed) == len(df)
    assert 'returns' in processed.columns
