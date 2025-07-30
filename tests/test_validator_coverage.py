"""Tests for data validation module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data_processing.validator import DataValidator


class TestDataValidator:
    """Test cases for DataValidator."""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return DataValidator()
    
    @pytest.fixture
    def valid_ohlcv(self):
        """Create valid OHLCV data."""
        return pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 101.5, 103.0],
            'high': [101.0, 102.5, 103.0, 102.0, 104.0],
            'low': [99.0, 100.5, 101.0, 100.0, 102.0],
            'close': [100.5, 102.0, 101.5, 101.0, 103.5],
            'volume': [1000.0, 1100.0, 950.0, 1050.0, 1200.0]
        })
    
    def test_init(self, validator):
        """Test validator initialization."""
        assert validator is not None
        assert len(validator.validation_rules) == 5
        assert 'price_positive' in validator.validation_rules
        assert 'high_low_consistency' in validator.validation_rules
        assert 'high_consistency' in validator.validation_rules
        assert 'low_consistency' in validator.validation_rules
        assert 'volume_positive' in validator.validation_rules
    
    def test_validate_ohlcv_valid(self, validator, valid_ohlcv):
        """Test validation of valid OHLCV data."""
        assert validator.validate_ohlcv(valid_ohlcv) is True
    
    def test_validate_ohlcv_missing_columns(self, validator):
        """Test validation with missing columns."""
        # Missing 'volume' column
        data = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [101.0, 102.0],
            'low': [99.0, 100.0],
            'close': [100.5, 101.5]
        })
        assert validator.validate_ohlcv(data) is False
        
        # Missing multiple columns
        data = pd.DataFrame({
            'open': [100.0, 101.0],
            'close': [100.5, 101.5]
        })
        assert validator.validate_ohlcv(data) is False
    
    def test_validate_ohlcv_empty_dataframe(self, validator):
        """Test validation of empty DataFrame."""
        data = pd.DataFrame({
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': []
        })
        assert validator.validate_ohlcv(data) is False
    
    def test_validate_ohlcv_negative_prices(self, validator):
        """Test validation with negative prices."""
        data = pd.DataFrame({
            'open': [100.0, -101.0, 102.0],
            'high': [101.0, 102.0, 103.0],
            'low': [99.0, 100.0, 101.0],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000.0, 1100.0, 950.0]
        })
        assert validator.validate_ohlcv(data) is False
    
    def test_validate_ohlcv_high_low_inconsistency(self, validator):
        """Test validation with high < low."""
        data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [101.0, 102.0, 103.0],
            'low': [99.0, 103.0, 101.0],  # low > high at index 1
            'close': [100.5, 101.5, 102.5],
            'volume': [1000.0, 1100.0, 950.0]
        })
        assert validator.validate_ohlcv(data) is False
    
    def test_validate_ohlcv_high_not_highest(self, validator):
        """Test validation with high not being highest price."""
        data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [101.0, 102.0, 103.0],
            'low': [99.0, 100.0, 101.0],
            'close': [102.0, 101.5, 102.5],  # close > high at index 0
            'volume': [1000.0, 1100.0, 950.0]
        })
        data.loc[0, 'high'] = 101.0  # Make high < close
        assert validator.validate_ohlcv(data) is False
    
    def test_validate_ohlcv_low_not_lowest(self, validator):
        """Test validation with low not being lowest price."""
        data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [101.0, 102.0, 103.0],
            'low': [99.0, 100.0, 101.0],
            'close': [100.5, 99.5, 102.5],  # close < low at index 1
            'volume': [1000.0, 1100.0, 950.0]
        })
        assert validator.validate_ohlcv(data) is False
    
    def test_validate_ohlcv_negative_volume(self, validator):
        """Test validation with negative volume."""
        data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [101.0, 102.0, 103.0],
            'low': [99.0, 100.0, 101.0],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000.0, -100.0, 950.0]  # Negative volume
        })
        assert validator.validate_ohlcv(data) is False
    
    def test_validate_ohlcv_zero_volume(self, validator):
        """Test validation with zero volume (should be valid)."""
        data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [101.0, 102.0, 103.0],
            'low': [99.0, 100.0, 101.0],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000.0, 0.0, 950.0]  # Zero volume is allowed
        })
        assert validator.validate_ohlcv(data) is True
    
    def test_validate_timestamps_valid(self, validator):
        """Test validation of valid timestamps."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='h'),
            'value': [1, 2, 3, 4, 5]
        })
        assert validator.validate_timestamps(data) is True
    
    def test_validate_timestamps_non_monotonic(self, validator):
        """Test validation of non-monotonic timestamps."""
        timestamps = pd.date_range('2024-01-01', periods=5, freq='h')
        # Swap two timestamps
        timestamps = list(timestamps)
        timestamps[2], timestamps[3] = timestamps[3], timestamps[2]
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'value': [1, 2, 3, 4, 5]
        })
        assert validator.validate_timestamps(data) is False
    
    def test_validate_timestamps_missing_column(self, validator):
        """Test validation with missing timestamp column."""
        data = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=5, freq='h'),
            'value': [1, 2, 3, 4, 5]
        })
        assert validator.validate_timestamps(data) is False
    
    def test_validate_timestamps_custom_column(self, validator):
        """Test validation with custom timestamp column."""
        data = pd.DataFrame({
            'custom_time': pd.date_range('2024-01-01', periods=5, freq='h'),
            'value': [1, 2, 3, 4, 5]
        })
        assert validator.validate_timestamps(data, 'custom_time') is True
    
    def test_get_validation_report_valid(self, validator, valid_ohlcv):
        """Test validation report for valid data."""
        report = validator.get_validation_report(valid_ohlcv)
        
        assert report['is_valid'] is True
        assert len(report['errors']) == 0
        assert 'stats' in report
        assert report['stats']['num_records'] == 5
        assert report['stats']['price_range'] == [100.5, 103.5]
        assert report['stats']['volume_range'] == [950.0, 1200.0]
    
    def test_get_validation_report_invalid(self, validator):
        """Test validation report for invalid data."""
        # Invalid data with negative prices
        data = pd.DataFrame({
            'open': [100.0, -101.0],
            'high': [101.0, 102.0],
            'low': [99.0, 100.0],
            'close': [100.5, 101.5],
            'volume': [1000.0, 1100.0]
        })
        
        report = validator.get_validation_report(data)
        
        assert report['is_valid'] is False
        assert len(report['errors']) > 0
        assert 'OHLCV validation failed' in report['errors']
        # Stats should still be calculated
        assert report['stats']['num_records'] == 2
    
    def test_get_validation_report_missing_columns(self, validator):
        """Test validation report with missing columns."""
        data = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [101.0, 102.0]
        })
        
        report = validator.get_validation_report(data)
        
        assert report['is_valid'] is False
        assert len(report['errors']) > 0
        # Stats should be empty due to missing columns
        assert 'price_range' not in report['stats']
    
    def test_validation_rules_exception_handling(self, validator):
        """Test that exceptions in validation rules are handled."""
        # Data that might cause exceptions
        data = pd.DataFrame({
            'open': [100.0, np.nan, 102.0],  # NaN value
            'high': [101.0, 102.0, 103.0],
            'low': [99.0, 100.0, 101.0],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000.0, 1100.0, 950.0]
        })
        
        # Should return False instead of raising exception
        assert validator.validate_ohlcv(data) is False
    
    def test_edge_case_single_row(self, validator):
        """Test validation with single row of data."""
        data = pd.DataFrame({
            'open': [100.0],
            'high': [101.0],
            'low': [99.0],
            'close': [100.5],
            'volume': [1000.0]
        })
        
        assert validator.validate_ohlcv(data) is True
        
        report = validator.get_validation_report(data)
        assert report['is_valid'] is True
        assert report['stats']['num_records'] == 1