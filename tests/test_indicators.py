"""Tests for specific indicator implementations."""

import sys
from unittest.mock import MagicMock, patch
import pytest

# Mock external dependencies
sys.modules['numpy'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['scipy'] = MagicMock()
sys.modules['scipy.stats'] = MagicMock()

# Setup mocks
mock_numpy = sys.modules['numpy']
mock_numpy.nan = float('nan')
mock_numpy.sqrt = lambda x: x ** 0.5
mock_numpy.arange = lambda n: list(range(n))
mock_numpy.sum = sum
mock_numpy.arctan = lambda x: x
mock_numpy.pi = 3.14159265359

# Mock pandas Series
class MockSeries:
    def __init__(self, data=None, index=None):
        self.data = data or []
        self.values = self.data
        self.index = index or list(range(len(self.data)))
        self.iloc = self
        
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.data[idx] if idx < len(self.data) else None
        return self
        
    def rolling(self, window):
        return self
        
    def mean(self):
        return MockSeries([30100] * len(self.data))
        
    def std(self):
        return MockSeries([50] * len(self.data))
        
    def ewm(self, span=None, adjust=False):
        return self
        
    def fillna(self, value=0, method=None):
        return self
        
    def diff(self):
        result = [None]
        for i in range(1, len(self.data)):
            result.append(self.data[i] - self.data[i-1])
        return MockSeries(result)
        
    def shift(self, periods=1):
        result = [None] * periods + self.data[:-periods]
        return MockSeries(result)
        
    def where(self, cond, other):
        return self
        
    def __len__(self):
        return len(self.data)
        
    def max(self):
        return MockSeries([30200] * len(self.data))
        
    def min(self):
        return MockSeries([30000] * len(self.data))


class MockDataFrame:
    def __init__(self, data=None):
        self.data = data or {}
        self.columns = list(self.data.keys())
        
    def __getitem__(self, key):
        if key in self.data:
            return MockSeries(self.data[key])
        return MockSeries()
        
    def copy(self):
        return MockDataFrame(self.data.copy())


sys.modules['pandas'].Series = MockSeries
sys.modules['pandas'].DataFrame = MockDataFrame
sys.modules['pandas'].concat = lambda dfs, axis=1: MockDataFrame()

# Import indicators after mocking
from src.feature_engineering.trend.moving_averages import SMA, EMA, WMA
from src.feature_engineering.momentum.oscillators import RSI, CCI
from src.feature_engineering.volatility.bands import BollingerUpper, BollingerLower
from src.feature_engineering.volume.classic import OBV
from src.feature_engineering.trend_strength.adx import ADX


@pytest.fixture
def sample_df():
    """Create a sample OHLCV dataframe."""
    return MockDataFrame({
        'open': [30000, 30100, 30200, 30150, 30250],
        'high': [30100, 30200, 30300, 30250, 30350],
        'low': [29900, 30000, 30100, 30050, 30150],
        'close': [30050, 30150, 30250, 30200, 30300],
        'volume': [100, 150, 200, 120, 180]
    })


def test_sma_indicator(sample_df):
    """Test SMA indicator."""
    sma = SMA(window=3)
    assert sma.name == "SMA_3"
    
    result = sma.transform(sample_df)
    assert isinstance(result, MockSeries)
    assert len(result) == len(sample_df)


def test_ema_indicator(sample_df):
    """Test EMA indicator."""
    ema = EMA(window=5)
    assert ema.name == "EMA_5"
    
    result = ema.transform(sample_df)
    assert isinstance(result, MockSeries)
    assert len(result) == len(sample_df)


def test_rsi_indicator(sample_df):
    """Test RSI indicator."""
    rsi = RSI(window=14)
    assert rsi.name == "RSI_14"
    
    result = rsi.transform(sample_df)
    assert isinstance(result, MockSeries)
    assert len(result) == len(sample_df)


def test_rsi_price_column_validation(sample_df):
    """Test RSI with missing price column."""
    rsi = RSI(price_col="invalid")
    
    with pytest.raises(ValueError, match="Column 'invalid' not found"):
        rsi.transform(sample_df)


def test_cci_indicator(sample_df):
    """Test CCI indicator."""
    cci = CCI(window=20)
    assert cci.name == "CCI_20"
    
    # CCI requires OHLCV validation
    result = cci.transform(sample_df)
    assert isinstance(result, MockSeries)


def test_bollinger_bands(sample_df):
    """Test Bollinger Bands indicators."""
    bb_upper = BollingerUpper(window=20, std=2)
    bb_lower = BollingerLower(window=20, std=2)
    
    assert bb_upper.name == "BB_UPPER_20_2"
    assert bb_lower.name == "BB_LOWER_20_2"
    
    upper = bb_upper.transform(sample_df)
    lower = bb_lower.transform(sample_df)
    
    assert isinstance(upper, MockSeries)
    assert isinstance(lower, MockSeries)


def test_obv_indicator(sample_df):
    """Test OBV indicator."""
    obv = OBV()
    assert obv.name == "OBV"
    
    result = obv.transform(sample_df)
    assert isinstance(result, MockSeries)
    assert len(result) == len(sample_df)


def test_obv_missing_columns():
    """Test OBV with missing columns."""
    obv = OBV()
    
    # Missing close column
    df_no_close = MockDataFrame({'volume': [100, 200]})
    with pytest.raises(ValueError, match="close"):
        obv.transform(df_no_close)
    
    # Missing volume column
    df_no_volume = MockDataFrame({'close': [100, 200]})
    with pytest.raises(ValueError, match="volume"):
        obv.transform(df_no_volume)


def test_indicator_fillna_handling(sample_df):
    """Test NaN handling in indicators."""
    # Test with fillna=True (default)
    sma_fill = SMA(window=3, fillna=True)
    result_fill = sma_fill.transform(sample_df)
    assert isinstance(result_fill, MockSeries)
    
    # Test with fillna=False
    sma_no_fill = SMA(window=3, fillna=False)
    result_no_fill = sma_no_fill.transform(sample_df)
    assert isinstance(result_no_fill, MockSeries)


def test_indicator_parameter_validation():
    """Test indicator parameter validation."""
    # Valid parameters
    sma = SMA(window=20)
    assert sma.window_size == 20
    
    ema = EMA(window=50)
    assert ema.window_size == 50
    
    rsi = RSI(window=7)
    assert rsi.window_size == 7


def test_ohlcv_validation_detailed():
    """Test detailed OHLCV validation."""
    from src.feature_engineering.volatility.atr import ATR
    
    atr = ATR(window=14)
    
    # Complete OHLCV data
    df_complete = MockDataFrame({
        'open': [100], 'high': [110], 'low': [90],
        'close': [105], 'volume': [1000]
    })
    result = atr.transform(df_complete)
    assert isinstance(result, MockSeries)
    
    # Missing required columns
    df_missing_high = MockDataFrame({
        'open': [100], 'low': [90],
        'close': [105], 'volume': [1000]
    })
    with pytest.raises(ValueError, match="Missing required columns"):
        atr.transform(df_missing_high)