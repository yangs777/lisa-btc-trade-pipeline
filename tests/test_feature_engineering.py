"""Tests for feature engineering module."""

# mypy: ignore-errors

import sys
from unittest.mock import MagicMock, patch

# Import real numpy to avoid conflicts
import numpy as np
import pytest


# Define mock classes at module level
class MockSeries:
    def __init__(self, data=None, index=None, dtype=None):
        self.data = data or []
        self.index = index or list(range(len(self.data)))
        self.values = self.data
        self.iloc = self
        self.empty = len(self.data) == 0

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.data[idx] if idx < len(self.data) else None
        return self

    def __setitem__(self, idx, value):
        if isinstance(idx, int) and idx < len(self.data):
            self.data[idx] = value

    def rolling(self, window):
        return self

    def mean(self):
        return self

    def std(self):
        return self

    def sum(self):
        return self

    def max(self):
        return self

    def min(self):
        return self

    def ewm(self, span=None, adjust=False):
        return self

    def diff(self):
        return self

    def shift(self, periods=1):
        return self

    def pct_change(self):
        return self

    def cumsum(self):
        return self

    def fillna(self, value=0, method=None):
        return self

    def where(self, cond, other):
        return self

    def replace(self, to_replace, value):
        return self

    def apply(self, func, raw=False):
        return self

    def __len__(self):
        return len(self.data)

    def combine(self, other, func):
        return self

    def abs(self):
        return self

    def var(self):
        return self

    def skew(self):
        return self

    def kurt(self):
        return self

    def corr(self, other):
        return self

    def cov(self, other):
        return self


class MockDataFrame:
    def __init__(self, data=None):
        self.data = data or {}
        self.columns = list(self.data.keys()) if data else []
        self.index = None
        self.empty = len(self.columns) == 0

    def __getitem__(self, key):
        if key in self.data:
            return MockSeries(self.data[key])
        return MockSeries()

    def __setitem__(self, key, value):
        self.data[key] = value
        if key not in self.columns:
            self.columns.append(key)

    def copy(self):
        return MockDataFrame(self.data.copy())

    def __len__(self):
        if self.columns:
            return len(self.data[self.columns[0]])
        return 0


@pytest.fixture(autouse=True)
def mock_pandas_scipy(monkeypatch):
    """Mock pandas and scipy for each test."""
    # Create mocks
    mock_pandas = MagicMock()
    mock_scipy = MagicMock()
    mock_scipy_stats = MagicMock()
    
    # Patch sys.modules
    monkeypatch.setitem(sys.modules, "pandas", mock_pandas)
    monkeypatch.setitem(sys.modules, "scipy", mock_scipy)
    monkeypatch.setitem(sys.modules, "scipy.stats", mock_scipy_stats)
    
    # Use real numpy
    mock_numpy = np

    mock_pandas.Series = MockSeries
    mock_pandas.DataFrame = MockDataFrame
    mock_pandas.concat = lambda dfs, axis=1: MockDataFrame()
    mock_pandas.NA = None
    
    # Mock scipy.stats
    mock_scipy_stats.linregress = lambda x, y: (0.1, 30000, 0.9, 0.01, 0.1)  # slope, intercept, r, p, se
    
    yield
    
    # Cleanup happens automatically with monkeypatch


# Now import the actual modules
from src.feature_engineering import FeatureEngineer  # noqa: E402
from src.feature_engineering.base import (  # noqa: E402
    BaseIndicator,
    OHLCVIndicator,
    PriceIndicator,
    VolumeIndicator,
)
from src.feature_engineering.registry import IndicatorRegistry  # noqa: E402


@pytest.fixture
def sample_ohlcv_df():
    """Create a sample OHLCV dataframe."""
    return MockDataFrame(
        {
            "open": [30000, 30100, 30200, 30150, 30250],
            "high": [30100, 30200, 30300, 30250, 30350],
            "low": [29900, 30000, 30100, 30050, 30150],
            "close": [30050, 30150, 30250, 30200, 30300],
            "volume": [100, 150, 200, 120, 180],
        }
    )


@pytest.fixture
def indicators_yaml(tmp_path):
    """Create a minimal indicators config file."""
    config = {
        "trend": [
            {"name": "SMA_20", "class": "SMA", "params": {"window": 20}},
            {"name": "EMA_20", "class": "EMA", "params": {"window": 20}},
        ],
        "momentum": [{"name": "RSI_14", "class": "RSI", "params": {"window": 14}}],
    }

    config_path = tmp_path / "indicators.yaml"

    # Mock yaml module
    with patch("src.feature_engineering.registry.yaml") as mock_yaml:
        mock_yaml.safe_load.return_value = config

        # Write dummy file
        config_path.write_text("dummy")

        yield config_path


def test_base_indicator_abstract():
    """Test that BaseIndicator is abstract."""
    # Should not be able to instantiate directly
    with pytest.raises(TypeError):
        BaseIndicator()


def test_price_indicator_initialization():
    """Test PriceIndicator initialization."""

    class TestPriceIndicator(PriceIndicator):
        @property
        def name(self):
            return "TEST"

        def transform(self, df):
            return self._get_price(df)

    indicator = TestPriceIndicator(price_col="close", window_size=20)
    assert indicator.price_col == "close"
    assert indicator.window_size == 20
    assert indicator.fillna is True


def test_volume_indicator_get_volume():
    """Test VolumeIndicator volume extraction."""

    class TestVolumeIndicator(VolumeIndicator):
        @property
        def name(self):
            return "TEST"

        def transform(self, df):
            return self._get_volume(df)

    indicator = TestVolumeIndicator()
    df = MockDataFrame({"volume": [100, 200, 300]})

    # Should work with volume column
    volume = indicator.transform(df)
    assert volume is not None

    # Should raise error without volume column
    df_no_volume = MockDataFrame({"close": [100, 200, 300]})
    with pytest.raises(ValueError, match="volume"):
        indicator.transform(df_no_volume)


def test_ohlcv_indicator_validation():
    """Test OHLCV indicator validation."""

    class TestOHLCVIndicator(OHLCVIndicator):
        @property
        def name(self):
            return "TEST"

        def transform(self, df):
            self._validate_ohlcv(df)
            return MockSeries()

    indicator = TestOHLCVIndicator()

    # Should work with all columns
    df_complete = MockDataFrame(
        {"open": [100], "high": [110], "low": [90], "close": [105], "volume": [1000]}
    )
    indicator.transform(df_complete)  # Should not raise

    # Should raise error with missing columns
    df_incomplete = MockDataFrame({"close": [100]})
    with pytest.raises(ValueError, match="Missing required columns"):
        indicator.transform(df_incomplete)


def test_indicator_registry():
    """Test indicator registry functionality."""
    registry = IndicatorRegistry()

    # Test registration
    class TestIndicator(BaseIndicator):
        @property
        def name(self):
            return "TEST"

        def transform(self, df):
            return MockSeries()

    registry.register("TestIndicator", TestIndicator)

    # Test retrieval
    indicator_class = registry.get("TestIndicator")
    assert indicator_class == TestIndicator

    # Test listing
    indicators = registry.list_indicators()
    assert "TestIndicator" in indicators


def test_registry_config_loading(indicators_yaml):
    """Test loading indicator configurations."""
    registry = IndicatorRegistry()

    with patch("builtins.open", create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = "dummy"
        with patch("src.feature_engineering.registry.yaml.safe_load") as mock_yaml:
            mock_yaml.return_value = {
                "trend": [{"name": "SMA_20", "class": "SMA", "params": {"window": 20}}]
            }

            registry.load_config(str(indicators_yaml))

    assert "SMA_20" in registry._configs
    assert registry._configs["SMA_20"]["class"] == "SMA"
    assert registry._configs["SMA_20"]["params"]["window"] == 20


def test_feature_engineer_initialization(indicators_yaml):
    """Test FeatureEngineer initialization."""
    with patch("src.feature_engineering.engineer.Path") as mock_path:
        # Create a mock path chain
        mock_parent_parent_parent = MagicMock()
        mock_parent_parent_parent.__truediv__ = MagicMock(return_value=indicators_yaml)

        mock_parent_parent = MagicMock()
        mock_parent_parent.parent = mock_parent_parent_parent

        mock_parent = MagicMock()
        mock_parent.parent = mock_parent_parent

        mock_path_instance = MagicMock()
        mock_path_instance.parent = mock_parent

        mock_path.return_value = mock_path_instance

        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "dummy"
            with patch("src.feature_engineering.registry.yaml.safe_load") as mock_yaml:
                mock_yaml.return_value = {
                    "trend": [{"name": "SMA_20", "class": "SMA", "params": {"window": 20}}]
                }

                engineer = FeatureEngineer(str(indicators_yaml))

    assert hasattr(engineer, "indicators")
    assert isinstance(engineer.indicators, dict)


def test_feature_engineer_transform(sample_ohlcv_df, indicators_yaml):
    """Test feature transformation."""
    with patch("src.feature_engineering.engineer.Path") as mock_path:
        # Create a mock path chain
        mock_parent_parent_parent = MagicMock()
        mock_parent_parent_parent.__truediv__ = MagicMock(return_value=indicators_yaml)

        mock_parent_parent = MagicMock()
        mock_parent_parent.parent = mock_parent_parent_parent

        mock_parent = MagicMock()
        mock_parent.parent = mock_parent_parent

        mock_path_instance = MagicMock()
        mock_path_instance.parent = mock_parent

        mock_path.return_value = mock_path_instance

        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "dummy"
            with patch("src.feature_engineering.registry.yaml.safe_load") as mock_yaml:
                mock_yaml.return_value = {}  # Empty config

                engineer = FeatureEngineer(str(indicators_yaml))

                # Mock some indicators
                mock_indicator = MagicMock()
                mock_indicator.transform.return_value = MockSeries([1, 2, 3, 4, 5])
                engineer.indicators = {"TEST_IND": mock_indicator}

                # Transform
                result = engineer.transform(sample_ohlcv_df)

    assert "TEST_IND" in result.columns
    mock_indicator.transform.assert_called_once()


def test_feature_engineer_selective_transform(sample_ohlcv_df, indicators_yaml):
    """Test selective feature transformation."""
    with patch("src.feature_engineering.engineer.Path") as mock_path:
        # Create a mock path chain
        mock_parent_parent_parent = MagicMock()
        mock_parent_parent_parent.__truediv__ = MagicMock(return_value=indicators_yaml)

        mock_parent_parent = MagicMock()
        mock_parent_parent.parent = mock_parent_parent_parent

        mock_parent = MagicMock()
        mock_parent.parent = mock_parent_parent

        mock_path_instance = MagicMock()
        mock_path_instance.parent = mock_parent

        mock_path.return_value = mock_path_instance

        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "dummy"
            with patch("src.feature_engineering.registry.yaml.safe_load") as mock_yaml:
                mock_yaml.return_value = {}

                engineer = FeatureEngineer(str(indicators_yaml))

                # Mock indicators
                mock_ind1 = MagicMock()
                mock_ind1.transform.return_value = MockSeries([1, 2, 3, 4, 5])
                mock_ind2 = MagicMock()
                mock_ind2.transform.return_value = MockSeries([5, 4, 3, 2, 1])

                engineer.indicators = {"IND1": mock_ind1, "IND2": mock_ind2}

                # Transform only IND1
                result = engineer.transform_selective(sample_ohlcv_df, ["IND1"])

    assert "IND1" in result.columns
    mock_ind1.transform.assert_called_once()
    mock_ind2.transform.assert_not_called()


def test_feature_engineer_error_handling(sample_ohlcv_df, indicators_yaml):
    """Test error handling in feature transformation."""
    with patch("src.feature_engineering.engineer.Path") as mock_path:
        # Create a mock path chain
        mock_parent_parent_parent = MagicMock()
        mock_parent_parent_parent.__truediv__ = MagicMock(return_value=indicators_yaml)

        mock_parent_parent = MagicMock()
        mock_parent_parent.parent = mock_parent_parent_parent

        mock_parent = MagicMock()
        mock_parent.parent = mock_parent_parent

        mock_path_instance = MagicMock()
        mock_path_instance.parent = mock_parent

        mock_path.return_value = mock_path_instance

        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "dummy"
            with patch("src.feature_engineering.registry.yaml.safe_load") as mock_yaml:
                mock_yaml.return_value = {}

                engineer = FeatureEngineer(str(indicators_yaml))

                # Mock indicator that raises error
                mock_indicator = MagicMock()
                mock_indicator.transform.side_effect = ValueError("Test error")
                engineer.indicators = {"ERROR_IND": mock_indicator}

                # Should not raise, but fill with NA
                result = engineer.transform(sample_ohlcv_df)

    assert "ERROR_IND" in result.columns
    # Value should be pandas.NA (mocked as None)
    assert result.data.get("ERROR_IND") is None


def test_feature_engineer_get_indicator_info(indicators_yaml):
    """Test getting indicator information."""
    with patch("src.feature_engineering.engineer.Path") as mock_path:
        # Create a mock path chain
        mock_parent_parent_parent = MagicMock()
        mock_parent_parent_parent.__truediv__ = MagicMock(return_value=indicators_yaml)

        mock_parent_parent = MagicMock()
        mock_parent_parent.parent = mock_parent_parent_parent

        mock_parent = MagicMock()
        mock_parent.parent = mock_parent_parent

        mock_path_instance = MagicMock()
        mock_path_instance.parent = mock_parent

        mock_path.return_value = mock_path_instance

        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "dummy"
            with patch("src.feature_engineering.registry.yaml.safe_load") as mock_yaml:
                mock_yaml.return_value = {}

                engineer = FeatureEngineer(str(indicators_yaml))

                # Create mock indicators
                class MockIndicator:
                    def __init__(self, window_size=None):
                        self.window_size = window_size

                mock_ind1 = MockIndicator(window_size=20)
                mock_ind2 = MockIndicator()  # No window_size

                engineer.indicators = {"IND1": mock_ind1, "IND2": mock_ind2}

                info = engineer.get_indicator_info()

    assert "IND1" in info
    assert "IND2" in info
    assert info["IND1"]["window_size"] == 20
    assert info["IND2"]["window_size"] is None
    assert info["IND1"]["class"] == "MockIndicator"
    assert info["IND2"]["class"] == "MockIndicator"


def test_all_indicators_registered():
    """Test that all expected indicators are registered."""
    # Clear registry first
    from src.feature_engineering.registry import registry

    registry._indicators = {}
    registry._configs = {}

    with patch("src.feature_engineering.engineer.Path") as mock_path:
        # Create a mock path chain
        mock_parent_parent_parent = MagicMock()
        mock_parent_parent_parent.__truediv__ = MagicMock(return_value="dummy.yaml")

        mock_parent_parent = MagicMock()
        mock_parent_parent.parent = mock_parent_parent_parent

        mock_parent = MagicMock()
        mock_parent.parent = mock_parent_parent

        mock_path_instance = MagicMock()
        mock_path_instance.parent = mock_parent

        mock_path.return_value = mock_path_instance

        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "dummy"
            with patch("src.feature_engineering.registry.yaml.safe_load") as mock_yaml:
                mock_yaml.return_value = {}

                FeatureEngineer()

    # Check key indicators from each category
    expected_indicators = [
        # Trend
        "SMA",
        "EMA",
        "WMA",
        "HMA",
        "TEMA",
        "DEMA",
        "KAMA",
        # Momentum
        "RSI",
        "MACD",
        "CCI",
        "StochasticK",
        # Volatility
        "ATR",
        "BollingerUpper",
        "BollingerLower",
        # Volume
        "OBV",
        "AD",
        "CMF",
        "VWAP",
        # Trend strength
        "ADX",
        "AroonUp",
        "AroonDown",
        # Pattern
        "PSAR",
        "SuperTrend",
        # Statistical
        "StdDev",
        "Variance",
        "Skew",
    ]

    registered = registry._indicators.keys()
    for indicator in expected_indicators:
        assert indicator in registered, f"{indicator} not registered"


def test_transform_preserves_original_columns(sample_ohlcv_df, indicators_yaml):
    """Test that transform preserves original DataFrame columns."""
    with patch("src.feature_engineering.engineer.Path") as mock_path:
        # Create a mock path chain
        mock_parent_parent_parent = MagicMock()
        mock_parent_parent_parent.__truediv__ = MagicMock(return_value=indicators_yaml)

        mock_parent_parent = MagicMock()
        mock_parent_parent.parent = mock_parent_parent_parent

        mock_parent = MagicMock()
        mock_parent.parent = mock_parent_parent

        mock_path_instance = MagicMock()
        mock_path_instance.parent = mock_parent

        mock_path.return_value = mock_path_instance

        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "dummy"
            with patch("src.feature_engineering.registry.yaml.safe_load") as mock_yaml:
                mock_yaml.return_value = {}

                engineer = FeatureEngineer(str(indicators_yaml))

                # Mock indicator
                mock_indicator = MagicMock()
                mock_indicator.transform.return_value = MockSeries([1, 2, 3, 4, 5])
                engineer.indicators = {"NEW_IND": mock_indicator}

                # Transform
                result = engineer.transform(sample_ohlcv_df)

    # Check original columns preserved
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in result.columns

    # Check new indicator added
    assert "NEW_IND" in result.columns


def test_selective_transform_with_missing_indicator(sample_ohlcv_df, indicators_yaml):
    """Test selective transform when requested indicator doesn't exist."""
    with patch("src.feature_engineering.engineer.Path") as mock_path:
        # Create a mock path chain
        mock_parent_parent_parent = MagicMock()
        mock_parent_parent_parent.__truediv__ = MagicMock(return_value=indicators_yaml)

        mock_parent_parent = MagicMock()
        mock_parent_parent.parent = mock_parent_parent_parent

        mock_parent = MagicMock()
        mock_parent.parent = mock_parent_parent

        mock_path_instance = MagicMock()
        mock_path_instance.parent = mock_parent

        mock_path.return_value = mock_path_instance

        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "dummy"
            with patch("src.feature_engineering.registry.yaml.safe_load") as mock_yaml:
                mock_yaml.return_value = {}

                engineer = FeatureEngineer(str(indicators_yaml))

                # Only one indicator available
                mock_indicator = MagicMock()
                mock_indicator.transform.return_value = MockSeries([1, 2, 3, 4, 5])
                engineer.indicators = {"AVAILABLE": mock_indicator}

                # Request both available and missing
                with patch("src.feature_engineering.engineer.logger") as mock_logger:
                    result = engineer.transform_selective(sample_ohlcv_df, ["AVAILABLE", "MISSING"])

                # Should log warning about missing
                mock_logger.warning.assert_called()
                assert "MISSING" in str(mock_logger.warning.call_args)

    # Should still have available indicator
    assert "AVAILABLE" in result.columns
    assert "MISSING" not in result.columns


def test_empty_dataframe_handling(indicators_yaml):
    """Test handling of empty DataFrame."""
    empty_df = MockDataFrame()

    with patch("src.feature_engineering.engineer.Path") as mock_path:
        # Create a mock path chain
        mock_parent_parent_parent = MagicMock()
        mock_parent_parent_parent.__truediv__ = MagicMock(return_value=indicators_yaml)

        mock_parent_parent = MagicMock()
        mock_parent_parent.parent = mock_parent_parent_parent

        mock_parent = MagicMock()
        mock_parent.parent = mock_parent_parent

        mock_path_instance = MagicMock()
        mock_path_instance.parent = mock_parent

        mock_path.return_value = mock_path_instance

        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "dummy"
            with patch("src.feature_engineering.registry.yaml.safe_load") as mock_yaml:
                mock_yaml.return_value = {}

                engineer = FeatureEngineer(str(indicators_yaml))

                # Mock indicator
                mock_indicator = MagicMock()
                mock_indicator.transform.return_value = MockSeries()
                engineer.indicators = {"TEST": mock_indicator}

                # Should handle empty DataFrame
                result = engineer.transform(empty_df)

                assert result.empty


def test_config_loading_error_handling():
    """Test that FeatureEngineer raises error when config is missing."""
    with patch("src.feature_engineering.engineer.Path") as mock_path:
        # Create a mock path chain
        mock_parent_parent_parent = MagicMock()
        mock_parent_parent_parent.__truediv__ = MagicMock(return_value="bad_config.yaml")

        mock_parent_parent = MagicMock()
        mock_parent_parent.parent = mock_parent_parent_parent

        mock_parent = MagicMock()
        mock_parent.parent = mock_parent_parent

        mock_path_instance = MagicMock()
        mock_path_instance.parent = mock_parent

        mock_path.return_value = mock_path_instance

        with patch("builtins.open", create=True) as mock_open:
            # Simulate file error
            mock_open.side_effect = OSError("File not found")

            # Should raise OSError when config file is missing
            with pytest.raises(OSError, match="File not found"):
                FeatureEngineer()


def test_transform_does_not_modify_original(sample_ohlcv_df, indicators_yaml):
    """Test that transform doesn't modify the original DataFrame."""
    with patch("src.feature_engineering.engineer.Path") as mock_path:
        # Create a mock path chain
        mock_parent_parent_parent = MagicMock()
        mock_parent_parent_parent.__truediv__ = MagicMock(return_value=indicators_yaml)

        mock_parent_parent = MagicMock()
        mock_parent_parent.parent = mock_parent_parent_parent

        mock_parent = MagicMock()
        mock_parent.parent = mock_parent_parent

        mock_path_instance = MagicMock()
        mock_path_instance.parent = mock_parent

        mock_path.return_value = mock_path_instance

        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "dummy"
            with patch("src.feature_engineering.registry.yaml.safe_load") as mock_yaml:
                mock_yaml.return_value = {}

                engineer = FeatureEngineer(str(indicators_yaml))

                # Track original columns
                original_columns = list(sample_ohlcv_df.columns)

                # Mock indicator
                mock_indicator = MagicMock()
                mock_indicator.transform.return_value = MockSeries([1, 2, 3, 4, 5])
                engineer.indicators = {"NEW": mock_indicator}

                # Transform
                result = engineer.transform(sample_ohlcv_df)

    # Original should be unchanged
    assert list(sample_ohlcv_df.columns) == original_columns
    assert "NEW" not in sample_ohlcv_df.columns

    # Result should have new column
    assert "NEW" in result.columns