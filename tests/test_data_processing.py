"""Tests for data processing modules."""

from unittest.mock import patch

import numpy as np
import pandas as pd


class TestDailyPreprocessor:
    """Test daily preprocessor."""

    @patch("google.cloud.storage.Client")
    def test_preprocessor_init(self, mock_client):
        """Test preprocessor initialization."""
        from src.data_processing.daily_preprocessor import DailyPreprocessor

        preprocessor = DailyPreprocessor(project_id="test-project", bucket_name="test-bucket")

        assert preprocessor.project_id == "test-project"
        assert preprocessor.bucket_name == "test-bucket"

    def test_resample_orderbook(self):
        """Test orderbook resampling."""
        from src.data_processing.daily_preprocessor import DailyPreprocessor

        # Create test data
        timestamps = pd.date_range(start="2024-01-01", periods=100, freq="100ms")
        data = []

        for i, ts in enumerate(timestamps):
            data.append(
                {
                    "timestamp": ts,
                    "best_bid": 50000 + i,
                    "best_ask": 50001 + i,
                    "bid_volume": 0.5,
                    "ask_volume": 0.3,
                }
            )

        df = pd.DataFrame(data)

        # Resample
        with patch("google.cloud.storage.Client"):
            preprocessor = DailyPreprocessor("test", "test")
            resampled = preprocessor._resample_orderbook_data(df)

        # Verify
        assert len(resampled) < len(df)
        assert "vwap" in resampled.columns
        assert "spread" in resampled.columns

    def test_compute_features(self):
        """Test feature computation."""
        from src.data_processing.daily_preprocessor import DailyPreprocessor

        # Create test data
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range(start="2024-01-01", periods=100, freq="1s"),
                "best_bid": np.random.uniform(49000, 51000, 100),
                "best_ask": np.random.uniform(49001, 51001, 100),
                "bid_volume": np.random.uniform(0.1, 1.0, 100),
                "ask_volume": np.random.uniform(0.1, 1.0, 100),
            }
        )

        with patch("google.cloud.storage.Client"):
            preprocessor = DailyPreprocessor("test", "test")

            # Add derived features
            data["mid_price"] = (data["best_bid"] + data["best_ask"]) / 2
            data["spread"] = data["best_ask"] - data["best_bid"]
            data["volume_imbalance"] = (data["bid_volume"] - data["ask_volume"]) / (
                data["bid_volume"] + data["ask_volume"]
            )

            features = preprocessor._compute_additional_features(data)

        # Verify
        assert "price_change" in features.columns
        assert "volatility" in features.columns
        assert len(features) == len(data)


class TestTechnicalIndicators:
    """Test technical indicators."""

    def test_indicators_initialization(self):
        """Test indicators class initialization."""
        from src.features.technical_indicators import TechnicalIndicators

        indicators = TechnicalIndicators()
        assert hasattr(indicators, "add_all_indicators")

    def test_add_all_indicators(self):
        """Test adding all indicators."""
        from src.features.technical_indicators import TechnicalIndicators

        # Create test OHLCV data
        dates = pd.date_range(start="2024-01-01", periods=200, freq="1h")
        df = pd.DataFrame(
            {
                "open": np.random.uniform(49000, 51000, 200),
                "high": np.random.uniform(49500, 51500, 200),
                "low": np.random.uniform(48500, 50500, 200),
                "close": np.random.uniform(49000, 51000, 200),
                "volume": np.random.uniform(100, 1000, 200),
            },
            index=dates,
        )

        # Ensure high >= open/close and low <= open/close
        df["high"] = df[["open", "close", "high"]].max(axis=1)
        df["low"] = df[["open", "close", "low"]].min(axis=1)

        indicators = TechnicalIndicators()
        result = indicators.add_all_indicators(df)

        # Verify some indicators are added
        assert len(result.columns) > len(df.columns)

        # Check for some common indicators
        indicator_columns = result.columns.difference(df.columns)
        assert len(indicator_columns) > 50  # Should have many indicators

    def test_individual_indicators(self):
        """Test individual indicator methods."""
        from src.features.technical_indicators import TechnicalIndicators

        # Create minimal test data
        df = pd.DataFrame({"close": [100, 102, 101, 103, 105, 104, 106]})

        indicators = TechnicalIndicators()

        # Test SMA
        sma = indicators._sma(df["close"], period=3)
        assert len(sma) == len(df)
        assert not sma.iloc[:2].notna().any()  # First 2 should be NaN
        assert sma.iloc[2] == 101.0  # (100 + 102 + 101) / 3

        # Test RSI
        rsi = indicators._rsi(df["close"], period=3)
        assert len(rsi) == len(df)
        assert 0 <= rsi.dropna().max() <= 100
