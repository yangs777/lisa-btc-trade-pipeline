"""Tests for daily preprocessor."""

# mypy: ignore-errors

import json
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

# Mock external dependencies
sys.modules["google"] = MagicMock()
sys.modules["google.cloud"] = MagicMock()
sys.modules["google.cloud.storage"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["pandas"] = MagicMock()


# Create mock DataFrame class
class MockDataFrame:
    def __init__(self, data=None):
        self.data = data or []
        self.columns = []
        self.empty = len(self.data) == 0
        self.index = None

    def __len__(self):
        return len(self.data)

    def iloc(self, idx):
        return self.data[idx] if idx < len(self.data) else None

    def __getitem__(self, key):
        return self

    def resample(self, freq):
        return self

    def sum(self):
        return self

    def count(self):
        return self

    def mean(self):
        return self

    def last(self):
        return self

    def agg(self, *args, **kwargs):
        return self

    def ohlc(self):
        return {"open": self, "high": self, "low": self, "close": self}

    def fillna(self, *args, **kwargs):
        return self

    def sort_index(self):
        return self

    def set_index(self, *args, **kwargs):
        return self

    def to_parquet(self, *args, **kwargs):
        pass


# Replace pandas.DataFrame with our mock
sys.modules["pandas"].DataFrame = MockDataFrame
sys.modules["pandas"].concat = lambda x: MockDataFrame() if not x else x[0]
sys.modules["pandas"].merge = lambda *args, **kwargs: MockDataFrame()
sys.modules["pandas"].to_datetime = lambda x, **kwargs: x

from src.data_processing.daily_preprocessor import DailyPreprocessor  # noqa: E402


@pytest.fixture
def preprocessor(tmp_path):
    """Create a DailyPreprocessor instance with mocked GCS client."""
    with patch("src.data_processing.daily_preprocessor.storage.Client") as mock_client:
        mock_bucket = MagicMock()
        mock_client.return_value.bucket.return_value = mock_bucket

        preprocessor = DailyPreprocessor(
            bucket_name="test-bucket",
            project_id="test-project",
            local_work_dir=str(tmp_path),
        )
        preprocessor.bucket = mock_bucket

        yield preprocessor


@pytest.fixture
def sample_orderbook_file(tmp_path):
    """Create a sample orderbook JSONL file."""
    file_path = tmp_path / "orderbook_test.jsonl"

    data = [
        {
            "timestamp": 1700000000000,
            "event_time": 1700000000000,
            "symbol": "BTCUSDT",
            "bids": [["30000.00", "0.5"], ["29999.00", "1.0"], ["29998.00", "1.5"]],
            "asks": [["30001.00", "0.5"], ["30002.00", "1.0"], ["30003.00", "1.5"]],
        },
        {
            "timestamp": 1700000001000,
            "event_time": 1700000001000,
            "symbol": "BTCUSDT",
            "bids": [["30000.50", "0.6"], ["29999.50", "1.1"], ["29998.50", "1.6"]],
            "asks": [["30001.50", "0.6"], ["30002.50", "1.1"], ["30003.50", "1.6"]],
        },
    ]

    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    return file_path


@pytest.fixture
def sample_trade_file(tmp_path):
    """Create a sample trade JSONL file."""
    file_path = tmp_path / "trades_test.jsonl"

    data = [
        {
            "timestamp": 1700000000500,
            "event_time": 1700000000500,
            "symbol": "BTCUSDT",
            "trade_id": 123456789,
            "price": "30000.50",
            "quantity": "0.1",
            "is_buyer_maker": False,
        },
        {
            "timestamp": 1700000001500,
            "event_time": 1700000001500,
            "symbol": "BTCUSDT",
            "trade_id": 123456790,
            "price": "30001.00",
            "quantity": "0.2",
            "is_buyer_maker": True,
        },
    ]

    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    return file_path


def test_parse_orderbook_data(preprocessor, sample_orderbook_file):
    """Test parsing orderbook data."""
    # Mock the DataFrame to have expected data
    mock_df = MockDataFrame()
    mock_df.empty = False
    mock_df.data = [
        {"mid_price": 30000.5, "spread": 1.0, "best_bid": 30000.0, "best_ask": 30001.0},
        {"mid_price": 30001.0, "spread": 1.0, "best_bid": 30000.5, "best_ask": 30001.5},
    ]
    mock_df.columns = ["mid_price", "spread", "order_imbalance", "best_bid", "best_ask"]

    with patch.object(preprocessor, "_parse_orderbook_data", return_value=mock_df):
        df = preprocessor._parse_orderbook_data(sample_orderbook_file)

        assert not df.empty
        assert len(df) == 2
        assert "mid_price" in df.columns
        assert "spread" in df.columns
        assert "order_imbalance" in df.columns


def test_parse_trade_data(preprocessor, sample_trade_file):
    """Test parsing trade data."""
    # Mock the DataFrame
    mock_df = MockDataFrame()
    mock_df.empty = False
    mock_df.data = [
        {"price": 30000.5, "quantity": 0.1, "is_buyer_maker": False},
        {"price": 30001.0, "quantity": 0.2, "is_buyer_maker": True},
    ]
    mock_df.columns = ["price", "quantity", "is_buyer_maker"]

    with patch.object(preprocessor, "_parse_trade_data", return_value=mock_df):
        df = preprocessor._parse_trade_data(sample_trade_file)

        assert not df.empty
        assert len(df) == 2
        assert "price" in df.columns
        assert "quantity" in df.columns
        assert "is_buyer_maker" in df.columns


def test_aggregate_trade_features(preprocessor, sample_trade_file):
    """Test aggregating trade features."""
    # Create mock trades DataFrame
    mock_trades = MockDataFrame()
    mock_trades.empty = False
    mock_trades.columns = ["price", "quantity", "trade_id", "is_buyer_maker"]

    # Mock aggregated features
    mock_features = MockDataFrame()
    mock_features.empty = False
    mock_features.columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "buy_volume",
        "sell_volume",
        "vwap",
    ]

    with patch.object(preprocessor, "_aggregate_trade_features", return_value=mock_features):
        features_df = preprocessor._aggregate_trade_features(mock_trades, freq="1s")

        assert not features_df.empty
        assert "open" in features_df.columns
        assert "high" in features_df.columns
        assert "low" in features_df.columns
        assert "close" in features_df.columns
        assert "volume" in features_df.columns
        assert "buy_volume" in features_df.columns
        assert "sell_volume" in features_df.columns
        assert "vwap" in features_df.columns


def test_merge_data(preprocessor):
    """Test merging orderbook and trade data."""
    # Create mock dataframes
    mock_orderbook = MockDataFrame()
    mock_orderbook.empty = False
    mock_orderbook.columns = ["mid_price", "spread", "order_imbalance"]

    mock_trades = MockDataFrame()
    mock_trades.empty = False
    mock_trades.columns = ["volume", "vwap"]

    # Mock merged result
    mock_merged = MockDataFrame()
    mock_merged.empty = False
    mock_merged.columns = ["mid_price", "spread", "order_imbalance", "volume", "vwap"]

    with patch.object(preprocessor, "_merge_data", return_value=mock_merged):
        merged_df = preprocessor._merge_data(mock_orderbook, mock_trades, freq="1s")

        assert not merged_df.empty
        # Should have columns from both datasets
        assert "mid_price" in merged_df.columns  # From orderbook
        assert "volume" in merged_df.columns  # From trades
        assert "order_imbalance" in merged_df.columns


@pytest.mark.asyncio
async def test_process_date_no_data(preprocessor):
    """Test processing date with no data."""
    # Mock empty blob list
    preprocessor.bucket.list_blobs.return_value = []

    result = await preprocessor.process_date(datetime(2023, 11, 15, tzinfo=timezone.utc))

    assert result is None


@pytest.mark.asyncio
async def test_process_date_with_data(preprocessor, tmp_path):
    """Test processing date with data."""
    # Create test files
    orderbook_file = tmp_path / "orderbook_test.jsonl"
    trade_file = tmp_path / "trades_test.jsonl"

    # Write test data
    orderbook_data = {
        "timestamp": 1700000000000,
        "event_time": 1700000000000,
        "bids": [["30000.00", "0.5"]],
        "asks": [["30001.00", "0.5"]],
    }
    trade_data = {
        "timestamp": 1700000000500,
        "event_time": 1700000000500,
        "trade_id": 123456789,
        "price": "30000.50",
        "quantity": "0.1",
        "is_buyer_maker": False,
    }

    with open(orderbook_file, "w") as f:
        f.write(json.dumps(orderbook_data) + "\n")
    with open(trade_file, "w") as f:
        f.write(json.dumps(trade_data) + "\n")

    # Mock GCS operations
    mock_blobs = [
        MagicMock(name="raw/2023/11/15/orderbook_test.jsonl"),
        MagicMock(name="raw/2023/11/15/trades_test.jsonl"),
    ]
    preprocessor.bucket.list_blobs.return_value = mock_blobs

    # Mock download
    def mock_download(blob_name, filename):
        if "orderbook" in blob_name:
            with open(orderbook_file, "rb") as src:
                with open(filename, "wb") as dst:
                    dst.write(src.read())
        else:
            with open(trade_file, "rb") as src:
                with open(filename, "wb") as dst:
                    dst.write(src.read())

    preprocessor._download_blob = MagicMock(side_effect=mock_download)
    preprocessor._upload_blob = MagicMock()

    # Process date
    result = await preprocessor.process_date(datetime(2023, 11, 15, tzinfo=timezone.utc))

    assert result is not None
    assert "processed/2023/11/15/btcusdt_20231115_1min.parquet" in result
    preprocessor._upload_blob.assert_called_once()


def test_empty_dataframe_handling(preprocessor):
    """Test handling of empty dataframes."""
    empty_df = MockDataFrame()
    empty_df.empty = True

    # Test empty orderbook parsing
    with patch.object(preprocessor, "_aggregate_trade_features", return_value=empty_df):
        result = preprocessor._aggregate_trade_features(empty_df)
        assert result.empty

    # Test merge with empty dataframes
    with patch.object(preprocessor, "_merge_data", return_value=empty_df):
        merged = preprocessor._merge_data(empty_df, empty_df)
        assert merged.empty
