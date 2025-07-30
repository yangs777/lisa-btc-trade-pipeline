"""Tests for data collection modules."""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch


class TestBinanceConnector:
    """Test Binance WebSocket connector."""

    @patch("websocket.WebSocketApp")
    def test_binance_connector_init(self, mock_ws) -> None:
        """Test connector initialization."""
        from src.data_collection.binance_connector import BinanceOrderbookCollector

        collector = BinanceOrderbookCollector(
            symbol="BTCUSDT", save_interval=60, output_dir=Path("/tmp")
        )

        assert collector.symbol == "BTCUSDT"
        assert collector.save_interval == 60
        assert collector.output_dir == Path("/tmp")

    @patch("websocket.WebSocketApp")
    def test_on_message_handling(self, mock_ws) -> None:
        """Test message handling."""
        from src.data_collection.binance_connector import BinanceOrderbookCollector

        collector = BinanceOrderbookCollector("BTCUSDT")

        # Mock message
        message = json.dumps(
            {
                "e": "depthUpdate",
                "E": 1234567890,
                "s": "BTCUSDT",
                "U": 1,
                "u": 2,
                "b": [["50000", "0.5"]],
                "a": [["50001", "0.3"]],
            }
        )

        # Process message
        collector.on_message(None, message)

        # Check buffer
        assert len(collector.orderbook_buffer) > 0

    @patch("websocket.WebSocketApp")
    @patch("aiofiles.open", new_callable=MagicMock)
    @patch("asyncio.run")
    def test_save_buffer(self, mock_run, mock_aiofiles, mock_ws) -> None:
        """Test buffer saving."""
        from src.data_collection.binance_connector import BinanceOrderbookCollector

        collector = BinanceOrderbookCollector("BTCUSDT", output_dir=Path("/tmp"))

        # Add test data
        collector.orderbook_buffer.append({"test": "data"})

        # Mock the async save
        mock_run.side_effect = lambda coro: None

        # Save buffer
        collector._save_buffer_sync()

        # Verify async save was called
        mock_run.assert_called_once()


class TestGCSUploader:
    """Test Google Cloud Storage uploader."""

    @patch("google.cloud.storage.Client")
    def test_gcs_uploader_init(self, mock_client) -> None:
        """Test uploader initialization."""
        from src.data_collection.gcs_uploader import GCSUploader

        uploader = GCSUploader(project_id="test-project", bucket_name="test-bucket")

        assert uploader.project_id == "test-project"
        assert uploader.bucket_name == "test-bucket"
        mock_client.assert_called_once()

    @patch("google.cloud.storage.Client")
    def test_upload_file(self, mock_client) -> None:
        """Test file upload."""
        from src.data_collection.gcs_uploader import GCSUploader

        # Setup mocks
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        uploader = GCSUploader("test-project", "test-bucket")

        # Upload file
        import asyncio
        asyncio.run(uploader.upload_file(str(Path("/tmp/test.json"))))

        # Verify - the actual implementation creates blob differently
        mock_bucket.blob.assert_called_once()

