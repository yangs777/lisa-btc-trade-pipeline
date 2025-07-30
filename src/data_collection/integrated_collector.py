"""Integrated data collection system combining Binance WebSocket and GCS upload."""

import asyncio
import logging
import signal
from collections.abc import Callable

from .binance_websocket import BinanceWebSocketCollector
from .gcs_uploader import GCSUploader

logger = logging.getLogger(__name__)


class IntegratedDataCollector:
    """Combines Binance WebSocket data collection with automatic GCS upload."""

    def __init__(
        self,
        # Binance WebSocket settings
        symbol: str = "btcusdt",
        depth_levels: int = 20,
        buffer_size: int = 1000,
        local_data_dir: str = "./data/raw",
        # GCS settings
        bucket_name: str = "btc-orderbook-data",
        project_id: str = "my-project-779482",
        credentials_path: str | None = None,
        cleanup_after_upload: bool = True,
        upload_workers: int = 2,
    ):
        """Initialize integrated data collector.

        Args:
            symbol: Trading pair symbol
            depth_levels: Orderbook depth levels (5, 10, or 20)
            buffer_size: Buffer size before writing to disk
            local_data_dir: Directory to save collected data
            bucket_name: GCS bucket name
            project_id: GCP project ID
            credentials_path: Path to service account JSON file
            cleanup_after_upload: Whether to delete local files after upload
            upload_workers: Number of concurrent upload workers
        """
        self.local_data_dir = local_data_dir
        self.upload_workers = upload_workers

        # Initialize collectors
        self.websocket_collector = BinanceWebSocketCollector(
            symbol=symbol,
            depth_levels=depth_levels,
            buffer_size=buffer_size,
            output_dir=local_data_dir,
        )

        self.gcs_uploader = GCSUploader(
            bucket_name=bucket_name,
            project_id=project_id,
            credentials_path=credentials_path,
            local_data_dir=local_data_dir,
            cleanup_after_upload=cleanup_after_upload,
        )

        self._tasks: list[asyncio.Task] = []
        self._running = False

    async def start(self) -> None:
        """Start integrated data collection and upload."""
        self._running = True
        logger.info("Starting integrated data collection system...")

        # Start WebSocket collector
        websocket_task = asyncio.create_task(self.websocket_collector.start())
        self._tasks.append(websocket_task)

        # Wait a bit for initial data collection
        await asyncio.sleep(5)

        # Start GCS uploader
        uploader_task = asyncio.create_task(
            self.gcs_uploader.start(num_workers=self.upload_workers)
        )
        self._tasks.append(uploader_task)

        # Wait for all tasks
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            logger.info("Integrated collection cancelled")

    async def stop(self) -> None:
        """Stop integrated data collection and upload."""
        logger.info("Stopping integrated data collection system...")
        self._running = False

        # Stop collectors in order
        await self.websocket_collector.stop()
        await self.gcs_uploader.stop()

        # Log combined statistics
        ws_stats = self.websocket_collector.get_stats()
        gcs_stats = self.gcs_uploader.get_stats()

        logger.info(
            f"Integrated collection stopped. Combined statistics:\n"
            f"  WebSocket Collection:\n"
            f"    Orderbook updates: {ws_stats['orderbook_count']}\n"
            f"    Trades: {ws_stats['trade_count']}\n"
            f"    Aggregated trades: {ws_stats['agg_trade_count']}\n"
            f"  GCS Upload:\n"
            f"    Files uploaded: {gcs_stats['files_uploaded']}\n"
            f"    Bytes uploaded: {gcs_stats['bytes_uploaded']:,}\n"
            f"    Files failed: {gcs_stats['files_failed']}"
        )

    def get_stats(self) -> dict:
        """Get combined statistics from both collectors."""
        return {
            "websocket": self.websocket_collector.get_stats(),
            "gcs": self.gcs_uploader.get_stats(),
        }


async def main() -> None:
    """Run integrated data collection with signal handling."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create integrated collector with config
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.config import (  # noqa: I001
        BINANCE_SYMBOL,
        BINANCE_DEPTH_LEVELS,
        BINANCE_BUFFER_SIZE,
        GCP_PROJECT_ID,
        GCS_BUCKET,
        GCP_CREDENTIALS_PATH,
        RAW_DATA_DIR,
        UPLOAD_WORKERS,
        CLEANUP_AFTER_UPLOAD,
    )

    collector = IntegratedDataCollector(
        symbol=BINANCE_SYMBOL,
        depth_levels=BINANCE_DEPTH_LEVELS,
        buffer_size=BINANCE_BUFFER_SIZE,
        local_data_dir=str(RAW_DATA_DIR),
        bucket_name=GCS_BUCKET,
        project_id=GCP_PROJECT_ID,
        credentials_path=GCP_CREDENTIALS_PATH,
        cleanup_after_upload=CLEANUP_AFTER_UPLOAD,
        upload_workers=UPLOAD_WORKERS,
    )

    # Setup signal handlers
    loop = asyncio.get_event_loop()

    def signal_handler(sig: int) -> None:
        logger.info(f"Received signal {sig}, shutting down...")
        # Store reference to avoid RUF006
        _ = asyncio.create_task(collector.stop())  # noqa: RUF006

    for sig in (signal.SIGTERM, signal.SIGINT):
        # Create a closure to capture the signal value
        def make_handler(s: int) -> Callable[[], None]:
            def handler() -> None:
                signal_handler(s)

            return handler

        loop.add_signal_handler(sig, make_handler(sig))

    try:
        # Run collector
        await collector.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        await collector.stop()


if __name__ == "__main__":
    asyncio.run(main())
