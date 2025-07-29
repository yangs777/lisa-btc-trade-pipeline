"""Main entry point for the Bitcoin trading pipeline."""

import argparse
import sys
from pathlib import Path

from .utils import setup_logging

logger = setup_logging(__name__)


def main(args: list[str] | None = None):
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Bitcoin Trading Pipeline with τ-SAC")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Data collection command
    collect_parser = subparsers.add_parser("collect", help="Collect orderbook data from Binance")
    collect_parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    collect_parser.add_argument(
        "--duration", type=int, default=3600, help="Collection duration in seconds"
    )

    # Preprocessing command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess collected data")
    preprocess_parser.add_argument("--date", required=True, help="Date to process (YYYY-MM-DD)")
    preprocess_parser.add_argument("--bucket", required=True, help="GCS bucket name")

    # Training command
    train_parser = subparsers.add_parser("train", help="Train the τ-SAC model")
    train_parser.add_argument(
        "--config", type=Path, default="config/train.yaml", help="Training config"
    )
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")

    # API server command
    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI prediction server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Server host")
    serve_parser.add_argument("--port", type=int, default=8000, help="Server port")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # Parse arguments
    args = parser.parse_args(args)

    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    try:
        if args.command == "collect":
            logger.info(f"Starting data collection for {args.symbol}")
            # Import here to avoid circular imports
            from .data_collection.binance_connector import BinanceOrderbookCollector

            collector = BinanceOrderbookCollector(symbol=args.symbol, save_interval=60)
            collector.start(duration=args.duration)

        elif args.command == "preprocess":
            logger.info(f"Preprocessing data for {args.date}")
            from .data_processing.daily_preprocessor import DailyPreprocessor

            preprocessor = DailyPreprocessor(project_id="your-project", bucket_name=args.bucket)
            preprocessor.process_date(args.date)

        elif args.command == "train":
            logger.info("Starting model training")
            # Training logic would go here
            logger.info("Training not implemented yet")

        elif args.command == "serve":
            logger.info(f"Starting API server on {args.host}:{args.port}")
            import uvicorn

            from .api import create_app

            app = create_app()
            uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
