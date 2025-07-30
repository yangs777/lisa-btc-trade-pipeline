from typing import Dict, List, Any, Optional, Union, Tuple

"""Main entry point for the Bitcoin trading pipeline."""

import argparse
import logging
import sys
from pathlib import Path

from src.utils import setup_logging


logger = setup_logging(__name__)


def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Bitcoin Trading System with τ-SAC",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Bitcoin Trading System v1.0.0"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Collect command
    collect_parser = subparsers.add_parser("collect", help="Collect market data")
    collect_parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    collect_parser.add_argument("--duration", type=int, default=3600, help="Duration in seconds")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--config", required=True, help="Training config file")
    train_parser.add_argument("--data", help="Training data path")
    train_parser.add_argument("--output", help="Model output path")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start prediction server")
    serve_parser.add_argument("--port", type=int, default=8000, help="Server port")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Server host")
    serve_parser.add_argument("--workers", type=int, default=2, help="Number of workers")
    
    return parser.parse_args(args)


def run_collection(args):
    """Run data collection."""
    logger.info(f"Starting data collection for {args.symbol}")
    
    # Import here to avoid circular imports
    from src.data_collection.binance_websocket import BinanceWebSocket
    from src.data_collection.gcs_uploader import GCSUploader
    import asyncio
    
    async def collect():
        ws = BinanceWebSocket(args.symbol.lower())
        uploader = GCSUploader("btc-orderbook-data")
        
        await ws.connect()
        
        # Collect for specified duration
        import time
        start_time = time.time()
        
        while time.time() - start_time < args.duration:
            try:
                data = await ws.get_orderbook_update()
                if data:
                    # Upload to GCS
                    timestamp = int(time.time())
                    filename = f"orderbook/{args.symbol}/{timestamp}.json"
                    uploader.upload_json(filename, data)
            except Exception as e:
                logger.error(f"Error collecting data: {e}")
                await asyncio.sleep(1)
        
        await ws.disconnect()
    
    asyncio.run(collect())
    logger.info("Data collection completed")


def run_training(args):
    """Run model training."""
    logger.info(f"Starting training with config: {args.config}")
    
    # Import training modules
    from src.rl.training import TauSACTrainer, TrainingConfig
    import yaml
    
    # Load config
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = TrainingConfig(**config_dict)
    
    # Create trainer
    trainer = TauSACTrainer(config=config)
    
    # Run training
    trainer.train()
    
    # Save model
    if args.output:
        trainer.save_model(args.output)
    
    logger.info("Training completed")


def run_server(args):
    """Run prediction server."""
    logger.info(f"Starting prediction server on {args.host}:{args.port}")
    
    # Import server module
    import uvicorn
    from src.api.prediction_server import create_app
    
    app = create_app()
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )


def main():
    """Main entry point."""
    args = parse_args()
    
    if not args.command:
        logger.error("No command specified. Use --help for usage.")
        sys.exit(1)
    
    try:
        if args.command == "collect":
            run_collection(args)
        elif args.command == "train":
            run_training(args)
        elif args.command == "serve":
            run_server(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()