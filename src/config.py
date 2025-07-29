"""Configuration settings for the BTC trading pipeline."""

import os
from pathlib import Path  # noqa: I001


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# GCP settings
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "my-project-779482")
GCS_BUCKET = os.environ.get("GCS_BUCKET", "btc-orderbook-data")
GCP_CREDENTIALS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

# Binance settings
BINANCE_SYMBOL = "btcusdt"
BINANCE_DEPTH_LEVELS = 20
BINANCE_BUFFER_SIZE = 1000

# Data collection settings
UPLOAD_WORKERS = 2
CLEANUP_AFTER_UPLOAD = True

# Create directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
