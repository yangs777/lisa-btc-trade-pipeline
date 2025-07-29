"""Configuration settings for the BTC trading pipeline."""

import os
from pathlib import Path

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


def load_config():
    """Load configuration from environment."""
    return {
        "project_id": GCP_PROJECT_ID,
        "bucket_name": GCS_BUCKET,
        "credentials_path": GCP_CREDENTIALS_PATH,
        "symbol": BINANCE_SYMBOL,
        "depth_levels": BINANCE_DEPTH_LEVELS,
        "buffer_size": BINANCE_BUFFER_SIZE,
        "upload_workers": UPLOAD_WORKERS,
        "cleanup_after_upload": CLEANUP_AFTER_UPLOAD,
    }


def get_env_var(name: str, default: str = None) -> str:
    """Get environment variable with optional default."""
    return os.environ.get(name, default)


def validate_config(config: dict) -> bool:
    """Validate configuration dictionary."""
    if not isinstance(config, dict):
        return False
    
    required_keys = ["api_key", "bucket_name"]
    for key in required_keys:
        if key not in config or config[key] is None:
            return False
    
    return True
