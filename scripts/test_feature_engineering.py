#!/usr/bin/env python3
"""Test script for feature engineering functionality."""

import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_data(n_rows: int = 1000) -> pd.DataFrame:
    """Generate sample OHLCV data for testing.
    
    Args:
        n_rows: Number of rows to generate
        
    Returns:
        DataFrame with OHLCV columns
    """
    # Generate timestamps
    start_time = datetime.now() - timedelta(minutes=n_rows)
    timestamps = pd.date_range(start=start_time, periods=n_rows, freq='1min')
    
    # Generate realistic OHLCV data
    np.random.seed(42)
    
    # Start with a base price
    base_price = 30000
    
    # Generate returns
    returns = np.random.normal(0.0001, 0.001, n_rows)
    prices = base_price * np.cumprod(1 + returns)
    
    # Generate OHLCV
    data = {
        'timestamp': timestamps,
        'open': prices * (1 + np.random.normal(0, 0.0005, n_rows)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.001, n_rows))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.001, n_rows))),
        'close': prices,
        'volume': np.abs(np.random.normal(1000, 200, n_rows))
    }
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    # Ensure high >= low and high >= open/close, low <= open/close
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


def test_individual_indicators(df: pd.DataFrame):
    """Test individual indicator calculations."""
    logger.info("Testing individual indicators...")
    
    # Test a few indicators manually
    from src.feature_engineering.trend.moving_averages import SMA, EMA
    from src.feature_engineering.momentum.oscillators import RSI
    from src.feature_engineering.volatility.atr import ATR
    from src.feature_engineering.volume.classic import OBV
    
    # SMA
    sma = SMA(window=20)
    sma_result = sma.transform(df)
    logger.info(f"SMA_20 - First 5 values: {sma_result.head()}")
    logger.info(f"SMA_20 - Non-NaN count: {sma_result.notna().sum()}")
    
    # EMA
    ema = EMA(window=20)
    ema_result = ema.transform(df)
    logger.info(f"EMA_20 - First 5 values: {ema_result.head()}")
    
    # RSI
    rsi = RSI(window=14)
    rsi_result = rsi.transform(df)
    logger.info(f"RSI_14 - Range: [{rsi_result.min():.2f}, {rsi_result.max():.2f}]")
    
    # ATR
    atr = ATR(window=14)
    atr_result = atr.transform(df)
    logger.info(f"ATR_14 - Mean: {atr_result.mean():.4f}")
    
    # OBV
    obv = OBV()
    obv_result = obv.transform(df)
    logger.info(f"OBV - Last value: {obv_result.iloc[-1]:.0f}")


def test_feature_engineer(df: pd.DataFrame):
    """Test FeatureEngineer with all indicators."""
    logger.info("\nTesting FeatureEngineer...")
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Get indicator info
    info = engineer.get_indicator_info()
    logger.info(f"Total indicators loaded: {len(info)}")
    
    # Show some indicators by category
    categories = {}
    for name, details in info.items():
        module = details['module']
        category = module.split('.')[-2] if 'feature_engineering' in module else 'unknown'
        if category not in categories:
            categories[category] = []
        categories[category].append(name)
    
    for category, indicators in categories.items():
        logger.info(f"{category}: {len(indicators)} indicators")
        logger.info(f"  Examples: {', '.join(indicators[:5])}")
    
    # Transform with all indicators
    logger.info("\nCalculating all indicators...")
    result_df = engineer.transform(df)
    
    # Check results
    logger.info(f"Original columns: {len(df.columns)}")
    logger.info(f"Total columns after transformation: {len(result_df.columns)}")
    logger.info(f"New indicator columns: {len(result_df.columns) - len(df.columns)}")
    
    # Check for NaN values
    nan_counts = result_df.isna().sum()
    indicators_with_nan = nan_counts[nan_counts > 0]
    logger.info(f"Indicators with NaN values: {len(indicators_with_nan)}")
    
    # Show sample of results
    indicator_cols = [col for col in result_df.columns if col not in df.columns]
    sample_indicators = indicator_cols[:10]
    logger.info("\nSample indicator values (last row):")
    for ind in sample_indicators:
        value = result_df[ind].iloc[-1]
        if pd.notna(value):
            logger.info(f"  {ind}: {value:.4f}")


def test_selective_transform(df: pd.DataFrame):
    """Test selective indicator transformation."""
    logger.info("\nTesting selective transformation...")
    
    engineer = FeatureEngineer()
    
    # Select specific indicators
    selected = [
        'SMA_20', 'EMA_20', 'RSI_14', 'MACD', 
        'BB_UPPER', 'BB_LOWER', 'ATR_14', 'OBV'
    ]
    
    # Transform with selected indicators only
    result_df = engineer.transform_selective(df, selected)
    
    # Check results
    new_cols = [col for col in result_df.columns if col not in df.columns]
    logger.info(f"Requested indicators: {len(selected)}")
    logger.info(f"Actually calculated: {len(new_cols)}")
    logger.info(f"Calculated indicators: {new_cols}")


def test_performance(df: pd.DataFrame):
    """Test performance of indicator calculations."""
    logger.info("\nTesting performance...")
    
    import time
    
    engineer = FeatureEngineer()
    
    # Time full transformation
    start_time = time.time()
    result_df = engineer.transform(df)
    elapsed = time.time() - start_time
    
    n_indicators = len(result_df.columns) - len(df.columns)
    logger.info(f"Calculated {n_indicators} indicators in {elapsed:.2f} seconds")
    logger.info(f"Average time per indicator: {elapsed/n_indicators:.4f} seconds")
    logger.info(f"Rows processed per second: {len(df)/elapsed:.0f}")


def main():
    """Run all tests."""
    logger.info("Feature Engineering Test Script")
    logger.info("=" * 50)
    
    # Generate sample data
    logger.info("Generating sample data...")
    df = generate_sample_data(1000)
    logger.info(f"Generated {len(df)} rows of OHLCV data")
    logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Run tests
    test_individual_indicators(df)
    test_feature_engineer(df)
    test_selective_transform(df)
    test_performance(df)
    
    logger.info("\nAll tests completed!")


if __name__ == "__main__":
    main()