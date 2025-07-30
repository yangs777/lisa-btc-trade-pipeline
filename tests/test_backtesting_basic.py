"""Tests for backtesting module."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_backtester():
    """Test backtesting functionality."""
    from src.backtesting import Backtester
    
    # Create sample data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='h'),
        'open': np.random.uniform(100, 110, 100),
        'high': np.random.uniform(101, 111, 100),
        'low': np.random.uniform(99, 109, 100),
        'close': np.random.uniform(100, 110, 100),
        'volume': np.random.uniform(1000, 2000, 100)
    })
    
    # Create simple strategy
    class SimpleStrategy:
        def __init__(self):
            self.position = 0
            
        def generate_signals(self, data):
            signals = []
            for i in range(len(data)):
                if i % 10 == 0:
                    signals.append(1)  # Buy
                elif i % 10 == 5:
                    signals.append(-1)  # Sell
                else:
                    signals.append(0)  # Hold
            return signals
    
    strategy = SimpleStrategy()
    backtester = Backtester(initial_capital=10000)
    
    # Run backtest
    results = backtester.run(data, strategy)
    
    assert results is not None
    assert hasattr(results, 'total_return')
    assert hasattr(results, 'sharpe_ratio')
    assert hasattr(results, 'max_drawdown')
    assert results.total_trades >= 0
