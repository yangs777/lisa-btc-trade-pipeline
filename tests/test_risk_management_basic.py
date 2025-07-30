"""Tests for risk_management module."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_risk_manager_initialization():
    """Test RiskManager initialization."""
    from src.risk_management import RiskManager
    
    config = {
        "max_position_size": 0.1,
        "max_drawdown": 0.2,
        "risk_per_trade": 0.02
    }
    
    rm = RiskManager(config)
    assert rm is not None
    assert rm.max_position_size == 0.1
    assert rm.max_drawdown == 0.2


def test_position_sizing():
    """Test position sizing calculation."""
    from src.risk_management import RiskManager
    
    config = {
        "max_position_size": 0.1,
        "max_drawdown": 0.2,
        "risk_per_trade": 0.02
    }
    
    rm = RiskManager(config)
    
    # Test position size calculation
    position_size = rm.calculate_position_size(
        capital=10000,
        price=50000,
        stop_loss_pct=0.02
    )
    
    assert position_size > 0
    assert position_size <= 10000 * 0.1


def test_risk_validation():
    """Test risk validation."""
    from src.risk_management import RiskManager
    
    config = {
        "max_position_size": 0.1,
        "max_drawdown": 0.2,
        "risk_per_trade": 0.02
    }
    
    rm = RiskManager(config)
    
    # Test valid trade
    is_valid = rm.validate_trade(
        position_size=1000,
        capital=10000,
        current_drawdown=0.1
    )
    assert is_valid is True
    
    # Test invalid trade (exceeds max position)
    is_valid = rm.validate_trade(
        position_size=2000,
        capital=10000,
        current_drawdown=0.1
    )
    assert is_valid is False
