"""Tests for drawdown guard."""

import pytest
from datetime import datetime, timedelta

from src.risk_management.models.drawdown_guard import DrawdownGuard


class TestDrawdownGuard:
    """Test drawdown guard functionality."""

    def test_drawdown_calculation(self):
        """Test basic drawdown calculation."""
        guard = DrawdownGuard(max_drawdown=0.10)
        
        # Simulate equity curve
        equities = [100000, 105000, 110000, 108000, 105000, 103000]
        
        for equity in equities:
            status = guard.update(equity)
        
        # Peak was 110000, current is 103000
        # Drawdown = (110000 - 103000) / 110000 = 0.0636
        assert abs(status["current_drawdown"] - 0.0636) < 0.001
        assert status["peak_equity"] == 110000
        assert not status["is_triggered"]  # Below 10% threshold

    def test_drawdown_trigger(self):
        """Test drawdown trigger activation."""
        guard = DrawdownGuard(max_drawdown=0.10)
        
        # Start at 100k
        guard.update(100000)
        
        # Rise to 120k
        guard.update(120000)
        
        # Drop to 106k (11.67% drawdown)
        status = guard.update(106000)
        
        assert status["is_triggered"]
        assert status["current_drawdown"] > 0.10

    def test_position_scaling(self):
        """Test position scaling based on drawdown."""
        guard = DrawdownGuard(max_drawdown=0.10, scale_positions=True)
        
        # No drawdown
        guard.update(100000)
        assert guard.get_risk_multiplier() == 1.0
        
        # 4% drawdown (< 50% of max)
        guard.update(110000)
        guard.update(105600)
        assert guard.get_risk_multiplier() == 1.0
        
        # 6% drawdown (60% of max)
        guard.update(103400)
        multiplier = guard.get_risk_multiplier()
        assert 0.7 < multiplier < 0.8  # Should be 0.75
        
        # 9.5% drawdown (95% of max)
        # Peak is 110000, so 9.5% drawdown = 110000 * 0.095 = 10450
        # Current equity = 110000 - 10450 = 99550
        guard.update(99550)
        multiplier = guard.get_risk_multiplier()
        assert 0.2 < multiplier < 0.3  # Should be 0.25

    def test_recovery_period(self):
        """Test recovery period after trigger."""
        guard = DrawdownGuard(
            max_drawdown=0.10,
            recovery_days=7
        )
        
        base_time = datetime.now()
        
        # Trigger drawdown
        guard.update(100000, base_time)
        guard.update(120000, base_time)
        guard.update(106000, base_time)  # 11.67% drawdown
        
        assert guard.drawdown_triggered
        
        # After 5 days, still triggered
        status = guard.update(110000, base_time + timedelta(days=5))
        assert status["is_triggered"]
        
        # After 8 days with recovery, should deactivate
        status = guard.update(115000, base_time + timedelta(days=8))
        assert not status["is_triggered"]

    def test_lookback_window(self):
        """Test lookback window for peak calculation."""
        guard = DrawdownGuard(
            max_drawdown=0.10,
            lookback_days=30
        )
        
        base_time = datetime.now()
        
        # Add old peak
        guard.update(150000, base_time - timedelta(days=40))
        
        # Add recent data
        for i in range(20):
            guard.update(100000 + i * 1000, base_time - timedelta(days=20-i))
        
        # Old peak should be excluded
        assert guard.peak_equity < 150000
        assert guard.peak_equity == 119000  # Most recent value

    def test_max_drawdown_calculation(self):
        """Test maximum drawdown tracking."""
        guard = DrawdownGuard()
        
        # Simulate volatile equity curve
        equities = [
            100000, 110000, 105000, 115000, 100000,  # 13% drawdown
            90000,   # 21.7% drawdown from 115k
            95000, 100000, 105000
        ]
        
        for equity in equities:
            guard.update(equity)
        
        max_dd = guard._calculate_max_drawdown()
        assert abs(max_dd - 0.217) < 0.001

    def test_days_in_drawdown(self):
        """Test counting days in drawdown."""
        guard = DrawdownGuard()
        base_time = datetime.now()
        
        # Start at peak
        guard.update(100000, base_time)
        
        # Stay at peak for 5 days
        for i in range(1, 6):
            guard.update(100000, base_time + timedelta(days=i))
        
        # Enter drawdown
        status = guard.update(95000, base_time + timedelta(days=10))
        
        # Should be 5 days since last peak
        assert status["days_in_drawdown"] == 5

    def test_no_scaling_mode(self):
        """Test binary mode (no scaling)."""
        guard = DrawdownGuard(
            max_drawdown=0.10,
            scale_positions=False
        )
        
        # Before trigger
        guard.update(100000)
        guard.update(110000)
        guard.update(105000)  # 4.5% drawdown
        assert guard.get_risk_multiplier() == 1.0
        
        # After trigger
        guard.update(98000)  # 10.9% drawdown
        assert guard.get_risk_multiplier() == 0.0

    def test_reset_functionality(self):
        """Test guard reset."""
        guard = DrawdownGuard()
        
        # Add some history
        guard.update(100000)
        guard.update(110000)
        guard.update(95000)
        
        # Trigger drawdown
        guard.drawdown_triggered = True
        
        # Reset
        guard.reset()
        
        assert len(guard.equity_history) == 0
        assert guard.peak_equity == 0.0
        assert not guard.drawdown_triggered

    def test_empty_history_handling(self):
        """Test handling of empty history."""
        guard = DrawdownGuard()
        
        # Should handle empty history gracefully
        assert guard.get_risk_multiplier() == 1.0
        assert guard._calculate_max_drawdown() == 0.0
        assert guard._days_in_drawdown(datetime.now()) == 0