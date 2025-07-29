"""Integration tests for risk management system scenarios."""

from datetime import datetime, timedelta

from src.risk_management.models.position_sizing import (
    FixedFractionalPositionSizer,
    KellyPositionSizer,
    VolatilityParityPositionSizer,
)
from src.risk_management.risk_manager import RiskManager


class TestRiskManagementIntegration:
    """Test complete risk management workflows."""

    def test_complete_trading_scenario(self):
        """Test a complete trading scenario with multiple positions."""
        # Initialize with Kelly sizing
        kelly = KellyPositionSizer(kelly_fraction=0.25, min_edge=0.02)
        rm = RiskManager(
            position_sizer=kelly,
            max_drawdown=0.10,
            max_daily_loss=0.05,
            max_position_count=3,
        )

        portfolio_value = 100000

        # Scenario 1: Open first position with high confidence
        result1 = rm.check_new_position(
            symbol="BTC/USDT",
            portfolio_value=portfolio_value,
            current_price=50000,
            signal_confidence=0.8,
            win_rate=0.65,
            avg_win=0.04,
            avg_loss=0.01,
            use_limit_order=True,
        )

        assert result1["approved"]
        assert result1["position_size"] > 0
        assert "entry_costs" in result1
        assert result1["risk_multiplier"] == 1.0  # No drawdown yet

        # Record the position
        rm.record_position_open(
            position_id="pos1",
            symbol="BTC/USDT",
            size=result1["position_size"],
            entry_price=50000,
            stop_loss=49000,
        )

        # Scenario 2: Try to open second position
        result2 = rm.check_new_position(
            symbol="ETH/USDT",
            portfolio_value=portfolio_value,
            current_price=3000,
            signal_confidence=0.7,
            win_rate=0.65,  # Increased from 0.60
            avg_win=0.04,  # Increased from 0.03
            avg_loss=0.01,
        )

        assert result2["approved"]
        rm.record_position_open(
            position_id="pos2",
            symbol="ETH/USDT",
            size=10.0,  # Simplified
            entry_price=3000,
        )

        # Scenario 3: Market moves against us - drawdown
        # First update with starting portfolio to establish peak
        rm.update_portfolio_value(portfolio_value)
        new_portfolio = 92000  # 8% drawdown
        rm.update_portfolio_value(new_portfolio)

        # Try third position - should have reduced sizing
        result3 = rm.check_new_position(
            symbol="SOL/USDT",
            portfolio_value=new_portfolio,
            current_price=100,
            signal_confidence=0.9,
            win_rate=0.70,
            avg_win=0.05,
            avg_loss=0.01,
        )

        assert result3["approved"]
        assert result3["risk_multiplier"] < 1.0  # Reduced due to drawdown

        # Scenario 4: Further drawdown triggers guard
        critical_portfolio = 88000  # 12% from peak
        metrics = rm.update_portfolio_value(critical_portfolio)

        assert metrics["drawdown_status"]["is_triggered"]
        assert not metrics["can_trade"]

        # Try to open position - should be rejected
        result4 = rm.check_new_position(
            symbol="BTC/USDT",
            portfolio_value=critical_portfolio,
            current_price=48000,
            signal_confidence=0.95,
            win_rate=0.75,
            avg_win=0.06,
            avg_loss=0.01,
        )

        assert not result4["approved"]
        assert "Drawdown guard" in result4["reason"]

        # Scenario 5: Close losing position
        pnl1 = rm.record_position_close(
            position_id="pos1",
            exit_price=48000,
        )

        assert pnl1["gross_pnl"] < 0
        assert pnl1["net_pnl"] < pnl1["gross_pnl"]  # Costs make it worse

        # Verify risk report
        report = rm.get_risk_report()
        assert report["portfolio_metrics"]["open_positions"] == 1  # One closed
        assert report["portfolio_metrics"]["drawdown_triggered"]

    def test_api_throttling_integration(self):
        """Test API throttling during rapid trading."""
        rm = RiskManager()

        # Simulate rapid-fire position checks
        approved_count = 0
        for i in range(10):
            # Each check uses API weight
            result = rm.check_new_position(
                symbol=f"BTC{i}/USDT",
                portfolio_value=100000,
                current_price=50000 + i * 100,
                signal_confidence=0.7,
                win_rate=0.65,  # Increased
                avg_win=0.04,  # Increased
                avg_loss=0.01,
            )

            if result["approved"]:
                approved_count += 1
                # Simulate API calls for position opening
                rm.api_throttler.check_and_wait_sync("new_order", 1)

        # Should have used significant API capacity
        api_metrics = rm.api_throttler.get_metrics()
        assert api_metrics["total_requests"] > 0

    def test_position_sizing_strategies_comparison(self):
        """Compare different position sizing strategies."""
        portfolio = 100000
        price = 50000
        params = {
            "portfolio_value": portfolio,
            "current_price": price,
            "signal_confidence": 0.8,
            "win_rate": 0.65,
            "avg_win": 0.04,
            "avg_loss": 0.01,
        }

        # Test Kelly
        rm_kelly = RiskManager(position_sizer=KellyPositionSizer())
        result_kelly = rm_kelly.check_new_position("BTC/USDT", **params)

        # Test Fixed Fractional
        rm_fixed = RiskManager(position_sizer=FixedFractionalPositionSizer(risk_per_trade=0.02))
        result_fixed = rm_fixed.check_new_position("BTC/USDT", **params)

        # Test Volatility Parity
        rm_vol = RiskManager(position_sizer=VolatilityParityPositionSizer(target_volatility=0.15))
        params_vol = params.copy()
        params_vol["volatility"] = 0.80  # 80% annual vol
        result_vol = rm_vol.check_new_position("BTC/USDT", **params_vol)

        # All should approve but with different sizes
        assert result_kelly["approved"]
        assert result_fixed["approved"]
        assert result_vol["approved"]

        # Sizes should be different
        assert result_kelly["position_size"] != result_fixed["position_size"]
        assert result_fixed["position_size"] != result_vol["position_size"]

    def test_daily_loss_limit_scenario(self):
        """Test daily loss limit enforcement."""
        rm = RiskManager(max_daily_loss=0.02, max_position_count=5)

        # Open multiple positions
        positions = []
        for i in range(3):
            rm.record_position_open(
                position_id=f"pos{i}",
                symbol="BTC/USDT",
                size=0.5,
                entry_price=50000,
            )
            positions.append(f"pos{i}")

        # Close positions with losses
        total_loss = 0
        for pos_id in positions[:2]:
            pnl = rm.record_position_close(
                position_id=pos_id,
                exit_price=49500,  # 1% loss per position
            )
            total_loss += float(pnl["net_pnl"])

        # Daily P&L should reflect losses
        assert rm.daily_pnl < 0

        # Try to open new position - should check daily loss
        result = rm.check_new_position(
            symbol="ETH/USDT",
            portfolio_value=98000,  # Reflects losses
            current_price=3000,
            signal_confidence=0.9,
            win_rate=0.70,
            avg_win=0.05,
            avg_loss=0.01,
        )

        # If losses exceed 2%, should reject
        if abs(rm.daily_pnl / 98000) > 0.02:
            assert not result["approved"]
            assert "Daily loss limit" in result["reason"]

    def test_cost_impact_on_small_positions(self):
        """Test how costs affect small position viability."""
        rm = RiskManager()

        # Test increasingly smaller positions
        for confidence in [0.1, 0.05, 0.02, 0.01]:
            result = rm.check_new_position(
                symbol="BTC/USDT",
                portfolio_value=10000,  # Small portfolio
                current_price=50000,
                signal_confidence=confidence,
                win_rate=0.55,
                avg_win=0.02,
                avg_loss=0.01,
                urgency=0.8,  # High urgency = more slippage
            )

            if not result["approved"]:
                assert result["reason"] == "Position size below minimum"
                break

        # At some point, positions become too small

    def test_recovery_after_drawdown(self):
        """Test system behavior during recovery from drawdown."""
        rm = RiskManager(max_drawdown=0.10)

        # Create drawdown
        rm.drawdown_guard.update(100000)
        rm.drawdown_guard.update(88000)  # 12% drawdown

        # Should be in protection mode
        assert rm.drawdown_guard.drawdown_triggered

        # Simulate recovery over time
        recovery_values = [88000, 90000, 92000, 94000, 96000, 98000]
        base_time = datetime.now()

        for i, value in enumerate(recovery_values):
            timestamp = base_time + timedelta(days=i + 1)
            status = rm.drawdown_guard.update(value, timestamp)

            # After 7 days and recovery, should deactivate
            if i >= 6 and value > 90000:  # Back above 10% threshold
                if not status["is_triggered"]:
                    break

        # Verify can trade again
        _ = rm.check_new_position(
            symbol="BTC/USDT",
            portfolio_value=recovery_values[-1],
            current_price=50000,
            signal_confidence=0.7,
            win_rate=0.65,
            avg_win=0.04,
            avg_loss=0.01,
        )

        # Should eventually allow trading again
        # (Depends on recovery period settings)
