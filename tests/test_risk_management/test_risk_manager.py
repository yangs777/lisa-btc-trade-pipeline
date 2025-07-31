"""Tests for integrated risk manager."""

from datetime import datetime, timedelta

from src.risk_management.models.position_sizing import FixedFractionalPositionSizer
from src.risk_management.risk_manager import RiskManager


class TestRiskManager:
    """Test integrated risk management."""

    def test_risk_manager_initialization(self):
        """Test risk manager initialization."""
        rm = RiskManager(
            max_drawdown=0.15,
            max_daily_loss=0.03,
            max_position_count=5,
        )

        assert rm.drawdown_guard.max_drawdown == 0.15
        assert rm.max_daily_loss == 0.03
        assert rm.max_position_count == 5
        assert len(rm.open_positions) == 0

    def test_position_approval_basic(self):
        """Test basic position approval."""
        rm = RiskManager()

        result = rm.check_new_position(
            symbol="BTC/USDT",
            portfolio_value=100000,
            current_price=50000,
            signal_confidence=0.8,
            win_rate=0.65,
            avg_win=0.04,  # Higher win to meet min edge
            avg_loss=0.01,
        )

        # Edge = 0.65 * 0.04 - 0.35 * 0.01 = 0.026 - 0.0035 = 0.0225 > 0.02
        assert result["approved"]
        assert result["position_size"] > 0
        assert "entry_costs" in result
        assert "risk_multiplier" in result

    def test_position_rejection_max_positions(self):
        """Test rejection due to max positions."""
        rm = RiskManager(max_position_count=2)

        # Open 2 positions
        rm.record_position_open("pos1", "BTC/USDT", 1.0, 50000)
        rm.record_position_open("pos2", "ETH/USDT", 10.0, 3000)

        # Try to open 3rd position
        result = rm.check_new_position(
            symbol="BTC/USDT",
            portfolio_value=100000,
            current_price=50000,
            signal_confidence=0.8,
            win_rate=0.65,
            avg_win=0.04,
            avg_loss=0.01,
        )

        assert not result["approved"]
        assert result["reason"] == "Maximum positions reached"

    def test_position_rejection_drawdown(self):
        """Test rejection due to drawdown."""
        rm = RiskManager(max_drawdown=0.10)

        # Simulate drawdown
        rm.drawdown_guard.update(100000)
        rm.drawdown_guard.update(88000)  # 12% drawdown

        result = rm.check_new_position(
            symbol="BTC/USDT",
            portfolio_value=88000,
            current_price=50000,
            signal_confidence=0.8,
            win_rate=0.65,
            avg_win=0.04,
            avg_loss=0.01,
        )

        assert not result["approved"]
        assert result["reason"] == "Drawdown guard triggered"

    def test_position_rejection_daily_loss(self):
        """Test rejection due to daily loss limit."""
        rm = RiskManager(max_daily_loss=0.02)

        # Simulate daily loss
        rm.daily_pnl = -2500  # 2.5% loss on 100k

        result = rm.check_new_position(
            symbol="BTC/USDT",
            portfolio_value=97500,
            current_price=50000,
            signal_confidence=0.8,
            win_rate=0.65,
            avg_win=0.04,
            avg_loss=0.01,
        )

        assert not result["approved"]
        assert result["reason"] == "Daily loss limit exceeded"

    def test_position_size_too_small(self):
        """Test rejection due to position size too small."""
        rm = RiskManager()

        # Very low confidence should result in tiny position
        result = rm.check_new_position(
            symbol="BTC/USDT",
            portfolio_value=100000,
            current_price=50000,
            signal_confidence=0.01,  # Very low
            win_rate=0.51,
            avg_win=0.01,
            avg_loss=0.01,
        )

        assert not result["approved"]
        assert result["reason"] == "Position size below minimum"

    def test_position_recording(self):
        """Test position recording."""
        rm = RiskManager()

        # Record position
        rm.record_position_open(
            position_id="test123",
            symbol="BTC/USDT",
            size=2.0,
            entry_price=50000,
            position_type="long",
            stop_loss=48000,
            take_profit=52000,
        )

        assert "test123" in rm.open_positions
        pos = rm.open_positions["test123"]
        assert pos["size"] == 2.0
        assert pos["entry_price"] == 50000
        assert pos["stop_loss"] == 48000

    def test_position_close_profit(self):
        """Test closing profitable position."""
        rm = RiskManager()

        # Open position
        rm.record_position_open(
            position_id="test123",
            symbol="BTC/USDT",
            size=2.0,
            entry_price=50000,
            position_type="long",
        )

        # Close with profit
        result = rm.record_position_close(
            position_id="test123",
            exit_price=52000,
        )

        # Gross profit: (52000 - 50000) * 2 = $4000
        assert result["gross_pnl"] == 4000
        assert result["net_pnl"] < result["gross_pnl"]  # After costs
        assert result["return_pct"] > 0

        # Position should be removed
        assert "test123" not in rm.open_positions

    def test_position_close_loss(self):
        """Test closing losing position."""
        rm = RiskManager()

        # Open short position
        rm.record_position_open(
            position_id="test123",
            symbol="BTC/USDT",
            size=1.0,
            entry_price=50000,
            position_type="short",
        )

        # Close with loss (price went up)
        result = rm.record_position_close(
            position_id="test123",
            exit_price=51000,
        )

        # Gross loss: (50000 - 51000) * 1 = -$1000
        assert result["gross_pnl"] == -1000
        assert result["net_pnl"] < result["gross_pnl"]  # More negative after costs

    def test_partial_position_close(self):
        """Test partial position closing."""
        rm = RiskManager()

        # Open position
        rm.record_position_open(
            position_id="test123",
            symbol="BTC/USDT",
            size=2.0,
            entry_price=50000,
        )

        # Close half
        result = rm.record_position_close(
            position_id="test123",
            exit_price=51000,
            exit_size=1.0,
        )

        assert result["gross_pnl"] == 1000  # (51000-50000) * 1

        # Position should still exist with reduced size
        assert "test123" in rm.open_positions
        assert rm.open_positions["test123"]["size"] == 1.0

    def test_portfolio_update(self):
        """Test portfolio value update."""
        rm = RiskManager()

        # Add a position
        rm.record_position_open("pos1", "BTC/USDT", 1.0, 50000)

        # Update portfolio
        metrics = rm.update_portfolio_value(110000)

        assert metrics["portfolio_value"] == 110000
        assert metrics["open_positions"] == 1
        assert metrics["open_exposure"] == 50000
        assert abs(metrics["exposure_pct"] - 45.45) < 0.1

    def test_daily_reset(self):
        """Test daily P&L reset."""
        rm = RiskManager()

        # Set some P&L
        rm.daily_pnl = -1000
        rm.last_reset_date = datetime.now().date() - timedelta(days=1)

        # Trigger reset by checking position
        rm.check_new_position(
            symbol="BTC/USDT",
            portfolio_value=100000,
            current_price=50000,
            signal_confidence=0.5,
            win_rate=0.65,
            avg_win=0.04,
            avg_loss=0.01,
        )

        assert rm.daily_pnl == 0
        assert rm.last_reset_date == datetime.now().date()

    def test_risk_report_generation(self):
        """Test risk report generation."""
        rm = RiskManager()

        # Add some activity
        rm.record_position_open("pos1", "BTC/USDT", 1.0, 50000)
        rm.daily_pnl = 500

        report = rm.get_risk_report()

        assert "timestamp" in report
        assert "portfolio_metrics" in report
        assert "risk_limits" in report
        assert "api_metrics" in report
        assert "cost_estimates" in report

        assert report["portfolio_metrics"]["open_positions"] == 1
        assert report["portfolio_metrics"]["daily_pnl"] == 500

    def test_custom_position_sizer(self):
        """Test using custom position sizer."""
        sizer = FixedFractionalPositionSizer(
            risk_per_trade=0.01,
            stop_loss_pct=0.02
        )

        rm = RiskManager(position_sizer=sizer)

        result = rm.check_new_position(
            symbol="BTC/USDT",
            portfolio_value=100000,
            current_price=50000,
            signal_confidence=1.0,
        )

        # Should use fixed fractional sizing
        # Risk $1000 with 2% stop = $50k position = 1 BTC
        assert abs(result["position_size"] - 1.0) < 0.1

    def test_api_capacity_check(self):
        """Test API capacity checking."""
        rm = RiskManager()

        # Exhaust API capacity
        rm.api_throttler.limits["1m"].current_weight = 1199

        result = rm.check_new_position(
            symbol="BTC/USDT",
            portfolio_value=100000,
            current_price=50000,
            signal_confidence=0.8,
            win_rate=0.65,
            avg_win=0.04,
            avg_loss=0.01,
        )

        # Should reject due to API limits
        assert not result["approved"]
        assert "API capacity" in result["reason"]
