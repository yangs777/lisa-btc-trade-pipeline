"""Tests for position sizing strategies."""

import numpy as np

from src.risk_management.models.position_sizing import (
    FixedFractionalPositionSizer,
    KellyPositionSizer,
    VolatilityParityPositionSizer,
)


class TestKellyPositionSizer:
    """Test Kelly Criterion position sizing."""

    def test_kelly_basic_calculation(self):
        """Test basic Kelly calculation."""
        kelly = KellyPositionSizer(
            kelly_fraction=0.25, min_edge=0.005  # Lower threshold to allow this test case
        )

        # Favorable edge: 60% win rate, 2:1 reward/risk
        position_size = kelly.calculate_position_size(
            portfolio_value=100000,
            price=50000,
            confidence=1.0,
            win_rate=0.6,
            avg_win=0.02,  # 2% average win
            avg_loss=0.01,  # 1% average loss
        )

        # Edge = 0.6 * 0.02 - 0.4 * 0.01 = 0.012 - 0.004 = 0.008
        # Kelly % = (0.6 * 2 - 0.4) / 2 = 0.8 / 2 = 0.4
        # With 0.25 fraction = 0.4 * 0.25 = 0.1 (10%)
        expected_btc = (100000 * 0.1) / 50000
        assert abs(position_size - expected_btc) < 0.01

    def test_kelly_with_no_edge(self):
        """Test Kelly with no edge."""
        kelly = KellyPositionSizer(min_edge=0.02)

        # No edge: 50% win rate, 1:1 reward/risk
        position_size = kelly.calculate_position_size(
            portfolio_value=100000,
            price=50000,
            confidence=1.0,
            win_rate=0.5,
            avg_win=0.01,
            avg_loss=0.01,
        )

        # Edge = 0.5 * 0.01 - 0.5 * 0.01 = 0
        # Below minimum edge, should return 0
        assert position_size == 0.0

    def test_kelly_confidence_scaling(self):
        """Test confidence scaling."""
        kelly = KellyPositionSizer(min_edge=0.005)  # Lower threshold

        # High confidence
        size_high = kelly.calculate_position_size(
            portfolio_value=100000,
            price=50000,
            confidence=1.0,
            win_rate=0.6,
            avg_win=0.02,
            avg_loss=0.01,
        )

        # Low confidence
        size_low = kelly.calculate_position_size(
            portfolio_value=100000,
            price=50000,
            confidence=0.5,
            win_rate=0.6,
            avg_win=0.02,
            avg_loss=0.01,
        )

        # Low confidence should result in smaller position
        assert size_low < size_high
        assert abs(size_low - size_high * 0.5) < 0.01

    def test_kelly_max_position_limit(self):
        """Test maximum position size constraint."""
        kelly = KellyPositionSizer(
            max_position_size=0.5, min_edge=0.01  # Adjust for this test's edge
        )

        # Very favorable edge that would suggest large position
        position_size = kelly.calculate_position_size(
            portfolio_value=100000,
            price=50000,
            confidence=1.0,
            win_rate=0.9,
            avg_win=0.05,
            avg_loss=0.01,
        )

        # Edge = 0.9 * 0.05 - 0.1 * 0.01 = 0.045 - 0.001 = 0.044 (above threshold)
        # Should be capped at 50% of portfolio
        max_btc = (100000 * 0.5) / 50000
        assert position_size <= max_btc


class TestFixedFractionalPositionSizer:
    """Test fixed fractional position sizing."""

    def test_fixed_fractional_basic(self):
        """Test basic fixed fractional calculation."""
        ff = FixedFractionalPositionSizer(risk_per_trade=0.02, stop_loss_pct=0.05)

        position_size = ff.calculate_position_size(
            portfolio_value=100000,
            price=50000,
            confidence=1.0,
        )

        # Risk $2000 with 5% stop = $40,000 position = 0.8 BTC
        expected_btc = (100000 * 0.02) / 0.05 / 50000
        assert abs(position_size - expected_btc) < 0.01

    def test_fixed_fractional_with_custom_stop(self):
        """Test with custom stop loss."""
        ff = FixedFractionalPositionSizer(risk_per_trade=0.01)

        position_size = ff.calculate_position_size(
            portfolio_value=100000,
            price=50000,
            confidence=1.0,
            stop_loss_pct=0.03,  # 3% stop
        )

        # Risk $1000 with 3% stop
        expected_btc = (100000 * 0.01) / 0.03 / 50000
        assert abs(position_size - expected_btc) < 0.01

    def test_fixed_fractional_confidence_scaling(self):
        """Test confidence scaling."""
        ff = FixedFractionalPositionSizer(
            risk_per_trade=0.01, stop_loss_pct=0.02  # Lower risk to avoid clipping
        )

        size_high = ff.calculate_position_size(
            portfolio_value=100000,
            price=50000,
            confidence=1.0,
        )

        size_low = ff.calculate_position_size(
            portfolio_value=100000,
            price=50000,
            confidence=0.5,
        )

        assert size_low == size_high * 0.5


class TestVolatilityParityPositionSizer:
    """Test volatility parity position sizing."""

    def test_volatility_parity_basic(self):
        """Test basic volatility parity calculation."""
        vp = VolatilityParityPositionSizer(target_volatility=0.15, lookback_days=30)

        # Create returns with 20% annualized vol
        daily_vol = 0.20 / np.sqrt(365)
        returns = np.random.normal(0, daily_vol, 100)

        position_size = vp.calculate_position_size(
            portfolio_value=100000,
            price=50000,
            confidence=1.0,
            returns=returns,
        )

        # With 20% realized vol and 15% target, scale = 0.75
        # Position should be around 75% of portfolio
        expected_fraction = 0.75
        (100000 * expected_fraction) / 50000

        # Allow for randomness in returns
        assert 0.5 < position_size < 2.5

    def test_volatility_parity_high_vol(self):
        """Test with high volatility."""
        vp = VolatilityParityPositionSizer(target_volatility=0.10)

        # High volatility returns (50% annualized)
        daily_vol = 0.50 / np.sqrt(365)
        returns = np.random.normal(0, daily_vol, 100)

        position_size = vp.calculate_position_size(
            portfolio_value=100000,
            price=50000,
            confidence=1.0,
            returns=returns,
        )

        # With 50% vol and 10% target, scale = 0.2
        # Should result in small position
        assert position_size < 1.0  # Less than 1 BTC

    def test_volatility_parity_insufficient_data(self):
        """Test with insufficient data."""
        vp = VolatilityParityPositionSizer(lookback_days=30)

        # Only 10 days of returns
        returns = np.random.normal(0, 0.01, 10)

        position_size = vp.calculate_position_size(
            portfolio_value=100000,
            price=50000,
            confidence=1.0,
            returns=returns,
        )

        # Should use conservative default volatility
        # Default 50% vol with 15% target = 0.3 scale
        assert position_size > 0
        assert position_size < 1.0  # Conservative

    def test_position_sizer_max_limit(self):
        """Test all sizers respect max position limit."""
        sizers = [
            KellyPositionSizer(max_position_size=0.2, min_edge=0.01),
            FixedFractionalPositionSizer(max_position_size=0.2),
            VolatilityParityPositionSizer(max_position_size=0.2),
        ]

        for sizer in sizers:
            # Test with parameters that would suggest large position
            if isinstance(sizer, KellyPositionSizer):
                size = sizer.calculate_position_size(
                    portfolio_value=100000,
                    price=50000,
                    confidence=1.0,
                    win_rate=0.9,
                    avg_win=0.1,
                    avg_loss=0.01,
                )
            elif isinstance(sizer, FixedFractionalPositionSizer):
                size = sizer.calculate_position_size(
                    portfolio_value=100000,
                    price=50000,
                    confidence=1.0,
                    stop_loss_pct=0.001,  # Very tight stop
                )
            else:  # VolatilityParityPositionSizer
                # Very low vol returns
                returns = np.ones(100) * 0.001
                size = sizer.calculate_position_size(
                    portfolio_value=100000,
                    price=50000,
                    confidence=1.0,
                    returns=returns,
                )

            # Check max position respected
            max_btc = (100000 * 0.2) / 50000
            assert size <= max_btc + 0.001  # Small tolerance for float precision
