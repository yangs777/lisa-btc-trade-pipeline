"""Position sizing strategies for risk management."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PositionSizer(ABC):
    """Abstract base class for position sizing strategies."""

    def __init__(self, max_position_size: float = 0.95):
        """Initialize position sizer.

        Args:
            max_position_size: Maximum position size as fraction of portfolio
        """
        self.max_position_size = max_position_size

    @abstractmethod
    def calculate_position_size(
        self,
        portfolio_value: float,
        price: float,
        confidence: float,
        **kwargs: Any,
    ) -> float:
        """Calculate position size in BTC.

        Args:
            portfolio_value: Total portfolio value in USD
            price: Current BTC price
            confidence: Model confidence score (0-1)
            **kwargs: Additional strategy-specific parameters

        Returns:
            Position size in BTC
        """
        pass

    def _clip_position_size(self, size: float, portfolio_value: float, price: float) -> float:
        """Clip position size to maximum allowed.

        Args:
            size: Calculated position size in BTC
            portfolio_value: Total portfolio value
            price: Current BTC price

        Returns:
            Clipped position size
        """
        max_btc = (portfolio_value * self.max_position_size) / price
        return min(size, max_btc)


class KellyPositionSizer(PositionSizer):
    """Kelly Criterion position sizing.

    Uses the Kelly formula: f = (p * b - q) / b
    where:
    - f = fraction of capital to bet
    - p = probability of winning
    - q = probability of losing (1 - p)
    - b = odds (win/loss ratio)
    """

    def __init__(
        self,
        max_position_size: float = 0.95,
        kelly_fraction: float = 0.25,
        min_edge: float = 0.02,
    ):
        """Initialize Kelly position sizer.

        Args:
            max_position_size: Maximum position size as fraction
            kelly_fraction: Fraction of Kelly to use (for safety)
            min_edge: Minimum edge required to take position
        """
        super().__init__(max_position_size)
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge

    def calculate_position_size(
        self,
        portfolio_value: float,
        price: float,
        confidence: float,
        **kwargs: Any,
    ) -> float:
        """Calculate position size using Kelly Criterion.

        Args:
            portfolio_value: Total portfolio value
            price: Current BTC price
            confidence: Model confidence (used to scale Kelly)
            **kwargs: Must include win_rate, avg_win, avg_loss

        Returns:
            Position size in BTC
        """
        # Extract required parameters
        win_rate = kwargs.get('win_rate', 0.5)
        avg_win = kwargs.get('avg_win', 0.02)
        avg_loss = kwargs.get('avg_loss', 0.01)
        
        # Calculate Kelly percentage
        if avg_loss <= 0:
            logger.warning("Invalid avg_loss <= 0, returning 0 position")
            return 0.0

        odds = avg_win / avg_loss
        kelly_pct = (win_rate * odds - (1 - win_rate)) / odds

        # Check minimum edge
        edge = win_rate * avg_win - (1 - win_rate) * avg_loss
        if edge < self.min_edge:
            logger.info(f"Edge {edge:.4f} below minimum {self.min_edge}, no position")
            return 0.0

        # Apply Kelly fraction and confidence scaling
        position_fraction = kelly_pct * self.kelly_fraction * confidence

        # Ensure non-negative
        position_fraction = max(0, position_fraction)

        # Calculate BTC amount
        position_value = portfolio_value * position_fraction
        position_btc = position_value / price

        return self._clip_position_size(position_btc, portfolio_value, price)


class FixedFractionalPositionSizer(PositionSizer):
    """Fixed fractional position sizing.

    Risks a fixed percentage of portfolio on each trade.
    """

    def __init__(
        self,
        max_position_size: float = 0.95,
        risk_per_trade: float = 0.02,
        stop_loss_pct: float = 0.02,
    ):
        """Initialize fixed fractional sizer.

        Args:
            max_position_size: Maximum position size
            risk_per_trade: Fraction of portfolio to risk per trade
            stop_loss_pct: Stop loss percentage for position
        """
        super().__init__(max_position_size)
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct

    def calculate_position_size(
        self,
        portfolio_value: float,
        price: float,
        confidence: float,
        **kwargs: Any,
    ) -> float:
        """Calculate fixed fractional position size.

        Args:
            portfolio_value: Total portfolio value
            price: Current BTC price
            confidence: Model confidence (scales risk)
            **kwargs: Optional stop_loss_pct parameter

        Returns:
            Position size in BTC
        """
        # Use provided stop loss or default
        stop_loss_pct = kwargs.get('stop_loss_pct')
        stop_pct = stop_loss_pct or self.stop_loss_pct

        # Calculate position size based on risk
        risk_amount = portfolio_value * self.risk_per_trade * confidence
        position_value = risk_amount / stop_pct
        position_btc = position_value / price

        return self._clip_position_size(position_btc, portfolio_value, price)


class VolatilityParityPositionSizer(PositionSizer):
    """Volatility parity position sizing.

    Scales position size inversely with volatility to maintain
    constant risk across different market conditions.
    """

    def __init__(
        self,
        max_position_size: float = 0.95,
        target_volatility: float = 0.15,
        lookback_days: int = 30,
    ):
        """Initialize volatility parity sizer.

        Args:
            max_position_size: Maximum position size
            target_volatility: Target annualized volatility
            lookback_days: Days for volatility calculation
        """
        super().__init__(max_position_size)
        self.target_volatility = target_volatility
        self.lookback_days = lookback_days

    def calculate_position_size(
        self,
        portfolio_value: float,
        price: float,
        confidence: float,
        **kwargs: Any,
    ) -> float:
        """Calculate volatility-adjusted position size.

        Args:
            portfolio_value: Total portfolio value
            price: Current BTC price
            confidence: Model confidence
            **kwargs: Must include 'returns' parameter (np.ndarray)

        Returns:
            Position size in BTC
        """
        # Extract returns from kwargs
        returns = kwargs.get('returns', np.array([]))
        
        # Calculate realized volatility
        if len(returns) < self.lookback_days:
            logger.warning("Insufficient data for volatility calculation")
            # Use a conservative default
            realized_vol = 0.5
        else:
            # Use recent returns
            recent_returns = returns[-self.lookback_days :]
            realized_vol = np.std(recent_returns) * np.sqrt(365)

        # Avoid division by zero
        if realized_vol < 0.01:
            realized_vol = 0.01

        # Calculate position size
        vol_scalar = self.target_volatility / realized_vol
        position_fraction = vol_scalar * confidence

        # Apply position limits
        position_fraction = min(position_fraction, self.max_position_size)
        position_value = portfolio_value * position_fraction
        position_btc = position_value / price

        return self._clip_position_size(position_btc, portfolio_value, price)