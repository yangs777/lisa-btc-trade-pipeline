"""Drawdown guard for portfolio protection."""

import numpy as np
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DrawdownGuard:
    """Monitor and protect against excessive drawdowns.

    Tracks portfolio equity curve and triggers risk reduction
    when drawdown exceeds threshold.
    """

    def __init__(
        self,
        max_drawdown: float = 0.10,
        lookback_days: int = 30,
        recovery_days: int = 7,
        scale_positions: bool = True,
    ):
        """Initialize drawdown guard.

        Args:
            max_drawdown: Maximum allowed drawdown (e.g., 0.10 = 10%)
            lookback_days: Days to look back for peak equity
            recovery_days: Days to wait before resuming after trigger
            scale_positions: Whether to scale positions during drawdown
        """
        self.max_drawdown = max_drawdown
        self.lookback_days = lookback_days
        self.recovery_days = recovery_days
        self.scale_positions = scale_positions

        # Track equity history
        self.equity_history: List[float] = []
        self.timestamp_history: List[datetime] = []
        self.peak_equity: float = 0.0
        self.drawdown_triggered: bool = False
        self.trigger_timestamp: Optional[datetime] = None

    def update(self, equity: float, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Update equity and check drawdown status.

        Args:
            equity: Current portfolio equity
            timestamp: Current timestamp (defaults to now)

        Returns:
            Status dictionary with drawdown metrics
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Update history
        self.equity_history.append(equity)
        self.timestamp_history.append(timestamp)

        # Maintain lookback window
        cutoff_time = timestamp - timedelta(days=self.lookback_days)
        while (
            len(self.timestamp_history) > 1
            and self.timestamp_history[0] < cutoff_time
        ):
            self.equity_history.pop(0)
            self.timestamp_history.pop(0)

        # Update peak equity
        self.peak_equity = max(self.equity_history)

        # Calculate current drawdown
        current_drawdown = 0.0
        if self.peak_equity > 0:
            current_drawdown = (self.peak_equity - equity) / self.peak_equity

        # Check drawdown trigger
        if current_drawdown >= self.max_drawdown and not self.drawdown_triggered:
            self.drawdown_triggered = True
            self.trigger_timestamp = timestamp
            logger.warning(
                f"Drawdown guard triggered! Current: {current_drawdown:.2%}, "
                f"Max allowed: {self.max_drawdown:.2%}"
            )

        # Check recovery period
        if self.drawdown_triggered and self.trigger_timestamp:
            recovery_time = self.trigger_timestamp + timedelta(days=self.recovery_days)
            if timestamp >= recovery_time and current_drawdown < self.max_drawdown * 0.5:
                self.drawdown_triggered = False
                self.trigger_timestamp = None
                logger.info("Drawdown guard deactivated - recovery complete")

        return {
            "current_equity": equity,
            "peak_equity": self.peak_equity,
            "current_drawdown": current_drawdown,
            "max_drawdown": self._calculate_max_drawdown(),
            "is_triggered": self.drawdown_triggered,
            "position_scale": self._get_position_scale(current_drawdown),
            "days_in_drawdown": self._days_in_drawdown(timestamp),
        }

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown in history."""
        if len(self.equity_history) < 2:
            return 0.0

        equity_array = np.array(self.equity_history)
        running_max = np.maximum.accumulate(equity_array)
        drawdowns = (running_max - equity_array) / running_max
        return float(np.max(drawdowns))

    def _get_position_scale(self, current_drawdown: float) -> float:
        """Get position scaling factor based on drawdown.

        Args:
            current_drawdown: Current drawdown percentage

        Returns:
            Scaling factor (0-1) for position sizes
        """
        if not self.scale_positions:
            return 1.0 if not self.drawdown_triggered else 0.0

        if self.drawdown_triggered:
            # Scale down aggressively when triggered
            return 0.1

        # Progressive scaling as approaching max drawdown
        if current_drawdown < self.max_drawdown * 0.5:
            return 1.0
        elif current_drawdown < self.max_drawdown * 0.75:
            return 0.75
        elif current_drawdown < self.max_drawdown * 0.9:
            return 0.5
        else:
            return 0.25

    def _days_in_drawdown(self, current_time: datetime) -> int:
        """Calculate days since entering drawdown."""
        if not self.equity_history or self.peak_equity == 0:
            return 0

        # Find when we last hit peak
        for i in range(len(self.equity_history) - 1, -1, -1):
            if self.equity_history[i] >= self.peak_equity * 0.99:  # 1% tolerance
                peak_time = self.timestamp_history[i]
                return (current_time - peak_time).days

        # If never hit peak, use full history
        return (current_time - self.timestamp_history[0]).days

    def get_risk_multiplier(self) -> float:
        """Get current risk multiplier for position sizing.

        Returns:
            Risk multiplier (0-1) based on drawdown status
        """
        if not self.equity_history:
            return 1.0

        current_equity = self.equity_history[-1]
        current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
        return self._get_position_scale(current_drawdown)

    def reset(self) -> None:
        """Reset drawdown guard state."""
        self.equity_history.clear()
        self.timestamp_history.clear()
        self.peak_equity = 0.0
        self.drawdown_triggered = False
        self.trigger_timestamp = None
        logger.info("Drawdown guard reset")