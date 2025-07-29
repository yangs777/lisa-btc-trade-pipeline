"""Integrated risk management system."""

import logging
from datetime import datetime
from typing import Any

import numpy as np

from .models.api_throttler import BinanceAPIThrottler
from .models.cost_model import CostModel
from .models.drawdown_guard import DrawdownGuard
from .models.position_sizing import (
    KellyPositionSizer,
    PositionSizer,
)

logger = logging.getLogger(__name__)


class RiskManager:
    """Integrated risk management system.

    Combines position sizing, drawdown protection, cost modeling,
    and API rate limiting for comprehensive risk management.
    """

    def __init__(
        self,
        position_sizer: PositionSizer | None = None,
        max_drawdown: float = 0.10,
        max_daily_loss: float = 0.05,
        max_position_count: int = 3,
        correlation_limit: float = 0.7,
    ):
        """Initialize risk manager.

        Args:
            position_sizer: Position sizing strategy
            max_drawdown: Maximum portfolio drawdown
            max_daily_loss: Maximum daily loss limit
            max_position_count: Maximum concurrent positions
            correlation_limit: Maximum correlation between positions
        """
        # Use default Kelly sizer if none provided
        self.position_sizer = position_sizer or KellyPositionSizer()

        # Initialize components
        self.drawdown_guard = DrawdownGuard(max_drawdown=max_drawdown)
        self.cost_model = CostModel()
        self.api_throttler = BinanceAPIThrottler()

        # Risk limits
        self.max_daily_loss = max_daily_loss
        self.max_position_count = max_position_count
        self.correlation_limit = correlation_limit

        # Track daily P&L
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()

        # Track open positions
        self.open_positions: dict[str, dict[str, Any]] = {}

        # Track historical returns for volatility sizing
        self.historical_returns: dict[str, list[float]] = {}

    def check_new_position(
        self,
        symbol: str,
        portfolio_value: float,
        current_price: float,
        signal_confidence: float,
        position_type: str = "long",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Check if new position can be opened.

        Args:
            symbol: Trading symbol
            portfolio_value: Current portfolio value
            current_price: Current asset price
            signal_confidence: Model confidence (0-1)
            position_type: 'long' or 'short'
            **kwargs: Additional parameters for position sizing

        Returns:
            Dictionary with approval status and details
        """
        # Reset daily P&L if new day
        self._check_daily_reset()

        # Check basic constraints
        constraints = self._check_constraints(portfolio_value)
        if not constraints["can_trade"]:
            return {
                "approved": False,
                "reason": constraints["reason"],
                "position_size": 0.0,
            }

        # Get risk multiplier from drawdown guard
        risk_multiplier = self.drawdown_guard.get_risk_multiplier()

        # Adjust confidence based on risk state
        adjusted_confidence = signal_confidence * risk_multiplier

        # Add returns for volatility sizing if needed
        from .models.position_sizing import VolatilityParityPositionSizer

        if isinstance(self.position_sizer, VolatilityParityPositionSizer):
            # Use dummy returns if not available
            kwargs["returns"] = self.historical_returns.get(symbol, [0.01] * 30)

        # Calculate position size
        position_size = self.position_sizer.calculate_position_size(
            portfolio_value=portfolio_value,
            price=current_price,
            confidence=adjusted_confidence,
            **kwargs,
        )

        # Check if position is too small
        min_position_value = portfolio_value * 0.01  # 1% minimum
        if position_size * current_price < min_position_value:
            return {
                "approved": False,
                "reason": "Position size below minimum",
                "position_size": 0.0,
            }

        # Calculate estimated costs
        notional_value = position_size * current_price
        entry_costs = self.cost_model.calculate_entry_costs(
            notional_value=notional_value,
            is_maker=kwargs.get("use_limit_order", False),
            urgency=kwargs.get("urgency", 0.5),
        )

        # Check API capacity
        remaining_capacity = self.api_throttler.get_remaining_capacity("new_order")
        if remaining_capacity < 2:  # Need capacity for entry and potential quick exit
            return {
                "approved": False,
                "reason": "Insufficient API capacity",
                "position_size": 0.0,
            }

        return {
            "approved": True,
            "position_size": position_size,
            "notional_value": notional_value,
            "entry_costs": entry_costs,
            "risk_multiplier": risk_multiplier,
            "adjusted_confidence": adjusted_confidence,
            "max_loss": self._calculate_max_loss(notional_value),
        }

    def record_position_open(
        self,
        position_id: str,
        symbol: str,
        size: float,
        entry_price: float,
        position_type: str = "long",
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> None:
        """Record a new position opening.

        Args:
            position_id: Unique position identifier
            symbol: Trading symbol
            size: Position size in base currency
            entry_price: Entry price
            position_type: 'long' or 'short'
            stop_loss: Stop loss price
            take_profit: Take profit price
        """
        self.open_positions[position_id] = {
            "symbol": symbol,
            "size": size,
            "entry_price": entry_price,
            "position_type": position_type,
            "entry_time": datetime.now(),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "realized_pnl": 0.0,
        }

        logger.info(
            f"Position opened: {position_id} - {position_type} {size} {symbol} @ {entry_price}"
        )

    def record_position_close(
        self,
        position_id: str,
        exit_price: float,
        exit_size: float | None = None,
    ) -> dict[str, float]:
        """Record position closing.

        Args:
            position_id: Position identifier
            exit_price: Exit price
            exit_size: Size to close (None = full position)

        Returns:
            Dictionary with P&L details
        """
        if position_id not in self.open_positions:
            logger.error(f"Unknown position: {position_id}")
            return {"realized_pnl": 0.0, "costs": 0.0}

        position = self.open_positions[position_id]
        size_to_close = exit_size or position["size"]

        # Calculate P&L
        if position["position_type"] == "long":
            gross_pnl = (exit_price - position["entry_price"]) * size_to_close
        else:
            gross_pnl = (position["entry_price"] - exit_price) * size_to_close

        # Calculate holding time
        holding_time = datetime.now() - position["entry_time"]
        holding_hours = holding_time.total_seconds() / 3600

        # Calculate costs
        notional_value = size_to_close * position["entry_price"]
        costs = self.cost_model.calculate_round_trip_costs(
            notional_value=notional_value,
            holding_hours=holding_hours,
        )

        # Net P&L
        net_pnl = gross_pnl - costs["total_cost"]

        # Update position
        if size_to_close >= position["size"]:
            # Full close
            del self.open_positions[position_id]
        else:
            # Partial close
            position["size"] -= size_to_close
            position["realized_pnl"] += net_pnl

        # Update daily P&L
        self.daily_pnl += net_pnl

        logger.info(
            f"Position closed: {position_id} - P&L: ${net_pnl:.2f} "
            f"(gross: ${gross_pnl:.2f}, costs: ${costs['total_cost']:.2f})"
        )

        return {
            "gross_pnl": gross_pnl,
            "costs": costs["total_cost"],
            "net_pnl": net_pnl,
            "return_pct": (net_pnl / notional_value) * 100,
        }

    def update_portfolio_value(self, portfolio_value: float) -> dict[str, Any]:
        """Update portfolio value and risk metrics.

        Args:
            portfolio_value: Current portfolio value

        Returns:
            Risk metrics and status
        """
        # Update drawdown guard
        dd_status = self.drawdown_guard.update(portfolio_value)

        # Calculate open position metrics
        open_exposure = sum(
            pos["size"] * pos["entry_price"] for pos in self.open_positions.values()
        )

        # Calculate portfolio metrics
        metrics = {
            "portfolio_value": portfolio_value,
            "open_positions": len(self.open_positions),
            "open_exposure": open_exposure,
            "exposure_pct": (open_exposure / portfolio_value * 100) if portfolio_value > 0 else 0,
            "daily_pnl": self.daily_pnl,
            "daily_return": (self.daily_pnl / portfolio_value * 100) if portfolio_value > 0 else 0,
            "drawdown_status": dd_status,
            "can_trade": self._check_constraints(portfolio_value)["can_trade"],
        }

        return metrics

    def _check_constraints(self, portfolio_value: float) -> dict[str, Any]:
        """Check if trading constraints are met.

        Args:
            portfolio_value: Current portfolio value

        Returns:
            Dictionary with constraint status
        """
        # Check drawdown guard
        if self.drawdown_guard.drawdown_triggered:
            return {"can_trade": False, "reason": "Drawdown guard triggered"}

        # Check daily loss limit
        daily_loss_pct = abs(self.daily_pnl / portfolio_value) if portfolio_value > 0 else 0
        if daily_loss_pct > self.max_daily_loss:
            return {"can_trade": False, "reason": "Daily loss limit exceeded"}

        # Check position count
        if len(self.open_positions) >= self.max_position_count:
            return {"can_trade": False, "reason": "Maximum positions reached"}

        return {"can_trade": True, "reason": None}

    def _calculate_max_loss(self, notional_value: float) -> float:
        """Calculate maximum loss for a position.

        Args:
            notional_value: Position notional value

        Returns:
            Maximum loss amount
        """
        # Use 2% stop loss as default
        stop_loss_pct = 0.02
        max_loss = notional_value * stop_loss_pct

        # Add estimated costs
        costs = self.cost_model.calculate_round_trip_costs(
            notional_value=notional_value,
            holding_hours=24,  # Assume 1 day hold
        )

        return max_loss + float(costs["total_cost"])

    def _check_daily_reset(self) -> None:
        """Reset daily metrics if new day."""
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = current_date
            logger.info("Daily P&L reset")

    def get_risk_report(self) -> dict[str, Any]:
        """Generate comprehensive risk report.

        Returns:
            Dictionary with detailed risk metrics
        """
        # API metrics
        api_metrics = self.api_throttler.get_metrics()

        # Cost estimates
        if self.open_positions:
            avg_position = np.mean(
                [pos["size"] * pos["entry_price"] for pos in self.open_positions.values()]
            )
        else:
            avg_position = 0

        cost_estimates = self.cost_model.estimate_annual_costs(
            avg_position_size=avg_position,
            trades_per_day=10,  # Estimate
            avg_holding_hours=24,
        )

        return {
            "timestamp": datetime.now().isoformat(),
            "portfolio_metrics": {
                "open_positions": len(self.open_positions),
                "daily_pnl": self.daily_pnl,
                "drawdown_triggered": self.drawdown_guard.drawdown_triggered,
            },
            "risk_limits": {
                "max_drawdown": self.drawdown_guard.max_drawdown,
                "max_daily_loss": self.max_daily_loss,
                "max_positions": self.max_position_count,
            },
            "api_metrics": api_metrics,
            "cost_estimates": cost_estimates,
        }
