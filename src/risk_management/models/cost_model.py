"""Trading cost model for accurate PnL calculation."""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradeCosts:
    """Container for trade cost breakdown."""

    commission: float
    slippage: float
    funding: float
    total: float


class CostModel:
    """Model trading costs including commissions, slippage, and funding.

    Binance Futures fee structure:
    - Maker: 0.02% (0.0002)
    - Taker: 0.05% (0.0005)
    - VIP levels can reduce fees
    - Funding rate: Variable (typically ±0.01% every 8 hours)
    """

    def __init__(
        self,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0005,
        slippage_bps: float = 5.0,  # basis points
        funding_rate: float = 0.0001,  # per 8 hours
        vip_level: int = 0,
    ):
        """Initialize cost model.

        Args:
            maker_fee: Maker fee rate
            taker_fee: Taker fee rate
            slippage_bps: Expected slippage in basis points
            funding_rate: Average funding rate per period
            vip_level: VIP level for fee discounts
        """
        self.base_maker_fee = maker_fee
        self.base_taker_fee = taker_fee
        self.slippage_bps = slippage_bps
        self.funding_rate = funding_rate
        self.vip_level = vip_level

        # Apply VIP discounts
        self.maker_fee = self._apply_vip_discount(self.base_maker_fee)
        self.taker_fee = self._apply_vip_discount(self.base_taker_fee)

        logger.info(
            f"Cost model initialized - Maker: {self.maker_fee:.4%}, "
            f"Taker: {self.taker_fee:.4%}, Slippage: {slippage_bps}bps"
        )

    def _apply_vip_discount(self, base_fee: float) -> float:
        """Apply VIP level discount to fees.

        Binance VIP discounts (approximate):
        - VIP 0: 0% discount
        - VIP 1: 5% discount
        - VIP 2: 10% discount
        - VIP 3: 15% discount
        - ...up to VIP 9
        """
        discount_rates = {
            0: 0.00,
            1: 0.05,
            2: 0.10,
            3: 0.15,
            4: 0.20,
            5: 0.25,
            6: 0.30,
            7: 0.35,
            8: 0.40,
            9: 0.45,
        }

        discount = discount_rates.get(self.vip_level, 0.0)
        return base_fee * (1 - discount)

    def calculate_entry_costs(
        self,
        notional_value: float,
        is_maker: bool = False,
        urgency: float = 0.5,
    ) -> TradeCosts:
        """Calculate costs for entering a position.

        Args:
            notional_value: Position size in USD
            is_maker: Whether order is maker (limit) or taker (market)
            urgency: Trade urgency (0-1), affects slippage

        Returns:
            TradeCosts with breakdown
        """
        # Commission
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        commission = notional_value * fee_rate

        # Slippage based on urgency and size
        # Higher urgency = more aggressive execution = more slippage
        slippage_multiplier = 1.0 + (urgency * 0.5)  # 1.0 to 1.5x
        
        # Size impact (larger orders have more slippage)
        size_impact = 1.0
        if notional_value > 1000000:  # >$1M
            size_impact = 2.0
        elif notional_value > 500000:  # >$500k
            size_impact = 1.5
        elif notional_value > 100000:  # >$100k
            size_impact = 1.2

        slippage_bps_adjusted = self.slippage_bps * slippage_multiplier * size_impact
        slippage = notional_value * (slippage_bps_adjusted / 10000)

        # No funding on entry
        funding = 0.0

        total = commission + slippage + funding

        return TradeCosts(
            commission=commission,
            slippage=slippage,
            funding=funding,
            total=total,
        )

    def calculate_exit_costs(
        self,
        notional_value: float,
        holding_hours: float,
        is_maker: bool = False,
        urgency: float = 0.5,
    ) -> TradeCosts:
        """Calculate costs for exiting a position.

        Args:
            notional_value: Position size in USD
            holding_hours: Hours position was held
            is_maker: Whether order is maker or taker
            urgency: Trade urgency (0-1)

        Returns:
            TradeCosts with breakdown
        """
        # Entry costs (commission + slippage)
        entry_costs = self.calculate_entry_costs(notional_value, is_maker, urgency)

        # Funding costs
        # Funding occurs every 8 hours
        funding_periods = holding_hours / 8.0
        funding = notional_value * self.funding_rate * funding_periods

        total = entry_costs.commission + entry_costs.slippage + funding

        return TradeCosts(
            commission=entry_costs.commission,
            slippage=entry_costs.slippage,
            funding=funding,
            total=total,
        )

    def calculate_round_trip_costs(
        self,
        notional_value: float,
        holding_hours: float,
        entry_is_maker: bool = False,
        exit_is_maker: bool = False,
        entry_urgency: float = 0.5,
        exit_urgency: float = 0.5,
    ) -> Dict[str, Any]:
        """Calculate total round-trip trading costs.

        Args:
            notional_value: Position size in USD
            holding_hours: Hours position held
            entry_is_maker: Whether entry is maker order
            exit_is_maker: Whether exit is maker order
            entry_urgency: Entry urgency
            exit_urgency: Exit urgency

        Returns:
            Dictionary with cost breakdown
        """
        # Entry costs
        entry = self.calculate_entry_costs(notional_value, entry_is_maker, entry_urgency)

        # Exit costs
        exit = self.calculate_exit_costs(
            notional_value, holding_hours, exit_is_maker, exit_urgency
        )

        # Total round trip
        total_commission = entry.commission + exit.commission
        total_slippage = entry.slippage + exit.slippage
        total_funding = exit.funding  # Only on exit
        total_cost = entry.total + exit.total

        # Calculate as percentage of notional
        cost_pct = (total_cost / notional_value) * 100

        return {
            "entry_costs": entry,
            "exit_costs": exit,
            "total_commission": total_commission,
            "total_slippage": total_slippage,
            "total_funding": total_funding,
            "total_cost": total_cost,
            "cost_percentage": cost_pct,
            "breakeven_move": cost_pct,  # Need this % move to break even
        }

    def estimate_annual_costs(
        self,
        avg_position_size: float,
        trades_per_day: float,
        avg_holding_hours: float,
        maker_ratio: float = 0.5,
    ) -> Dict[str, float]:
        """Estimate annual trading costs.

        Args:
            avg_position_size: Average position size in USD
            trades_per_day: Average number of round trips per day
            avg_holding_hours: Average holding period
            maker_ratio: Fraction of orders that are maker

        Returns:
            Annual cost estimates
        """
        # Average fees
        avg_fee = self.maker_fee * maker_ratio + self.taker_fee * (1 - maker_ratio)

        # Annual calculations
        trades_per_year = trades_per_day * 252  # Trading days
        
        # Commission costs (2x for round trip)
        annual_commission = avg_position_size * avg_fee * 2 * trades_per_year

        # Slippage costs (2x for round trip)
        annual_slippage = (
            avg_position_size * (self.slippage_bps / 10000) * 2 * trades_per_year
        )

        # Funding costs
        funding_periods_per_trade = avg_holding_hours / 8.0
        annual_funding = (
            avg_position_size * self.funding_rate * funding_periods_per_trade * trades_per_year
        )

        total_annual = annual_commission + annual_slippage + annual_funding

        return {
            "annual_commission": annual_commission,
            "annual_slippage": annual_slippage,
            "annual_funding": annual_funding,
            "total_annual_costs": total_annual,
            "cost_per_trade": total_annual / trades_per_year,
            "daily_costs": total_annual / 252,
        }