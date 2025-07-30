#!/usr/bin/env python3
"""Verify Risk Management System Implementation."""

import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, 'src')

from risk_management.risk_manager import RiskManager
from risk_management.models.position_sizing import (
    KellyPositionSizer,
    FixedFractionalPositionSizer,
    VolatilityParityPositionSizer,
)


def print_section(title):
    """Print section header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def verify_position_sizing():
    """Verify position sizing strategies."""
    print_section("Position Sizing Verification")
    
    portfolio = 100000
    btc_price = 50000
    
    # Test Kelly Criterion
    kelly = KellyPositionSizer(kelly_fraction=0.25, min_edge=0.02)
    kelly_size = kelly.calculate_position_size(
        portfolio_value=portfolio,
        price=btc_price,
        confidence=0.8,
        win_rate=0.65,
        avg_win=0.04,
        avg_loss=0.01
    )
    print(f"Kelly Criterion: {kelly_size:.4f} BTC (${kelly_size * btc_price:,.2f})")
    
    # Test Fixed Fractional
    fixed = FixedFractionalPositionSizer(risk_per_trade=0.02, stop_loss_pct=0.03)
    fixed_size = fixed.calculate_position_size(
        portfolio_value=portfolio,
        price=btc_price,
        confidence=0.8
    )
    print(f"Fixed Fractional: {fixed_size:.4f} BTC (${fixed_size * btc_price:,.2f})")
    
    # Test Volatility Parity (with sample returns)
    import numpy as np
    vol_parity = VolatilityParityPositionSizer(target_volatility=0.15)
    
    # Generate sample returns
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 30)  # 30 days of returns
    
    vol_size = vol_parity.calculate_position_size(
        portfolio_value=portfolio,
        price=btc_price,
        confidence=0.8,
        returns=returns,
        volatility=0.80  # 80% annual volatility
    )
    print(f"Volatility Parity: {vol_size:.4f} BTC (${vol_size * btc_price:,.2f})")


def verify_drawdown_protection():
    """Verify drawdown guard functionality."""
    print_section("Drawdown Protection Verification")
    
    rm = RiskManager(max_drawdown=0.10)
    
    # Simulate portfolio values
    values = [100000, 110000, 105000, 95000, 88000]  # 20% drawdown from peak
    
    for i, value in enumerate(values):
        metrics = rm.update_portfolio_value(value)
        dd_status = metrics["drawdown_status"]
        
        print(f"\nPortfolio: ${value:,}")
        print(f"  Drawdown: {dd_status['current_drawdown']*100:.1f}%")
        print(f"  Peak: ${dd_status['peak_equity']:,}")
        print(f"  Triggered: {dd_status['is_triggered']}")
        print(f"  Can Trade: {metrics['can_trade']}")


def verify_cost_calculations():
    """Verify cost model calculations."""
    print_section("Cost Model Verification")
    
    rm = RiskManager()
    
    # Test different trade sizes
    trade_sizes = [10000, 50000, 100000, 500000, 1000000]
    
    print("\nRound-trip costs by trade size:")
    print("Size ($)    | Entry | Exit  | Funding | Total | % of Trade")
    print("-" * 60)
    
    for size in trade_sizes:
        costs = rm.cost_model.calculate_round_trip_costs(
            notional_value=size,
            holding_hours=24,
            entry_is_maker=False,
            exit_is_maker=False
        )
        
        total_pct = (costs['total_cost'] / size) * 100
        print(f"{size:10,} | ${costs['entry_costs'].total:5.2f} | "
              f"${costs['exit_costs'].commission:5.2f} | ${costs['total_funding']:6.2f} | "
              f"${costs['total_cost']:7.2f} | {total_pct:.3f}%")


def verify_api_throttling():
    """Verify API rate limiting."""
    print_section("API Throttling Verification")
    
    rm = RiskManager()
    throttler = rm.api_throttler
    
    # Check initial capacity
    endpoints = ['ticker', 'new_order', 'account', 'ticker_24hr']
    
    print("\nEndpoint capacities:")
    for endpoint in endpoints:
        capacity = throttler.get_remaining_capacity(endpoint)
        weight = throttler.ENDPOINT_WEIGHTS.get(endpoint, None)
        if weight:
            print(f"  {endpoint}: {capacity} requests "
                  f"(weight: {weight.weight} per request)")
    
    # Get current usage
    usage = throttler.get_current_usage()
    for limit_name, stats in usage.items():
        print(f"\n{limit_name}:")
        print(f"  Current: {stats['current_weight']}/{stats['weight_limit']}")
        print(f"  Usage: {stats['usage_percentage']:.1f}%")
        print(f"  Reset in: {stats['time_until_reset']:.1f}s")


def verify_risk_manager_integration():
    """Verify integrated risk management."""
    print_section("Risk Manager Integration")
    
    # Create risk manager with specific settings
    rm = RiskManager(
        position_sizer=KellyPositionSizer(kelly_fraction=0.25),
        max_drawdown=0.10,
        max_daily_loss=0.05,
        max_position_count=3
    )
    
    print("\nRisk Limits:")
    print(f"  Max Drawdown: {rm.drawdown_guard.max_drawdown*100:.0f}%")
    print(f"  Max Daily Loss: {rm.max_daily_loss*100:.0f}%")
    print(f"  Max Positions: {rm.max_position_count}")
    
    # Test position approval
    result = rm.check_new_position(
        symbol="BTC/USDT",
        portfolio_value=100000,
        current_price=50000,
        signal_confidence=0.75,
        win_rate=0.60,
        avg_win=0.03,
        avg_loss=0.01,
        use_limit_order=True,
        urgency=0.3
    )
    
    print("\nPosition Check Result:")
    print(f"  Approved: {result['approved']}")
    if result['approved']:
        print(f"  Position Size: {result['position_size']:.4f} BTC")
        print(f"  Notional Value: ${result['notional_value']:,.2f}")
        print(f"  Entry Costs: ${result['entry_costs']['total_cost']:.2f}")
        print(f"  Max Loss: ${result['max_loss']:.2f}")
        print(f"  Risk Multiplier: {result['risk_multiplier']:.2f}")
    else:
        print(f"  Reason: {result['reason']}")
    
    # Generate risk report
    report = rm.get_risk_report()
    print("\nRisk Report Summary:")
    print(f"  Timestamp: {report['timestamp']}")
    print(f"  Open Positions: {report['portfolio_metrics']['open_positions']}")
    print(f"  Daily P&L: ${report['portfolio_metrics']['daily_pnl']:,.2f}")


def main():
    """Run all verifications."""
    print("\n" + "="*60)
    print(" RISK MANAGEMENT SYSTEM VERIFICATION")
    print("="*60)
    
    try:
        verify_position_sizing()
        verify_drawdown_protection()
        verify_cost_calculations()
        verify_api_throttling()
        verify_risk_manager_integration()
        
        print("\n" + "="*60)
        print(" ✅ All verifications completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()