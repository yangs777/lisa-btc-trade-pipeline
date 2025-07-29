"""Tests for trading cost model."""

from src.risk_management.models.cost_model import CostModel


class TestCostModel:
    """Test trading cost calculations."""

    def test_basic_commission_calculation(self):
        """Test basic commission calculation."""
        model = CostModel(
            maker_fee=0.0002,
            taker_fee=0.0005,
        )

        # Taker order
        costs = model.calculate_entry_costs(
            notional_value=100000,
            is_maker=False,
        )

        assert costs.commission == 100000 * 0.0005  # $50
        assert costs.commission == 50.0

    def test_maker_vs_taker_fees(self):
        """Test difference between maker and taker fees."""
        model = CostModel()

        # Maker order
        maker_costs = model.calculate_entry_costs(
            notional_value=100000,
            is_maker=True,
        )

        # Taker order
        taker_costs = model.calculate_entry_costs(
            notional_value=100000,
            is_maker=False,
        )

        assert maker_costs.commission < taker_costs.commission
        assert maker_costs.commission == 20.0  # 0.02%
        assert taker_costs.commission == 50.0  # 0.05%

    def test_slippage_calculation(self):
        """Test slippage calculation."""
        model = CostModel(slippage_bps=5.0)

        # Normal urgency
        costs_normal = model.calculate_entry_costs(
            notional_value=100000,
            urgency=0.5,
        )

        # High urgency
        costs_urgent = model.calculate_entry_costs(
            notional_value=100000,
            urgency=1.0,
        )

        # High urgency should have more slippage
        assert costs_urgent.slippage > costs_normal.slippage

        # Normal: 5bps * 1.25 = 6.25bps = $62.50
        assert abs(costs_normal.slippage - 62.50) < 1.0

        # Urgent: 5bps * 1.5 = 7.5bps = $75
        assert abs(costs_urgent.slippage - 75.0) < 1.0

    def test_size_impact_on_slippage(self):
        """Test size impact on slippage."""
        model = CostModel(slippage_bps=5.0)

        # Small order
        costs_small = model.calculate_entry_costs(
            notional_value=50000,
            urgency=0.5,
        )

        # Large order (>$100k)
        costs_large = model.calculate_entry_costs(
            notional_value=200000,
            urgency=0.5,
        )

        # Very large order (>$1M)
        costs_xlarge = model.calculate_entry_costs(
            notional_value=1500000,
            urgency=0.5,
        )

        # Slippage per dollar should increase with size
        slip_small = costs_small.slippage / 50000
        slip_large = costs_large.slippage / 200000
        slip_xlarge = costs_xlarge.slippage / 1500000

        assert slip_large > slip_small
        assert slip_xlarge > slip_large

    def test_funding_costs(self):
        """Test funding cost calculation."""
        model = CostModel(funding_rate=0.0001)  # 0.01% per 8 hours

        # Hold for 24 hours (3 funding periods)
        costs = model.calculate_exit_costs(
            notional_value=100000,
            holding_hours=24,
        )

        # 3 periods * 0.01% * $100k = $30
        assert abs(costs.funding - 30.0) < 0.1

    def test_round_trip_costs(self):
        """Test complete round trip cost calculation."""
        model = CostModel()

        result = model.calculate_round_trip_costs(
            notional_value=100000,
            holding_hours=48,
            entry_is_maker=False,
            exit_is_maker=True,
            entry_urgency=0.8,
            exit_urgency=0.3,
        )

        # Check components
        assert result["total_commission"] > 0
        assert result["total_slippage"] > 0
        assert result["total_funding"] > 0

        # Entry: taker fee + high urgency slippage
        # Exit: maker fee + low urgency slippage
        # Funding: 48 hours = 6 periods

        # Total should be sum of components
        total = result["total_commission"] + result["total_slippage"] + result["total_funding"]
        assert abs(result["total_cost"] - total) < 0.01

    def test_vip_discounts(self):
        """Test VIP level fee discounts."""
        # No discount
        model_vip0 = CostModel(vip_level=0)
        costs_vip0 = model_vip0.calculate_entry_costs(100000)

        # VIP 3 (15% discount)
        model_vip3 = CostModel(vip_level=3)
        costs_vip3 = model_vip3.calculate_entry_costs(100000)

        # VIP fees should be lower
        assert costs_vip3.commission < costs_vip0.commission
        assert abs(costs_vip3.commission - costs_vip0.commission * 0.85) < 0.01

    def test_annual_cost_estimation(self):
        """Test annual cost estimation."""
        model = CostModel()

        estimates = model.estimate_annual_costs(
            avg_position_size=50000,
            trades_per_day=5,
            avg_holding_hours=12,
            maker_ratio=0.7,
        )

        # Should have all components
        assert "annual_commission" in estimates
        assert "annual_slippage" in estimates
        assert "annual_funding" in estimates
        assert "total_annual_costs" in estimates
        assert "cost_per_trade" in estimates
        assert "daily_costs" in estimates

        # Sanity checks
        assert estimates["total_annual_costs"] > 0
        assert estimates["cost_per_trade"] == estimates["total_annual_costs"] / (5 * 252)
        assert estimates["daily_costs"] == estimates["total_annual_costs"] / 252

    def test_breakeven_calculation(self):
        """Test breakeven move calculation."""
        model = CostModel()

        result = model.calculate_round_trip_costs(
            notional_value=100000,
            holding_hours=24,
        )

        # Breakeven should equal cost percentage
        assert result["breakeven_move"] == result["cost_percentage"]

        # For $100k position, if costs are $200, need 0.2% move to break even
        if result["total_cost"] == 200:
            assert result["breakeven_move"] == 0.2

    def test_edge_cases(self):
        """Test edge cases."""
        model = CostModel()

        # Zero notional
        costs = model.calculate_entry_costs(
            notional_value=0,
            is_maker=True,
        )
        assert costs.total == 0

        # Zero holding time (scalping)
        costs = model.calculate_exit_costs(
            notional_value=100000,
            holding_hours=0,
        )
        assert costs.funding == 0  # No funding for instant trade
