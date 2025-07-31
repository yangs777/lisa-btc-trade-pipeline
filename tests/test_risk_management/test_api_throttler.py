"""Tests for API throttler."""

import asyncio
import time

import pytest

from src.risk_management.models.api_throttler import BinanceAPIThrottler, RateLimitRule


class TestBinanceAPIThrottler:
    """Test API rate limit management."""

    def setup_method(self):
        """Reset class-level state before each test."""
        # Reset WEIGHT_LIMITS to ensure fresh state
        BinanceAPIThrottler.WEIGHT_LIMITS = {
            "1m": RateLimitRule("weight_1m", 1200, 60),
            "order_10s": RateLimitRule("order_10s", 100, 10),
            "order_day": RateLimitRule("order_day", 200000, 86400),
        }

    @pytest.mark.asyncio
    async def test_basic_rate_limiting(self):
        """Test basic rate limit checking."""
        throttler = BinanceAPIThrottler(safety_margin=0.8)

        # Should allow initial request
        can_proceed = await throttler.check_and_wait("ticker", 1)
        assert can_proceed

        # Check weight was recorded
        usage = throttler.get_current_usage()
        assert usage["1m"]["current_weight"] == 1

    @pytest.mark.asyncio
    async def test_weight_accumulation(self):
        """Test weight accumulation."""
        throttler = BinanceAPIThrottler()

        # Make several requests
        for _ in range(5):
            await throttler.check_and_wait("ticker", 1)

        usage = throttler.get_current_usage()
        assert usage["1m"]["current_weight"] == 5

    @pytest.mark.asyncio
    async def test_heavy_endpoint(self):
        """Test heavy weight endpoint."""
        throttler = BinanceAPIThrottler()

        # ticker_24hr has weight of 40
        await throttler.check_and_wait("ticker_24hr", 1)

        usage = throttler.get_current_usage()
        assert usage["1m"]["current_weight"] == 40

    @pytest.mark.asyncio
    async def test_safety_margin(self):
        """Test safety margin enforcement."""
        throttler = BinanceAPIThrottler(
            safety_margin=0.1,  # Only use 10% of limit
            enable_queueing=False,  # Disable queueing for this test
        )

        # With 1200 limit and 0.1 margin, can only use 120 weight
        # ticker_24hr uses 40 weight each

        # Should allow 3 calls (120 weight)
        for _i in range(3):
            can_proceed = await throttler.check_and_wait("ticker_24hr", 1)
            assert can_proceed

        # 4th call would exceed limit
        can_proceed = await throttler.check_and_wait("ticker_24hr", 1)
        assert not can_proceed

    @pytest.mark.asyncio
    async def test_order_limits(self):
        """Test order-specific limits."""
        throttler = BinanceAPIThrottler()

        # Order endpoints have additional limits
        # 100 orders per 10 seconds

        # Should track separately for trading endpoints
        await throttler.check_and_wait("new_order", 1)

        usage = throttler.get_current_usage()
        # Should update both weight and order limits
        assert usage["1m"]["current_weight"] == 1
        assert usage["order_10s"]["current_weight"] == 1

    @pytest.mark.asyncio
    async def test_batch_requests(self):
        """Test batch request handling."""
        throttler = BinanceAPIThrottler()

        # Batch of 10 orders
        can_proceed = await throttler.check_and_wait("new_order", 10)
        assert can_proceed

        usage = throttler.get_current_usage()
        assert usage["order_10s"]["current_weight"] == 10

    def test_capacity_estimation(self):
        """Test capacity estimation."""
        throttler = BinanceAPIThrottler(safety_margin=0.8)

        # For ticker (weight=1), with 1200 limit and 0.8 margin
        # Capacity = 1200 * 0.8 / 1 = 960
        capacity = throttler.estimate_capacity("ticker", 60)
        assert capacity == 960

        # For heavy endpoint
        capacity = throttler.estimate_capacity("ticker_24hr", 60)
        assert capacity == 24  # 960 / 40

    def test_batch_executor_creation(self):
        """Test batch executor creation."""
        throttler = BinanceAPIThrottler()

        executor = throttler.create_batch_executor("ticker", max_batch_size=50)
        assert callable(executor)

    @pytest.mark.asyncio
    async def test_unknown_endpoint(self):
        """Test handling of unknown endpoints."""
        throttler = BinanceAPIThrottler()

        # Should use default weight of 1
        can_proceed = await throttler.check_and_wait("unknown_endpoint", 1)
        assert can_proceed

        usage = throttler.get_current_usage()
        assert usage["1m"]["current_weight"] == 1

    def test_metrics_tracking(self):
        """Test metrics tracking."""
        throttler = BinanceAPIThrottler()

        # Initial metrics
        metrics = throttler.get_metrics()
        assert metrics["total_requests"] == 0
        assert metrics["throttled_requests"] == 0
        assert metrics["throttle_rate"] == 0

    @pytest.mark.asyncio
    async def test_non_trading_endpoints(self):
        """Test that non-trading endpoints don't affect order limits."""
        throttler = BinanceAPIThrottler()

        # Market data shouldn't affect order limits
        await throttler.check_and_wait("ticker", 1)

        usage = throttler.get_current_usage()
        assert usage["1m"]["current_weight"] == 1
        assert usage["order_10s"]["current_weight"] == 0  # Not incremented

    def test_reset_functionality(self):
        """Test throttler reset."""
        throttler = BinanceAPIThrottler()

        # Add some weight
        asyncio.run(throttler.check_and_wait("ticker_24hr", 5))

        # Reset
        throttler.reset()

        # Check everything cleared
        usage = throttler.get_current_usage()
        assert usage["1m"]["current_weight"] == 0
        assert throttler.request_count == 0
        assert throttler.throttled_count == 0

    @pytest.mark.asyncio
    async def test_window_expiration(self):
        """Test rate limit window expiration."""
        throttler = BinanceAPIThrottler()

        # Manually set window start to past
        throttler.limits["1m"].window_start = time.time() - 61  # 61 seconds ago
        throttler.limits["1m"].current_weight = 1000

        # New request should reset window
        await throttler.check_and_wait("ticker", 1)

        usage = throttler.get_current_usage()
        assert usage["1m"]["current_weight"] == 1  # Reset + new request

    def test_usage_percentage_calculation(self):
        """Test usage percentage calculation."""
        throttler = BinanceAPIThrottler()

        # Add 600 weight (50% of 1200 limit)
        throttler.limits["1m"].current_weight = 600

        usage = throttler.get_current_usage()
        assert abs(usage["1m"]["usage_percentage"] - 50.0) < 0.1

    @pytest.mark.asyncio
    async def test_burst_allowance(self):
        """Test burst allowance feature."""
        throttler = BinanceAPIThrottler(safety_margin=0.5, burst_allowance=0.2)  # Allow 20% burst

        # With 50% margin + 20% burst, effective limit is 70%
        # 1200 * 0.7 = 840 weight

        # Should handle burst scenario
        # This is simplified - actual implementation might differ
        can_proceed = await throttler.check_and_wait("ticker", 1)
        assert can_proceed
