"""Binance API rate limit management."""

import asyncio
import time
from typing import Dict, Optional, List, Callable, Any
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimitRule:
    """Rate limit rule definition."""

    name: str
    weight_limit: int
    time_window: int  # seconds
    current_weight: int = 0
    window_start: float = field(default_factory=time.time)


@dataclass
class APIEndpoint:
    """API endpoint configuration."""

    name: str
    weight: int
    category: str  # 'market_data', 'trading', 'account'


class BinanceAPIThrottler:
    """Manage Binance API rate limits.

    Binance rate limits:
    - Weight limits: 1200 per minute
    - Order limits: 100 per 10 seconds, 200000 per day
    - Different endpoints have different weights
    """

    # Binance rate limit rules
    WEIGHT_LIMITS = {
        "1m": RateLimitRule("weight_1m", 1200, 60),
        "order_10s": RateLimitRule("order_10s", 100, 10),
        "order_day": RateLimitRule("order_day", 200000, 86400),
    }

    # Common endpoint weights
    ENDPOINT_WEIGHTS = {
        # Market data
        "ticker": APIEndpoint("ticker", 1, "market_data"),
        "depth": APIEndpoint("depth", 1, "market_data"),
        "trades": APIEndpoint("trades", 1, "market_data"),
        "klines": APIEndpoint("klines", 1, "market_data"),
        "ticker_24hr": APIEndpoint("ticker_24hr", 40, "market_data"),
        
        # Account
        "account": APIEndpoint("account", 5, "account"),
        "balance": APIEndpoint("balance", 5, "account"),
        "positions": APIEndpoint("positions", 5, "account"),
        
        # Trading
        "new_order": APIEndpoint("new_order", 1, "trading"),
        "cancel_order": APIEndpoint("cancel_order", 1, "trading"),
        "open_orders": APIEndpoint("open_orders", 40, "trading"),
        "all_orders": APIEndpoint("all_orders", 10, "trading"),
    }

    def __init__(
        self,
        safety_margin: float = 0.8,
        burst_allowance: float = 0.2,
        enable_queueing: bool = True,
    ):
        """Initialize API throttler.

        Args:
            safety_margin: Use only this fraction of rate limit
            burst_allowance: Allow burst up to this fraction above normal
            enable_queueing: Whether to queue requests when rate limited
        """
        self.safety_margin = safety_margin
        self.burst_allowance = burst_allowance
        self.enable_queueing = enable_queueing

        # Initialize rate limit rules
        self.limits = self.WEIGHT_LIMITS.copy()
        
        # Request queue
        self.request_queue: deque = deque()
        self.processing_queue = False
        
        # Metrics
        self.request_count = 0
        self.throttled_count = 0
        self.queued_count = 0

    async def check_and_wait(self, endpoint: str, count: int = 1) -> bool:
        """Check rate limits and wait if necessary.

        Args:
            endpoint: API endpoint name
            count: Number of requests (for batch operations)

        Returns:
            True if request can proceed, False if rejected
        """
        endpoint_info = self.ENDPOINT_WEIGHTS.get(endpoint)
        if not endpoint_info:
            logger.warning(f"Unknown endpoint: {endpoint}, using weight=1")
            weight = 1
        else:
            weight = endpoint_info.weight * count

        # Check all applicable limits
        can_proceed = True
        wait_time = 0.0

        for limit_name, limit in self.limits.items():
            # Skip order limits for non-trading endpoints
            if "order" in limit_name and endpoint_info and endpoint_info.category != "trading":
                continue

            # Update window if expired
            current_time = time.time()
            if current_time - limit.window_start >= limit.time_window:
                limit.current_weight = 0
                limit.window_start = current_time

            # Check if request would exceed limit
            effective_limit = int(limit.weight_limit * self.safety_margin)
            
            if limit.current_weight + weight > effective_limit:
                # Calculate wait time
                window_remaining = limit.time_window - (current_time - limit.window_start)
                wait_time = max(wait_time, window_remaining)
                can_proceed = False

        if not can_proceed:
            self.throttled_count += 1
            
            if self.enable_queueing and wait_time < 60:  # Don't queue if wait > 1 min
                # Queue the request
                self.queued_count += 1
                await asyncio.sleep(wait_time)
                return await self.check_and_wait(endpoint, count)  # Retry
            else:
                logger.warning(
                    f"Rate limit would be exceeded for {endpoint}, "
                    f"need to wait {wait_time:.1f}s"
                )
                return False

        # Update weights
        for limit_name, limit in self.limits.items():
            if "order" in limit_name and endpoint_info and endpoint_info.category != "trading":
                continue
            limit.current_weight += weight

        self.request_count += 1
        return True

    def get_current_usage(self) -> Dict[str, Dict[str, Any]]:
        """Get current rate limit usage.

        Returns:
            Dictionary with usage statistics for each limit
        """
        usage = {}
        current_time = time.time()

        for limit_name, limit in self.limits.items():
            # Check if window expired
            window_elapsed = current_time - limit.window_start
            if window_elapsed >= limit.time_window:
                # Window expired, would reset
                usage_pct = 0.0
                time_until_reset = 0.0
            else:
                usage_pct = (limit.current_weight / limit.weight_limit) * 100
                time_until_reset = limit.time_window - window_elapsed

            usage[limit_name] = {
                "current_weight": limit.current_weight,
                "weight_limit": limit.weight_limit,
                "usage_percentage": usage_pct,
                "time_until_reset": time_until_reset,
                "safety_margin": self.safety_margin,
                "effective_limit": int(limit.weight_limit * self.safety_margin),
            }

        return usage

    def get_remaining_capacity(self, endpoint: str) -> int:
        """Get remaining capacity for an endpoint.
        
        Args:
            endpoint: API endpoint name
            
        Returns:
            Number of requests that can still be made
        """
        endpoint_info = self.ENDPOINT_WEIGHTS.get(endpoint)
        if not endpoint_info:
            return 0
            
        weight = endpoint_info.weight
        effective_limit = int(self.limits["1m"].weight_limit * self.safety_margin)
        current_weight = self.limits["1m"].current_weight
        
        remaining_weight = max(0, effective_limit - current_weight)
        return int(remaining_weight / weight) if weight > 0 else 0
    
    def estimate_capacity(self, endpoint: str, time_horizon: int = 60) -> int:
        """Estimate how many requests can be made in time horizon.

        Args:
            endpoint: API endpoint name
            time_horizon: Time period in seconds

        Returns:
            Estimated number of requests possible
        """
        endpoint_info = self.ENDPOINT_WEIGHTS.get(endpoint)
        if not endpoint_info:
            return 0

        weight = endpoint_info.weight
        
        # For simplicity, use the 1-minute weight limit
        weight_limit = self.limits["1m"].weight_limit * self.safety_margin
        
        # Calculate based on time horizon
        if time_horizon <= 60:
            capacity = int(weight_limit / weight)
        else:
            # Multiple minutes
            minutes = time_horizon / 60
            capacity = int(weight_limit * minutes / weight)

        return capacity

    def create_batch_executor(
        self, 
        endpoint: str,
        max_batch_size: int = 100
    ) -> Callable:
        """Create a batch executor for efficient API usage.

        Args:
            endpoint: API endpoint for the batch
            max_batch_size: Maximum items per batch

        Returns:
            Async function that executes batched requests
        """
        async def execute_batch(items: List[Any], api_func: Callable) -> List[Any]:
            """Execute items in batches respecting rate limits."""
            results = []
            
            for i in range(0, len(items), max_batch_size):
                batch = items[i : i + max_batch_size]
                
                # Check rate limit for batch
                can_proceed = await self.check_and_wait(endpoint, len(batch))
                if not can_proceed:
                    logger.error(f"Failed to get rate limit clearance for batch")
                    continue
                
                # Execute batch
                try:
                    batch_results = await api_func(batch)
                    results.extend(batch_results)
                except Exception as e:
                    logger.error(f"Batch execution failed: {e}")
                    # Could implement retry logic here
            
            return results
        
        return execute_batch

    def get_metrics(self) -> Dict[str, Any]:
        """Get throttler metrics.

        Returns:
            Dictionary with performance metrics
        """
        return {
            "total_requests": self.request_count,
            "throttled_requests": self.throttled_count,
            "queued_requests": self.queued_count,
            "throttle_rate": (
                self.throttled_count / self.request_count if self.request_count > 0 else 0
            ),
            "current_usage": self.get_current_usage(),
        }

    def reset(self) -> None:
        """Reset throttler state."""
        for limit in self.limits.values():
            limit.current_weight = 0
            limit.window_start = time.time()
        
        self.request_count = 0
        self.throttled_count = 0
        self.queued_count = 0
        self.request_queue.clear()
        
        logger.info("API throttler reset")
    
    def check_and_wait_sync(self, endpoint: str, count: int = 1) -> bool:
        """Synchronous version of check_and_wait for non-async contexts.
        
        Args:
            endpoint: API endpoint name
            count: Number of requests
            
        Returns:
            True if request can proceed
        """
        # Create new event loop for sync context
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we can't use run_until_complete
                # Just do a simple synchronous check without waiting
                endpoint_config = self.ENDPOINT_WEIGHTS.get(endpoint)
                if not endpoint_config:
                    return True
                
                weight = endpoint_config.weight * count
                # Check if we have capacity
                for rule in self.limits.values():
                    elapsed = time.time() - rule.window_start
                    if elapsed >= rule.time_window:
                        rule.current_weight = 0
                        rule.window_start = time.time()
                    
                    if rule.current_weight + weight > rule.weight_limit * self.safety_margin:
                        return False
                
                # Update weights
                for rule in self.limits.values():
                    rule.current_weight += weight
                
                self.request_count += count
                return True
            else:
                # Safe to run in new loop
                return loop.run_until_complete(self.check_and_wait(endpoint, count))
        except RuntimeError:
            # No event loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.check_and_wait(endpoint, count))
            finally:
                loop.close()