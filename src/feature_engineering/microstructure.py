"""Market microstructure features."""

from typing import Dict, List, Any
import numpy as np


class MicrostructureFeatures:
    """Extract market microstructure features from order book data."""
    
    def __init__(self, depth_levels: int = 20):
        """Initialize microstructure feature extractor.
        
        Args:
            depth_levels: Number of order book levels to use
        """
        self.depth_levels = depth_levels
    
    def compute(self, orderbook: Dict[str, List[List[float]]]) -> Dict[str, float]:
        """Compute microstructure features from order book.
        
        Args:
            orderbook: Dictionary with 'bids' and 'asks' lists
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Extract bid/ask data
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        
        if not bids or not asks:
            return self._empty_features()
        
        # Basic spread features
        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 0
        
        features["bid_ask_spread"] = best_ask - best_bid
        features["spread_bps"] = (features["bid_ask_spread"] / best_bid * 10000) if best_bid > 0 else 0
        features["mid_price"] = (best_bid + best_ask) / 2
        
        # Volume imbalance
        bid_volume = sum(level[1] for level in bids[:self.depth_levels])
        ask_volume = sum(level[1] for level in asks[:self.depth_levels])
        total_volume = bid_volume + ask_volume
        
        features["order_book_imbalance"] = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
        features["bid_volume"] = bid_volume
        features["ask_volume"] = ask_volume
        
        # Depth-weighted prices
        features["depth_weighted_bidprice"] = self._weighted_price(bids[:self.depth_levels])
        features["depth_weighted_askprice"] = self._weighted_price(asks[:self.depth_levels])
        features["depth_weighted_midprice"] = (
            features["depth_weighted_bidprice"] + features["depth_weighted_askprice"]
        ) / 2
        
        # Price levels
        features["bid_levels"] = len(bids)
        features["ask_levels"] = len(asks)
        
        # Liquidity measures
        features["liquidity_1bps"] = self._liquidity_at_distance(bids, asks, best_bid, 0.0001)
        features["liquidity_5bps"] = self._liquidity_at_distance(bids, asks, best_bid, 0.0005)
        features["liquidity_10bps"] = self._liquidity_at_distance(bids, asks, best_bid, 0.0010)
        
        # Slope of order book
        features["bid_slope"] = self._orderbook_slope(bids[:self.depth_levels])
        features["ask_slope"] = self._orderbook_slope(asks[:self.depth_levels])
        
        return features
    
    def _weighted_price(self, levels: List[List[float]]) -> float:
        """Calculate volume-weighted average price.
        
        Args:
            levels: List of [price, volume] pairs
            
        Returns:
            Weighted average price
        """
        if not levels:
            return 0.0
        
        total_volume = sum(level[1] for level in levels)
        if total_volume == 0:
            return levels[0][0] if levels else 0.0
        
        weighted_sum = sum(level[0] * level[1] for level in levels)
        return weighted_sum / total_volume
    
    def _liquidity_at_distance(
        self,
        bids: List[List[float]],
        asks: List[List[float]],
        mid_price: float,
        distance_ratio: float
    ) -> float:
        """Calculate liquidity within a price distance.
        
        Args:
            bids: Bid levels
            asks: Ask levels
            mid_price: Reference price
            distance_ratio: Distance as ratio of price
            
        Returns:
            Total liquidity within distance
        """
        distance = mid_price * distance_ratio
        bid_threshold = mid_price - distance
        ask_threshold = mid_price + distance
        
        bid_liquidity = sum(
            level[1] for level in bids
            if level[0] >= bid_threshold
        )
        
        ask_liquidity = sum(
            level[1] for level in asks
            if level[0] <= ask_threshold
        )
        
        return bid_liquidity + ask_liquidity
    
    def _orderbook_slope(self, levels: List[List[float]]) -> float:
        """Calculate slope of order book depth.
        
        Args:
            levels: Order book levels
            
        Returns:
            Slope coefficient
        """
        if len(levels) < 2:
            return 0.0
        
        # Extract prices and cumulative volumes
        prices = [level[0] for level in levels]
        volumes = [level[1] for level in levels]
        cum_volumes = np.cumsum(volumes)
        
        # Calculate slope using linear regression
        if len(prices) > 1:
            # Normalize prices to avoid numerical issues
            price_range = prices[-1] - prices[0]
            if price_range > 0:
                norm_prices = [(p - prices[0]) / price_range for p in prices]
                
                # Simple linear regression
                n = len(norm_prices)
                xy = sum(p * v for p, v in zip(norm_prices, cum_volumes))
                x_sum = sum(norm_prices)
                y_sum = sum(cum_volumes)
                x2 = sum(p * p for p in norm_prices)
                
                denominator = n * x2 - x_sum * x_sum
                if denominator != 0:
                    slope = (n * xy - x_sum * y_sum) / denominator
                    return slope
        
        return 0.0
    
    def _empty_features(self) -> Dict[str, float]:
        """Return empty feature set when no data available."""
        return {
            "bid_ask_spread": 0.0,
            "spread_bps": 0.0,
            "mid_price": 0.0,
            "order_book_imbalance": 0.0,
            "bid_volume": 0.0,
            "ask_volume": 0.0,
            "depth_weighted_bidprice": 0.0,
            "depth_weighted_askprice": 0.0,
            "depth_weighted_midprice": 0.0,
            "bid_levels": 0,
            "ask_levels": 0,
            "liquidity_1bps": 0.0,
            "liquidity_5bps": 0.0,
            "liquidity_10bps": 0.0,
            "bid_slope": 0.0,
            "ask_slope": 0.0,
        }