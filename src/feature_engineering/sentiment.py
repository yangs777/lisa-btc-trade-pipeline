"""Sentiment analysis features for trading."""

from typing import Dict, Any, List, Optional
import numpy as np
from collections import deque


class SentimentAnalyzer:
    """Analyze market sentiment from various sources."""
    
    def __init__(self, window_size: int = 100):
        """Initialize sentiment analyzer.
        
        Args:
            window_size: Window size for rolling calculations
        """
        self.window_size = window_size
        self.sentiment_history = deque(maxlen=window_size)
        self.volume_history = deque(maxlen=window_size)
    
    def compute_features(self, sentiment_data: Dict[str, Any]) -> Dict[str, float]:
        """Compute sentiment features.
        
        Args:
            sentiment_data: Dictionary with sentiment indicators
            
        Returns:
            Dictionary of sentiment features
        """
        features = {}
        
        # Fear & Greed Index features
        if "fear_greed_index" in sentiment_data:
            fgi = sentiment_data["fear_greed_index"]
            features["sentiment_score"] = fgi / 100  # Normalize to 0-1
            features["sentiment_extreme"] = abs(fgi - 50) / 50  # Distance from neutral
            features["sentiment_fear"] = max(0, (50 - fgi) / 50)  # Fear component
            features["sentiment_greed"] = max(0, (fgi - 50) / 50)  # Greed component
        
        # Social volume features
        if "social_volume" in sentiment_data:
            volume = sentiment_data["social_volume"]
            self.volume_history.append(volume)
            
            if len(self.volume_history) > 1:
                features["social_volume_change"] = (
                    volume - np.mean(self.volume_history)
                ) / (np.std(self.volume_history) + 1e-6)
                features["social_volume_spike"] = volume > np.percentile(self.volume_history, 90)
            else:
                features["social_volume_change"] = 0
                features["social_volume_spike"] = False
        
        # News sentiment
        if "news_sentiment" in sentiment_data:
            news_sent = sentiment_data["news_sentiment"]
            features["news_sentiment"] = news_sent
            features["news_bullish"] = max(0, news_sent)
            features["news_bearish"] = max(0, -news_sent)
        
        # Momentum of sentiment
        if features.get("sentiment_score") is not None:
            self.sentiment_history.append(features["sentiment_score"])
            
            if len(self.sentiment_history) >= 2:
                # Short-term momentum
                features["sentiment_momentum"] = (
                    self.sentiment_history[-1] - self.sentiment_history[-2]
                )
                
                # Longer-term momentum
                if len(self.sentiment_history) >= 10:
                    features["sentiment_momentum_10"] = (
                        np.mean(list(self.sentiment_history)[-5:]) -
                        np.mean(list(self.sentiment_history)[-10:-5])
                    )
        
        # Composite sentiment score
        weights = {
            "sentiment_score": 0.4,
            "news_sentiment": 0.3,
            "social_volume_change": 0.3
        }
        
        composite = 0
        total_weight = 0
        
        for feature, weight in weights.items():
            if feature in features:
                composite += features[feature] * weight
                total_weight += weight
        
        if total_weight > 0:
            features["composite_sentiment"] = composite / total_weight
        
        return features
    
    def analyze_crowd_behavior(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Analyze crowd behavior patterns.
        
        Args:
            metrics: Market metrics including volume, volatility
            
        Returns:
            Crowd behavior features
        """
        features = {}
        
        # FOMO indicator (Fear of Missing Out)
        if "price_change_24h" in metrics and "volume_change_24h" in metrics:
            price_change = metrics["price_change_24h"]
            volume_change = metrics["volume_change_24h"]
            
            # FOMO score: high positive price change + high volume increase
            features["fomo_score"] = (
                max(0, price_change / 100) * max(0, volume_change / 100)
            )
        
        # Panic indicator
        if "price_change_1h" in metrics and "volatility" in metrics:
            price_drop = min(0, metrics["price_change_1h"])
            volatility = metrics["volatility"]
            
            # Panic score: sharp price drop + high volatility
            features["panic_score"] = abs(price_drop / 100) * volatility
        
        # Herd behavior
        if "long_ratio" in metrics:
            long_ratio = metrics["long_ratio"]
            features["herd_bias"] = abs(long_ratio - 0.5) * 2  # 0 = balanced, 1 = extreme
            features["herd_direction"] = 1 if long_ratio > 0.5 else -1
        
        return features
    
    def get_sentiment_signals(self) -> Dict[str, float]:
        """Get current sentiment-based trading signals.
        
        Returns:
            Dictionary of signals
        """
        if len(self.sentiment_history) < 2:
            return {"signal": 0, "confidence": 0}
        
        current = self.sentiment_history[-1]
        previous = self.sentiment_history[-2]
        
        # Basic contrarian signals
        signal = 0
        confidence = 0
        
        # Extreme fear = potential buy
        if current < 0.2:
            signal = 1
            confidence = (0.2 - current) / 0.2
        
        # Extreme greed = potential sell
        elif current > 0.8:
            signal = -1
            confidence = (current - 0.8) / 0.2
        
        # Momentum reversal
        elif len(self.sentiment_history) >= 5:
            recent_avg = np.mean(list(self.sentiment_history)[-5:])
            
            # Bullish reversal
            if recent_avg < 0.3 and current > previous:
                signal = 0.5
                confidence = 0.5
            
            # Bearish reversal
            elif recent_avg > 0.7 and current < previous:
                signal = -0.5
                confidence = 0.5
        
        return {
            "signal": signal,
            "confidence": confidence,
            "sentiment_level": current
        }