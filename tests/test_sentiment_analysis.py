"""Test coverage for sentiment analysis module."""

import pytest
import numpy as np
from src.feature_engineering.sentiment import SentimentAnalyzer


class TestSentimentAnalyzer:
    """Test SentimentAnalyzer class."""
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = SentimentAnalyzer(window_size=50)
        
        assert analyzer.window_size == 50
        assert len(analyzer.sentiment_history) == 0
        assert len(analyzer.volume_history) == 0
    
    def test_compute_features_with_fear_greed_index(self):
        """Test computing features with Fear & Greed Index."""
        analyzer = SentimentAnalyzer()
        
        sentiment_data = {
            "fear_greed_index": 75  # Greed
        }
        
        features = analyzer.compute_features(sentiment_data)
        
        assert features["sentiment_score"] == 0.75
        assert features["sentiment_extreme"] == 0.5  # (75-50)/50
        assert features["sentiment_fear"] == 0
        assert features["sentiment_greed"] == 0.5  # (75-50)/50
    
    def test_compute_features_extreme_fear(self):
        """Test features during extreme fear."""
        analyzer = SentimentAnalyzer()
        
        sentiment_data = {
            "fear_greed_index": 10  # Extreme fear
        }
        
        features = analyzer.compute_features(sentiment_data)
        
        assert features["sentiment_score"] == 0.1
        assert features["sentiment_extreme"] == 0.8  # (50-10)/50
        assert features["sentiment_fear"] == 0.8  # (50-10)/50
        assert features["sentiment_greed"] == 0
    
    def test_compute_features_with_social_volume(self):
        """Test computing features with social volume."""
        analyzer = SentimentAnalyzer()
        
        # Add historical volume
        for i in range(10):
            analyzer.volume_history.append(100 + i * 10)
        
        sentiment_data = {
            "social_volume": 200  # High volume
        }
        
        features = analyzer.compute_features(sentiment_data)
        
        assert "social_volume_change" in features
        assert features["social_volume_spike"] == True  # 200 > 90th percentile
    
    def test_compute_features_social_volume_no_history(self):
        """Test social volume with no history."""
        analyzer = SentimentAnalyzer()
        
        sentiment_data = {
            "social_volume": 100
        }
        
        features = analyzer.compute_features(sentiment_data)
        
        assert features["social_volume_change"] == 0
        assert features["social_volume_spike"] == False
    
    def test_compute_features_with_news_sentiment(self):
        """Test computing features with news sentiment."""
        analyzer = SentimentAnalyzer()
        
        # Test positive news
        sentiment_data = {
            "news_sentiment": 0.8
        }
        
        features = analyzer.compute_features(sentiment_data)
        
        assert features["news_sentiment"] == 0.8
        assert features["news_bullish"] == 0.8
        assert features["news_bearish"] == 0
        
        # Test negative news
        sentiment_data = {
            "news_sentiment": -0.6
        }
        
        features = analyzer.compute_features(sentiment_data)
        
        assert features["news_sentiment"] == -0.6
        assert features["news_bullish"] == 0
        assert features["news_bearish"] == 0.6
    
    def test_sentiment_momentum(self):
        """Test sentiment momentum calculation."""
        analyzer = SentimentAnalyzer()
        
        # Add sentiment history
        for i in range(15):
            sentiment_data = {
                "fear_greed_index": 30 + i * 3  # Increasing sentiment
            }
            features = analyzer.compute_features(sentiment_data)
        
        # Check momentum
        assert "sentiment_momentum" in features
        assert features["sentiment_momentum"] > 0  # Positive momentum
        assert "sentiment_momentum_10" in features
    
    def test_composite_sentiment_score(self):
        """Test composite sentiment calculation."""
        analyzer = SentimentAnalyzer()
        
        sentiment_data = {
            "fear_greed_index": 60,  # Slightly greedy
            "news_sentiment": 0.5,   # Positive news
            "social_volume": 150
        }
        
        # Add some volume history
        for i in range(5):
            analyzer.volume_history.append(100)
        
        features = analyzer.compute_features(sentiment_data)
        
        assert "composite_sentiment" in features
        assert 0 <= features["composite_sentiment"] <= 1
    
    def test_analyze_crowd_behavior_fomo(self):
        """Test FOMO indicator calculation."""
        analyzer = SentimentAnalyzer()
        
        metrics = {
            "price_change_24h": 15,  # 15% increase
            "volume_change_24h": 200  # 200% volume increase
        }
        
        features = analyzer.analyze_crowd_behavior(metrics)
        
        assert features["fomo_score"] == 0.15 * 2.0  # High FOMO
    
    def test_analyze_crowd_behavior_panic(self):
        """Test panic indicator calculation."""
        analyzer = SentimentAnalyzer()
        
        metrics = {
            "price_change_1h": -5,  # 5% drop
            "volatility": 0.8
        }
        
        features = analyzer.analyze_crowd_behavior(metrics)
        
        assert features["panic_score"] == 0.05 * 0.8  # Moderate panic
    
    def test_analyze_crowd_behavior_herd(self):
        """Test herd behavior indicators."""
        analyzer = SentimentAnalyzer()
        
        # Test extreme long bias
        metrics = {
            "long_ratio": 0.9  # 90% longs
        }
        
        features = analyzer.analyze_crowd_behavior(metrics)
        
        assert features["herd_bias"] == 0.8  # abs(0.9-0.5)*2
        assert features["herd_direction"] == 1  # Long bias
        
        # Test extreme short bias
        metrics = {
            "long_ratio": 0.2  # 20% longs (80% shorts)
        }
        
        features = analyzer.analyze_crowd_behavior(metrics)
        
        assert features["herd_bias"] == 0.6  # abs(0.2-0.5)*2
        assert features["herd_direction"] == -1  # Short bias
    
    def test_get_sentiment_signals_no_history(self):
        """Test getting signals with no history."""
        analyzer = SentimentAnalyzer()
        
        signals = analyzer.get_sentiment_signals()
        
        assert signals["signal"] == 0
        assert signals["confidence"] == 0
    
    def test_get_sentiment_signals_extreme_fear(self):
        """Test signals during extreme fear."""
        analyzer = SentimentAnalyzer()
        
        # Add extreme fear readings
        analyzer.sentiment_history.append(0.3)
        analyzer.sentiment_history.append(0.15)  # Current: extreme fear
        
        signals = analyzer.get_sentiment_signals()
        
        assert signals["signal"] == 1  # Buy signal
        assert signals["confidence"] == 0.25  # (0.2-0.15)/0.2
        assert signals["sentiment_level"] == 0.15
    
    def test_get_sentiment_signals_extreme_greed(self):
        """Test signals during extreme greed."""
        analyzer = SentimentAnalyzer()
        
        # Add extreme greed readings
        analyzer.sentiment_history.append(0.7)
        analyzer.sentiment_history.append(0.9)  # Current: extreme greed
        
        signals = analyzer.get_sentiment_signals()
        
        assert signals["signal"] == -1  # Sell signal
        assert signals["confidence"] == 0.5  # (0.9-0.8)/0.2
        assert signals["sentiment_level"] == 0.9
    
    def test_get_sentiment_signals_bullish_reversal(self):
        """Test bullish reversal signal."""
        analyzer = SentimentAnalyzer()
        
        # Add history showing low sentiment then improvement
        for value in [0.25, 0.2, 0.25, 0.28, 0.3]:
            analyzer.sentiment_history.append(value)
        
        signals = analyzer.get_sentiment_signals()
        
        assert signals["signal"] == 0.5  # Moderate buy
        assert signals["confidence"] == 0.5
    
    def test_get_sentiment_signals_bearish_reversal(self):
        """Test bearish reversal signal."""
        analyzer = SentimentAnalyzer()
        
        # Add history showing high sentiment then decline
        for value in [0.75, 0.8, 0.75, 0.72, 0.7]:
            analyzer.sentiment_history.append(value)
        
        signals = analyzer.get_sentiment_signals()
        
        assert signals["signal"] == -0.5  # Moderate sell
        assert signals["confidence"] == 0.5
    
    def test_sentiment_history_window_limit(self):
        """Test that history respects window size."""
        analyzer = SentimentAnalyzer(window_size=5)
        
        # Add more than window size
        for i in range(10):
            sentiment_data = {"fear_greed_index": i * 10}
            analyzer.compute_features(sentiment_data)
        
        # Should only keep last 5
        assert len(analyzer.sentiment_history) == 5
        assert analyzer.sentiment_history[-1] == 0.9  # Last value
    
    def test_volume_history_window_limit(self):
        """Test that volume history respects window size."""
        analyzer = SentimentAnalyzer(window_size=5)
        
        # Add more than window size
        for i in range(10):
            sentiment_data = {"social_volume": i * 100}
            analyzer.compute_features(sentiment_data)
        
        # Should only keep last 5
        assert len(analyzer.volume_history) == 5
        assert analyzer.volume_history[-1] == 900  # Last value
    
    def test_empty_sentiment_data(self):
        """Test handling empty sentiment data."""
        analyzer = SentimentAnalyzer()
        
        features = analyzer.compute_features({})
        
        # Should return empty or minimal features
        assert isinstance(features, dict)
        assert "composite_sentiment" not in features  # No data for composite
    
    def test_partial_crowd_metrics(self):
        """Test crowd behavior with partial metrics."""
        analyzer = SentimentAnalyzer()
        
        # Only price change
        metrics = {"price_change_24h": 10}
        features = analyzer.analyze_crowd_behavior(metrics)
        assert "fomo_score" not in features  # Missing volume
        
        # Only volatility
        metrics = {"volatility": 0.5}
        features = analyzer.analyze_crowd_behavior(metrics)
        assert "panic_score" not in features  # Missing price change
    
    def test_neutral_sentiment(self):
        """Test neutral sentiment (50 on Fear & Greed)."""
        analyzer = SentimentAnalyzer()
        
        sentiment_data = {"fear_greed_index": 50}
        features = analyzer.compute_features(sentiment_data)
        
        assert features["sentiment_score"] == 0.5
        assert features["sentiment_extreme"] == 0  # No extreme
        assert features["sentiment_fear"] == 0
        assert features["sentiment_greed"] == 0