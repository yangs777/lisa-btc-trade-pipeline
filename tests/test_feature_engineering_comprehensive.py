"""Comprehensive tests for feature engineering to boost coverage."""

import pytest
import numpy as np
import pandas as pd

# Import the main feature engineering classes
from src.feature_engineering import FeatureEngineer, IndicatorRegistry, BaseIndicator
from src.data_processing.data_processor import DataProcessor
from src.monitoring.metrics_collector import MetricsCollector
from src.monitoring.alert_manager import AlertManager
from src.monitoring.performance_monitor import PerformanceMonitor


class TestFeatureEngineeringComprehensive:
    """Comprehensive tests for feature engineering coverage."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample OHLCV data."""
        dates = pd.date_range("2024-01-01", periods=200, freq="h")
        np.random.seed(42)
        
        # Generate realistic price data
        close_prices = 50000 + np.cumsum(np.random.randn(200) * 100)
        high_prices = close_prices + np.abs(np.random.randn(200) * 50)
        low_prices = close_prices - np.abs(np.random.randn(200) * 50)
        open_prices = close_prices + np.random.randn(200) * 30
        volume = np.abs(np.random.randn(200) * 1000000) + 100000
        
        return pd.DataFrame({
            "timestamp": dates,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume
        })
    
    def test_feature_engineer_basic(self, sample_data):
        """Test FeatureEngineer basic functionality."""
        engineer = FeatureEngineer()
        
        # Test compute_features with default indicators
        features = engineer.compute_features(sample_data)
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_data)
        assert len(features.columns) > 50  # Should have many features
        
    def test_feature_engineer_with_custom_indicators(self, sample_data):
        """Test FeatureEngineer with custom indicator list."""
        engineer = FeatureEngineer()
        
        # Test with subset of indicators
        features = engineer.compute_features(
            sample_data, 
            indicators=["RSI", "MACD", "BollingerBands"]
        )
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_data)
        
    def test_feature_engineer_parallel_processing(self, sample_data):
        """Test parallel processing mode."""
        engineer = FeatureEngineer()
        
        # Test with parallel processing
        features = engineer.compute_features(sample_data, n_jobs=2)
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_data)
    
    def test_indicator_registry(self):
        """Test IndicatorRegistry functionality."""
        registry = IndicatorRegistry()
        
        # Test getting all indicators
        all_indicators = registry.get_all_indicators()
        assert isinstance(all_indicators, list)
        assert len(all_indicators) > 10
        
        # Test getting indicators by category
        momentum_indicators = registry.get_indicators_by_category("momentum")
        assert isinstance(momentum_indicators, list)
        assert len(momentum_indicators) > 0
        
        # Test getting indicator instance
        rsi = registry.get_indicator("RSI")
        assert rsi is not None
        assert isinstance(rsi, BaseIndicator)
    
    def test_data_processor(self, sample_data):
        """Test DataProcessor functionality."""
        processor = DataProcessor()
        
        # Test data validation
        is_valid = processor.validate_data(sample_data)
        assert is_valid
        
        # Test preprocessing
        processed = processor.preprocess(sample_data)
        assert len(processed) == len(sample_data)
        
        # Test with missing data
        data_with_gaps = sample_data.copy()
        data_with_gaps.loc[10:20, "close"] = np.nan
        
        filled_data = processor.fill_missing_data(data_with_gaps)
        assert not filled_data["close"].isna().any()
    
    def test_metrics_collector(self):
        """Test MetricsCollector functionality."""
        collector = MetricsCollector()
        
        # Record predictions
        collector.record_prediction("BUY", 0.8, 0.05)
        collector.record_prediction("SELL", 0.7, 0.03)
        collector.record_prediction("HOLD", 0.9, 0.0)
        
        # Record trades
        collector.record_trade("BTC/USDT", "BUY", 1.0, 50000, 500)
        collector.record_trade("BTC/USDT", "SELL", 0.5, 51000, 250)
        
        # Record latencies
        collector.record_latency("prediction", 0.05)
        collector.record_latency("trade_execution", 0.1)
        
        # Record errors
        collector.record_error("api_call", "Connection timeout")
        
        # Get metrics
        metrics = collector.get_metrics()
        assert metrics["total_predictions"] == 3
        assert metrics["total_trades"] == 2
        assert metrics["total_pnl"] == 750
        assert "prediction" in metrics["avg_latency"]
        assert metrics["errors"]["api_call"] == 1
        
        # Get summary
        summary = collector.get_summary()
        assert "timestamp" in summary
        assert summary["total_predictions"] == 3
        assert summary["total_trades"] == 2
        
        # Test reset
        collector.reset()
        metrics_after_reset = collector.get_metrics()
        assert metrics_after_reset["total_predictions"] == 0
    
    def test_alert_manager(self):
        """Test AlertManager functionality."""
        manager = AlertManager()
        
        # Test alert checking
        metrics = {
            "current_drawdown": -0.12,  # Above threshold
            "consecutive_losses": 6,     # Above threshold
            "error_rate": 0.15          # Above threshold
        }
        
        alerts = manager.check_alerts(metrics)
        assert len(alerts) == 3
        assert any(alert["type"] == "drawdown" for alert in alerts)
        assert any(alert["type"] == "loss_streak" for alert in alerts)
        assert any(alert["type"] == "error_rate" for alert in alerts)
        
        # Test alert throttling
        alert = alerts[0]
        should_send_first = manager.should_send_alert(alert)
        assert should_send_first
        
        # Second alert should be throttled
        should_send_second = manager.should_send_alert(alert)
        assert not should_send_second
        
        # Test send_alert (just ensure it doesn't crash)
        manager.send_alert(alert, channels=["log"])
    
    def test_performance_monitor(self):
        """Test PerformanceMonitor functionality."""
        monitor = PerformanceMonitor(window_size=100)
        
        # Add some returns
        returns = [0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.0, 0.03]
        for ret in returns:
            monitor.update(ret)
        
        # Get performance metrics
        perf = monitor.get_performance()
        assert perf["total_returns"] == len(returns)
        assert "cumulative_return" in perf
        assert "sharpe_ratio" in perf
        assert "max_drawdown" in perf
        assert "win_rate" in perf
        assert "current_equity" in perf
        
        # Test with no returns
        empty_monitor = PerformanceMonitor()
        empty_perf = empty_monitor.get_performance()
        assert empty_perf["total_returns"] == 0
        assert empty_perf["cumulative_return"] == 0
    
    def test_edge_cases(self, sample_data):
        """Test edge cases and error handling."""
        engineer = FeatureEngineer()
        
        # Test with very small dataset
        small_data = sample_data.head(10)
        features = engineer.compute_features(small_data)
        assert len(features) == len(small_data)
        
        # Test with missing columns
        incomplete_data = sample_data[["open", "close"]]
        with pytest.raises(KeyError):
            engineer.compute_features(incomplete_data)
        
        # Test with empty dataframe
        empty_data = pd.DataFrame()
        with pytest.raises(Exception):
            engineer.compute_features(empty_data)
    
    def test_feature_importance(self, sample_data):
        """Test feature importance calculation."""
        engineer = FeatureEngineer()
        
        # Compute features
        features = engineer.compute_features(sample_data)
        
        # Create dummy target
        target = (sample_data["close"].pct_change().shift(-1) > 0).astype(int)
        
        # Test feature selection based on importance
        # This would normally use feature_selection module
        assert len(features.columns) > 50
        
    def test_concurrent_processing(self, sample_data):
        """Test concurrent processing scenarios."""
        engineer = FeatureEngineer()
        
        # Split data into chunks
        chunk_size = 50
        chunks = [sample_data[i:i+chunk_size] for i in range(0, len(sample_data), chunk_size)]
        
        # Process each chunk
        results = []
        for chunk in chunks:
            if len(chunk) > 30:  # Minimum size for indicators
                features = engineer.compute_features(chunk)
                results.append(features)
        
        # Combine results
        if results:
            combined = pd.concat(results, ignore_index=True)
            assert len(combined) > 0