"""Comprehensive test coverage for all modules."""

import sys
import os
from unittest.mock import Mock, patch, MagicMock
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))


class TestImports:
    """Test basic imports of all modules."""
    
    def test_import_config(self):
        """Test importing config module."""
        import src.config
        assert hasattr(src.config, 'load_config')
    
    def test_import_main(self):
        """Test importing main module."""
        import src.main
        assert hasattr(src.main, 'main')
    
    @patch('google.cloud.storage.Client')
    def test_import_data_collection(self, mock_client):
        """Test importing data collection modules."""
        # Mock the GCS client
        mock_client.return_value = MagicMock()
        
        # Import modules
        import src.data_collection.gcs_uploader
        import src.data_collection.integrated_collector
        
        assert hasattr(src.data_collection.gcs_uploader, 'GCSUploader')
        assert hasattr(src.data_collection.integrated_collector, 'IntegratedCollector')
    
    @patch('google.cloud.storage.Client')
    def test_import_data_processing(self, mock_client):
        """Test importing data processing modules."""
        import src.data_processing.daily_preprocessor
        assert hasattr(src.data_processing.daily_preprocessor, 'DailyPreprocessor')
    
    def test_import_feature_engineering(self):
        """Test importing feature engineering modules."""
        import src.feature_engineering.base
        import src.feature_engineering.engineer
        import src.feature_engineering.registry
        
        assert hasattr(src.feature_engineering.base, 'BaseFeature')
        assert hasattr(src.feature_engineering.engineer, 'FeatureEngineer')
        assert hasattr(src.feature_engineering.registry, 'FeatureRegistry')
    
    def test_import_feature_modules(self):
        """Test importing specific feature modules."""
        import src.feature_engineering.momentum.macd
        import src.feature_engineering.momentum.oscillators
        import src.feature_engineering.pattern.pivots
        import src.feature_engineering.pattern.psar
        import src.feature_engineering.pattern.supertrend
        import src.feature_engineering.pattern.zigzag
        import src.feature_engineering.statistical.basic
        import src.feature_engineering.statistical.regression
        import src.feature_engineering.trend.ichimoku
        import src.feature_engineering.trend.moving_averages
        import src.feature_engineering.trend_strength.adx
        import src.feature_engineering.trend_strength.aroon
        import src.feature_engineering.trend_strength.trix
        import src.feature_engineering.trend_strength.vortex
        import src.feature_engineering.volatility.atr
        import src.feature_engineering.volatility.bands
        import src.feature_engineering.volatility.other
        import src.feature_engineering.volume.classic
        import src.feature_engineering.volume.price_volume
        
        # Check one from each category
        assert hasattr(src.feature_engineering.momentum.macd, 'MACDFeatures')
        assert hasattr(src.feature_engineering.volatility.bands, 'BollingerBandsFeatures')
        assert hasattr(src.feature_engineering.volume.classic, 'VolumeFeatures')
    
    def test_import_risk_management(self):
        """Test importing risk management modules."""
        import src.risk_management.risk_manager
        import src.risk_management.models.position_sizing
        import src.risk_management.models.cost_model
        import src.risk_management.models.drawdown_guard
        import src.risk_management.models.api_throttler
        
        assert hasattr(src.risk_management.risk_manager, 'RiskManager')
        assert hasattr(src.risk_management.models.position_sizing, 'KellyPositionSizer')
        assert hasattr(src.risk_management.models.cost_model, 'BinanceCostModel')
        assert hasattr(src.risk_management.models.drawdown_guard, 'DrawdownGuard')
        assert hasattr(src.risk_management.models.api_throttler, 'BinanceAPIThrottler')
    
    def test_import_monitoring(self):
        """Test importing monitoring modules."""
        import src.monitoring.metrics_collector
        import src.monitoring.performance_monitor
        import src.monitoring.alert_manager
        
        assert hasattr(src.monitoring.metrics_collector, 'MetricsCollector')
        assert hasattr(src.monitoring.performance_monitor, 'PerformanceMonitor')
        assert hasattr(src.monitoring.alert_manager, 'AlertManager')
    
    @patch('gymnasium.make')
    @patch('torch.nn.Module')
    def test_import_rl_modules(self, mock_nn, mock_gym):
        """Test importing RL modules."""
        # Mock dependencies
        mock_gym.return_value = MagicMock()
        mock_nn.return_value = MagicMock()
        
        import src.rl.rewards
        assert hasattr(src.rl.rewards, 'RewardCalculator')
    
    def test_import_api_modules(self):
        """Test importing API modules."""
        import src.api
        import src.api.api
        
        assert hasattr(src.api, 'create_app')
        assert hasattr(src.api.api, 'create_app')


class TestModuleFunctionality:
    """Test basic functionality of modules."""
    
    def test_config_functionality(self):
        """Test config module functionality."""
        from src.config import load_config, get_env_var, validate_config
        
        # Test get_env_var with default
        result = get_env_var('NONEXISTENT_VAR', 'default')
        assert result == 'default'
        
        # Test validate_config
        valid_config = {
            'api_key': 'test',
            'bucket_name': 'test-bucket'
        }
        assert validate_config(valid_config) is True
        
        invalid_config = {}
        assert validate_config(invalid_config) is False
    
    def test_feature_registry(self):
        """Test feature registry functionality."""
        from src.feature_engineering.registry import FeatureRegistry
        
        registry = FeatureRegistry()
        
        # Test registration
        @registry.register('test_feature')
        class TestFeature:
            pass
        
        # Test retrieval
        assert registry.get('test_feature') == TestFeature
        assert 'test_feature' in registry.list_features()
    
    def test_risk_models(self):
        """Test risk management models."""
        from src.risk_management.models.position_sizing import (
            FixedFractionalSizer, VolatilityParitySizer
        )
        from src.risk_management.models.cost_model import FixedCostModel
        
        # Test fixed fractional
        ff_sizer = FixedFractionalSizer(fraction=0.02)
        size = ff_sizer.calculate_position_size(
            portfolio_value=10000,
            current_price=50000,
            signal_confidence=0.8
        )
        assert 0 <= size <= 0.02
        
        # Test volatility parity
        vp_sizer = VolatilityParitySizer(target_volatility=0.02)
        size = vp_sizer.calculate_position_size(
            portfolio_value=10000,
            current_price=50000,
            signal_confidence=0.8,
            volatility=0.03
        )
        assert 0 <= size <= 1
        
        # Test cost model
        cost_model = FixedCostModel(maker_fee=0.001, taker_fee=0.001)
        cost = cost_model.calculate_cost(
            order_type='market',
            size=0.1,
            price=50000
        )
        assert cost > 0
    
    def test_monitoring_functionality(self):
        """Test monitoring functionality."""
        from src.monitoring.metrics_collector import MetricsCollector
        from src.monitoring.performance_monitor import PerformanceMonitor
        
        # Test metrics collector
        collector = MetricsCollector()
        collector.record_prediction('buy', 0.8, 0.05)
        metrics = collector.get_metrics()
        assert metrics['total_predictions'] == 1
        
        # Test performance monitor
        monitor = PerformanceMonitor()
        monitor.update(0.01)
        perf = monitor.get_performance()
        assert perf['total_returns'] == 1
    
    @patch('google.cloud.storage.Client')
    def test_data_collection_basics(self, mock_client):
        """Test data collection basic functionality."""
        from src.data_collection.gcs_uploader import GCSUploader
        
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        uploader = GCSUploader(project_id='test', bucket_name='test-bucket')
        assert uploader.project_id == 'test'
        assert uploader.bucket_name == 'test-bucket'
    
    def test_feature_engineering_basics(self):
        """Test feature engineering basics."""
        from src.feature_engineering.base import BaseFeature
        import pandas as pd
        import numpy as np
        
        # Create a concrete implementation
        class TestFeature(BaseFeature):
            def calculate(self, df):
                return pd.Series(np.ones(len(df)), name='test_feature')
        
        # Test it
        feature = TestFeature(name='test')
        df = pd.DataFrame({'close': [1, 2, 3]})
        result = feature.calculate(df)
        assert len(result) == 3
        assert result.name == 'test_feature'
    
    def test_api_creation(self):
        """Test API creation."""
        from src.api import create_app
        
        app = create_app()
        assert app is not None
        assert hasattr(app, 'add_middleware')


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframes(self):
        """Test handling of empty dataframes."""
        from src.feature_engineering.base import BaseFeature
        import pandas as pd
        
        class TestFeature(BaseFeature):
            def calculate(self, df):
                if df.empty:
                    return pd.Series([], name='test_feature')
                return pd.Series([1] * len(df), name='test_feature')
        
        feature = TestFeature(name='test')
        empty_df = pd.DataFrame()
        result = feature.calculate(empty_df)
        assert len(result) == 0
    
    def test_invalid_config(self):
        """Test invalid configuration handling."""
        from src.config import validate_config
        
        # Various invalid configs
        assert validate_config(None) is False
        assert validate_config([]) is False
        assert validate_config('string') is False
        assert validate_config({'api_key': None}) is False
    
    def test_risk_edge_cases(self):
        """Test risk management edge cases."""
        from src.risk_management.models.position_sizing import KellyPositionSizer
        
        kelly = KellyPositionSizer(min_edge=0.02, kelly_fraction=0.25)
        
        # Test with no edge
        size = kelly.calculate_position_size(
            portfolio_value=10000,
            current_price=50000,
            signal_confidence=0.5,
            win_rate=0.5,
            avg_win=0.01,
            avg_loss=0.01
        )
        assert size == 0  # No edge, no position
        
        # Test with negative expectancy
        size = kelly.calculate_position_size(
            portfolio_value=10000,
            current_price=50000,
            signal_confidence=0.5,
            win_rate=0.3,
            avg_win=0.01,
            avg_loss=0.02
        )
        assert size == 0  # Negative expectancy