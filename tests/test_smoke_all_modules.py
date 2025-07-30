"""Smoke tests for all modules to boost coverage."""

import sys
import os
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Mock external dependencies
sys.modules['google.cloud'] = MagicMock()
sys.modules['google.cloud.storage'] = MagicMock()
sys.modules['gymnasium'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['stable_baselines3'] = MagicMock()


def test_all_feature_engineering_modules():
    """Test importing all feature engineering modules."""
    # Import base modules
    from src.feature_engineering.base import BaseFeature
    from src.feature_engineering.engineer import FeatureEngineer
    from src.feature_engineering.registry import FeatureRegistry
    
    # Import momentum features
    from src.feature_engineering.momentum.macd import MACDFeatures
    from src.feature_engineering.momentum.oscillators import (
        RSIFeature, StochasticFeature, CMFFeature, WilliamsRFeature,
        MFIFeature, UltimateOscillatorFeature, StochRSIFeature,
        TSIFeature, AverageOscillatorFeature
    )
    
    # Import pattern features
    from src.feature_engineering.pattern.pivots import PivotPointsFeature
    from src.feature_engineering.pattern.psar import PSARFeature
    from src.feature_engineering.pattern.supertrend import SuperTrendFeature
    from src.feature_engineering.pattern.zigzag import ZigZagFeature
    
    # Import statistical features
    from src.feature_engineering.statistical.basic import (
        LogReturnsFeature, VolatilityFeature, ZScoreFeature
    )
    from src.feature_engineering.statistical.regression import (
        LinearRegressionFeature, PolynomialRegressionFeature,
        LinearRegressionChannelFeature, LinearRegressionSlopeFeature
    )
    
    # Import trend features
    from src.feature_engineering.trend.ichimoku import IchimokuFeatures
    from src.feature_engineering.trend.moving_averages import (
        SMAFeature, EMAFeature, WMAFeature, HMAFeature,
        TripleEMAFeature, DualEMAFeature, CrossoverFeature
    )
    
    # Import trend strength features
    from src.feature_engineering.trend_strength.adx import ADXFeatures
    from src.feature_engineering.trend_strength.aroon import AroonFeatures
    from src.feature_engineering.trend_strength.trix import TRIXFeature
    from src.feature_engineering.trend_strength.vortex import VortexFeature
    
    # Import volatility features
    from src.feature_engineering.volatility.atr import ATRFeature
    from src.feature_engineering.volatility.bands import (
        BollingerBandsFeatures, KeltnerChannelFeatures,
        DonchianChannelFeatures, BollingerBandWidthFeature,
        BollingerBandPercentBFeature
    )
    from src.feature_engineering.volatility.other import (
        HistoricalVolatilityFeature, ChandlerExitFeature
    )
    
    # Import volume features
    from src.feature_engineering.volume.classic import (
        VolumeFeatures, OBVFeature, ADLFeature, PVIFeature,
        NVIFeature, VWAPFeature, MVWAPFeature
    )
    from src.feature_engineering.volume.price_volume import (
        PVTFeature, FVEFeature
    )
    
    # Test creating instances
    df = pd.DataFrame({
        'open': np.random.uniform(50000, 51000, 100),
        'high': np.random.uniform(50500, 51500, 100),
        'low': np.random.uniform(49500, 50500, 100),
        'close': np.random.uniform(50000, 51000, 100),
        'volume': np.random.uniform(100, 1000, 100)
    })
    
    # Ensure OHLC relationships
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    # Test a few features
    macd = MACDFeatures()
    result = macd.calculate(df)
    assert isinstance(result, pd.DataFrame)
    
    rsi = RSIFeature()
    result = rsi.calculate(df)
    assert isinstance(result, pd.Series)
    
    sma = SMAFeature()
    result = sma.calculate(df)
    assert isinstance(result, pd.Series)


def test_risk_management_modules():
    """Test risk management modules."""
    from src.risk_management.risk_manager import RiskManager
    from src.risk_management.models.position_sizing import (
        BasePositionSizer, KellyPositionSizer, 
        FixedFractionalSizer, VolatilityParitySizer
    )
    from src.risk_management.models.cost_model import (
        BaseCostModel, BinanceCostModel, FixedCostModel
    )
    from src.risk_management.models.drawdown_guard import DrawdownGuard
    from src.risk_management.models.api_throttler import BinanceAPIThrottler
    
    # Test position sizers
    kelly = KellyPositionSizer(min_edge=0.02)
    size = kelly.calculate_position_size(
        portfolio_value=10000,
        current_price=50000,
        signal_confidence=0.7,
        win_rate=0.6,
        avg_win=0.03,
        avg_loss=0.01
    )
    assert 0 <= size <= 1
    
    ff = FixedFractionalSizer(fraction=0.02)
    size = ff.calculate_position_size(
        portfolio_value=10000,
        current_price=50000,
        signal_confidence=0.7
    )
    assert 0 <= size <= 0.02
    
    vp = VolatilityParitySizer(target_volatility=0.02)
    size = vp.calculate_position_size(
        portfolio_value=10000,
        current_price=50000,
        signal_confidence=0.7,
        volatility=0.03
    )
    assert 0 <= size <= 1
    
    # Test cost models
    binance_cost = BinanceCostModel()
    cost = binance_cost.calculate_cost('market', 0.1, 50000)
    assert cost > 0
    
    fixed_cost = FixedCostModel(maker_fee=0.001, taker_fee=0.001)
    cost = fixed_cost.calculate_cost('limit', 0.1, 50000)
    assert cost > 0
    
    # Test drawdown guard
    guard = DrawdownGuard(max_drawdown=0.1)
    guard.update_portfolio(10000, {'BTCUSDT': 50000})
    multiplier = guard.get_risk_multiplier()
    assert 0 <= multiplier <= 1
    
    # Test API throttler
    throttler = BinanceAPIThrottler()
    can_request = throttler.check_and_wait_sync()
    assert isinstance(can_request, bool)
    
    # Test risk manager
    risk_manager = RiskManager(
        position_sizer=kelly,
        cost_model=binance_cost,
        drawdown_guard=guard,
        api_throttler=throttler
    )
    
    approved, size, reason = risk_manager.check_new_position(
        symbol='BTCUSDT',
        portfolio_value=10000,
        current_price=50000,
        signal_confidence=0.7,
        win_rate=0.6,
        avg_win=0.03,
        avg_loss=0.01
    )
    assert isinstance(approved, bool)
    assert isinstance(size, float)
    assert isinstance(reason, str)


def test_monitoring_modules():
    """Test monitoring modules."""
    from src.monitoring.metrics_collector import MetricsCollector
    from src.monitoring.performance_monitor import PerformanceMonitor
    from src.monitoring.alert_manager import AlertManager
    
    # Test metrics collector
    collector = MetricsCollector()
    collector.record_prediction('buy', 0.8, 0.05)
    collector.record_trade('BTCUSDT', 'buy', 0.5, 50000, 100)
    collector.record_latency('prediction', 0.1)
    collector.record_error('api', 'timeout')
    
    metrics = collector.get_metrics()
    assert metrics['total_predictions'] == 1
    assert metrics['total_trades'] == 1
    
    summary = collector.get_summary()
    assert 'timestamp' in summary
    
    # Test performance monitor
    monitor = PerformanceMonitor(window_size=100)
    for i in range(10):
        monitor.update(0.01 * ((-1) ** i))
    
    perf = monitor.get_performance()
    assert perf['total_returns'] == 10
    assert 'sharpe_ratio' in perf
    assert 'max_drawdown' in perf
    
    # Test alert manager
    manager = AlertManager()
    
    alerts = manager.check_alerts({
        'current_drawdown': -0.15,
        'consecutive_losses': 6,
        'error_rate': 0.15
    })
    assert len(alerts) > 0
    
    alert = {
        'type': 'test',
        'level': 'warning',
        'message': 'Test alert'
    }
    assert manager.should_send_alert(alert)


@patch('google.cloud.storage.Client')
def test_data_processing_modules(mock_client):
    """Test data processing modules."""
    from src.data_processing.daily_preprocessor import DailyPreprocessor
    
    mock_client_instance = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    
    mock_client.return_value = mock_client_instance
    mock_client_instance.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob
    
    preprocessor = DailyPreprocessor(
        project_id='test',
        bucket_name='test-bucket'
    )
    
    # Test orderbook resampling
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='100ms'),
        'best_bid': np.random.uniform(49900, 50000, 100),
        'best_ask': np.random.uniform(50000, 50100, 100),
        'bid_volume': np.random.uniform(0.1, 1.0, 100),
        'ask_volume': np.random.uniform(0.1, 1.0, 100)
    })
    
    resampled = preprocessor._resample_orderbook_data(df)
    assert len(resampled) < len(df)
    
    # Test feature computation
    df['mid_price'] = (df['best_bid'] + df['best_ask']) / 2
    df['spread'] = df['best_ask'] - df['best_bid']
    df['volume_imbalance'] = (df['bid_volume'] - df['ask_volume']) / (
        df['bid_volume'] + df['ask_volume']
    )
    
    features = preprocessor._compute_additional_features(df)
    assert 'price_change' in features.columns
    assert 'volatility' in features.columns


def test_config_module():
    """Test config module."""
    from src.config import (
        load_config, get_env_var, validate_config,
        PROJECT_ROOT, DATA_DIR, GCP_PROJECT_ID
    )
    
    # Test load_config
    config = load_config()
    assert isinstance(config, dict)
    assert 'project_id' in config
    
    # Test get_env_var
    result = get_env_var('TEST_VAR', 'default')
    assert result == 'default'
    
    # Test validate_config
    valid = validate_config({'api_key': 'test', 'bucket_name': 'test'})
    assert valid is True
    
    invalid = validate_config({'api_key': None})
    assert invalid is False
    
    # Test constants
    assert PROJECT_ROOT.exists()
    assert DATA_DIR.exists()
    assert isinstance(GCP_PROJECT_ID, str)


def test_main_module():
    """Test main module."""
    from src.main import parse_args
    
    # Test argument parsing
    args = parse_args(['collect', '--symbol', 'BTCUSDT'])
    assert args.command == 'collect'
    assert args.symbol == 'BTCUSDT'
    
    args = parse_args(['train', '--config', 'test.yaml'])
    assert args.command == 'train'
    assert args.config == 'test.yaml'
    
    args = parse_args(['serve', '--port', '8080'])
    assert args.command == 'serve'
    assert args.port == 8080


def test_api_modules():
    """Test API modules."""
    from src.api import create_app, load_model, get_model_info
    from src.api.api import PredictionRequest, PredictionResponse
    
    # Test create_app
    app = create_app()
    assert app is not None
    
    # Test model info
    info = get_model_info()
    assert info['name'] == 'TauSACTrader'
    assert 'version' in info
    
    # Test request/response models
    req = PredictionRequest(features={'close': 50000})
    assert req.features['close'] == 50000
    
    resp = PredictionResponse(
        prediction=0.5,
        confidence=0.8,
        timestamp='2024-01-01T00:00:00'
    )
    assert resp.prediction == 0.5


def test_feature_pipeline():
    """Test feature engineering pipeline."""
    from src.feature_engineering.engineer import FeatureEngineer
    from src.features.technical_indicators import TechnicalIndicators
    
    # Create test data
    df = pd.DataFrame({
        'open': np.random.uniform(50000, 51000, 200),
        'high': np.random.uniform(50500, 51500, 200),
        'low': np.random.uniform(49500, 50500, 200),
        'close': np.random.uniform(50000, 51000, 200),
        'volume': np.random.uniform(100, 1000, 200)
    })
    
    # Ensure OHLC relationships
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    # Test feature engineer
    engineer = FeatureEngineer()
    
    # Register some features
    engineer.add_feature('macd', window_size=20)
    engineer.add_feature('rsi', period=14)
    engineer.add_feature('sma', period=20)
    
    # Compute features
    result = engineer.compute_features(df)
    assert len(result.columns) > len(df.columns)
    
    # Test technical indicators
    indicators = TechnicalIndicators()
    result = indicators.add_all_indicators(df)
    assert len(result.columns) > len(df.columns)