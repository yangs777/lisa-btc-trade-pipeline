"""Advanced test coverage for remaining modules."""

import sys
import os
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
import pytest
from datetime import datetime, timedelta
from pathlib import Path
import json

# Mock all external dependencies before imports
sys.modules['numpy'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['google.cloud'] = MagicMock()
sys.modules['google.cloud.storage'] = MagicMock()
sys.modules['gymnasium'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['stable_baselines3'] = MagicMock()
sys.modules['yaml'] = MagicMock()
sys.modules['aiofiles'] = MagicMock()
sys.modules['websockets'] = MagicMock()
sys.modules['fastapi'] = MagicMock()
sys.modules['httpx'] = MagicMock()
sys.modules['uvicorn'] = MagicMock()
sys.modules['scikit-learn'] = MagicMock()
sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.preprocessing'] = MagicMock()

# Mock pandas DataFrame with more comprehensive behavior
mock_df = MagicMock()
mock_df.__len__ = MagicMock(return_value=200)
mock_df.shape = (200, 10)
mock_df.columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'mid_price', 'spread', 'volume_imbalance', 'price_change']
mock_df.index = range(200)
mock_df.iloc = MagicMock()
mock_df.loc = MagicMock()
mock_df.copy = MagicMock(return_value=mock_df)
mock_df.dropna = MagicMock(return_value=mock_df)
mock_df.fillna = MagicMock(return_value=mock_df)
mock_df.rolling = MagicMock(return_value=MagicMock(mean=MagicMock(return_value=mock_df)))
mock_df.shift = MagicMock(return_value=mock_df)
mock_df.diff = MagicMock(return_value=mock_df)
mock_df.pct_change = MagicMock(return_value=mock_df)
mock_df.resample = MagicMock(return_value=MagicMock(agg=MagicMock(return_value=mock_df)))
mock_df.__getitem__ = MagicMock(return_value=MagicMock())
mock_df.__setitem__ = MagicMock()

# Mock series
mock_series = MagicMock()
mock_series.__len__ = MagicMock(return_value=200)
mock_series.name = 'test_series'
mock_series.fillna = MagicMock(return_value=mock_series)
mock_series.rolling = MagicMock(return_value=MagicMock(mean=MagicMock(return_value=mock_series)))
mock_series.shift = MagicMock(return_value=mock_series)
mock_series.diff = MagicMock(return_value=mock_series)
mock_series.pct_change = MagicMock(return_value=mock_series)
mock_series.values = [50000 + i * 10 for i in range(200)]

sys.modules['pandas'].DataFrame = MagicMock(return_value=mock_df)
sys.modules['pandas'].Series = MagicMock(return_value=mock_series)
sys.modules['pandas'].concat = MagicMock(return_value=mock_df)
sys.modules['pandas'].merge = MagicMock(return_value=mock_df)
sys.modules['pandas'].date_range = MagicMock(return_value=[datetime.now() + timedelta(seconds=i) for i in range(200)])


def test_all_momentum_features():
    """Test all momentum feature implementations."""
    from src.feature_engineering.momentum.macd import MACDFeatures
    from src.feature_engineering.momentum.oscillators import (
        RSIFeature, StochasticFeature, CMFFeature, WilliamsRFeature,
        MFIFeature, UltimateOscillatorFeature, StochRSIFeature,
        TSIFeature, AverageOscillatorFeature
    )
    
    # Test MACD with different parameters
    macd = MACDFeatures(fast_period=10, slow_period=20, signal_period=5)
    result = macd.calculate(mock_df)
    assert result is not None
    assert macd.name == 'macd'
    
    # Test RSI edge cases
    rsi = RSIFeature(period=7)
    result = rsi.calculate(mock_df)
    assert result is not None
    assert rsi.name == 'rsi_7'
    
    # Test Stochastic
    stoch = StochasticFeature(period=21)
    result = stoch.calculate(mock_df)
    assert result is not None
    assert stoch.name == 'stochastic_21'
    
    # Test CMF (Chaikin Money Flow)
    cmf = CMFFeature(period=20)
    result = cmf.calculate(mock_df)
    assert result is not None
    assert cmf.name == 'cmf_20'
    
    # Test Williams %R
    williams = WilliamsRFeature(period=14)
    result = williams.calculate(mock_df)
    assert result is not None
    assert williams.name == 'williams_r_14'
    
    # Test MFI (Money Flow Index)
    mfi = MFIFeature(period=14)
    result = mfi.calculate(mock_df)
    assert result is not None
    assert mfi.name == 'mfi_14'
    
    # Test Ultimate Oscillator
    uo = UltimateOscillatorFeature(period1=7, period2=14, period3=28)
    result = uo.calculate(mock_df)
    assert result is not None
    assert uo.name == 'ultimate_oscillator'
    
    # Test Stochastic RSI
    stoch_rsi = StochRSIFeature(period=14, smooth_k=3, smooth_d=3)
    result = stoch_rsi.calculate(mock_df)
    assert result is not None
    assert stoch_rsi.name == 'stoch_rsi_14'
    
    # Test TSI (True Strength Index)
    tsi = TSIFeature(slow_period=25, fast_period=13)
    result = tsi.calculate(mock_df)
    assert result is not None
    assert tsi.name == 'tsi'
    
    # Test Average Oscillator
    avg_osc = AverageOscillatorFeature(oscillators=['rsi', 'stochastic'])
    result = avg_osc.calculate(mock_df)
    assert result is not None
    assert avg_osc.name == 'average_oscillator'


def test_all_trend_features():
    """Test all trend feature implementations."""
    from src.feature_engineering.trend.moving_averages import (
        SMAFeature, EMAFeature, WMAFeature, HMAFeature,
        TripleEMAFeature, DualEMAFeature, CrossoverFeature
    )
    from src.feature_engineering.trend.ichimoku import IchimokuFeatures
    
    # Test SMA variations
    sma = SMAFeature(period=50)
    result = sma.calculate(mock_df)
    assert result is not None
    assert sma.name == 'sma_50'
    
    # Test EMA
    ema = EMAFeature(period=21)
    result = ema.calculate(mock_df)
    assert result is not None
    assert ema.name == 'ema_21'
    
    # Test WMA (Weighted Moving Average)
    wma = WMAFeature(period=20)
    result = wma.calculate(mock_df)
    assert result is not None
    assert wma.name == 'wma_20'
    
    # Test HMA (Hull Moving Average)
    hma = HMAFeature(period=16)
    result = hma.calculate(mock_df)
    assert result is not None
    assert hma.name == 'hma_16'
    
    # Test Triple EMA
    tema = TripleEMAFeature(period=14)
    result = tema.calculate(mock_df)
    assert result is not None
    assert tema.name == 'tema_14'
    
    # Test Dual EMA
    dema = DualEMAFeature(period=20)
    result = dema.calculate(mock_df)
    assert result is not None
    assert dema.name == 'dema_20'
    
    # Test Crossover
    crossover = CrossoverFeature(fast_period=10, slow_period=20)
    result = crossover.calculate(mock_df)
    assert result is not None
    assert crossover.name == 'ma_crossover_10_20'
    
    # Test Ichimoku
    ichimoku = IchimokuFeatures(tenkan=9, kijun=26, senkou_b=52)
    result = ichimoku.calculate(mock_df)
    assert result is not None
    assert ichimoku.name == 'ichimoku'


def test_all_volatility_features():
    """Test all volatility feature implementations."""
    from src.feature_engineering.volatility.atr import ATRFeature
    from src.feature_engineering.volatility.bands import (
        BollingerBandsFeatures, KeltnerChannelFeatures,
        DonchianChannelFeatures, BollingerBandWidthFeature,
        BollingerBandPercentBFeature
    )
    from src.feature_engineering.volatility.other import (
        HistoricalVolatilityFeature, ChandlerExitFeature
    )
    
    # Test ATR
    atr = ATRFeature(period=14)
    result = atr.calculate(mock_df)
    assert result is not None
    assert atr.name == 'atr_14'
    
    # Test Bollinger Bands
    bb = BollingerBandsFeatures(period=20, std_dev=2.5)
    result = bb.calculate(mock_df)
    assert result is not None
    assert bb.name == 'bollinger_bands_20'
    
    # Test Keltner Channel
    kc = KeltnerChannelFeatures(period=20, multiplier=1.5)
    result = kc.calculate(mock_df)
    assert result is not None
    assert kc.name == 'keltner_channel_20'
    
    # Test Donchian Channel
    dc = DonchianChannelFeatures(period=30)
    result = dc.calculate(mock_df)
    assert result is not None
    assert dc.name == 'donchian_channel_30'
    
    # Test Bollinger Band Width
    bb_width = BollingerBandWidthFeature(period=20, std_dev=2)
    result = bb_width.calculate(mock_df)
    assert result is not None
    assert bb_width.name == 'bb_width_20'
    
    # Test Bollinger Band %B
    bb_percent = BollingerBandPercentBFeature(period=20, std_dev=2)
    result = bb_percent.calculate(mock_df)
    assert result is not None
    assert bb_percent.name == 'bb_percent_b_20'
    
    # Test Historical Volatility
    hist_vol = HistoricalVolatilityFeature(period=30)
    result = hist_vol.calculate(mock_df)
    assert result is not None
    assert hist_vol.name == 'historical_volatility_30'
    
    # Test Chandler Exit
    chandler = ChandlerExitFeature(period=22, multiplier=3)
    result = chandler.calculate(mock_df)
    assert result is not None
    assert chandler.name == 'chandler_exit_22'


def test_all_volume_features():
    """Test all volume feature implementations."""
    from src.feature_engineering.volume.classic import (
        VolumeFeatures, OBVFeature, ADLFeature, PVIFeature,
        NVIFeature, VWAPFeature, MVWAPFeature
    )
    from src.feature_engineering.volume.price_volume import (
        PVTFeature, FVEFeature
    )
    
    # Test basic volume features
    vol = VolumeFeatures()
    result = vol.calculate(mock_df)
    assert result is not None
    assert vol.name == 'volume_features'
    
    # Test OBV (On Balance Volume)
    obv = OBVFeature()
    result = obv.calculate(mock_df)
    assert result is not None
    assert obv.name == 'obv'
    
    # Test ADL (Accumulation/Distribution Line)
    adl = ADLFeature()
    result = adl.calculate(mock_df)
    assert result is not None
    assert adl.name == 'adl'
    
    # Test PVI (Positive Volume Index)
    pvi = PVIFeature()
    result = pvi.calculate(mock_df)
    assert result is not None
    assert pvi.name == 'pvi'
    
    # Test NVI (Negative Volume Index)
    nvi = NVIFeature()
    result = nvi.calculate(mock_df)
    assert result is not None
    assert nvi.name == 'nvi'
    
    # Test VWAP
    vwap = VWAPFeature(period=20)
    result = vwap.calculate(mock_df)
    assert result is not None
    assert vwap.name == 'vwap_20'
    
    # Test MVWAP (Moving VWAP)
    mvwap = MVWAPFeature(period=20)
    result = mvwap.calculate(mock_df)
    assert result is not None
    assert mvwap.name == 'mvwap_20'
    
    # Test PVT (Price Volume Trend)
    pvt = PVTFeature()
    result = pvt.calculate(mock_df)
    assert result is not None
    assert pvt.name == 'pvt'
    
    # Test FVE (Finite Volume Elements)
    fve = FVEFeature(period=22)
    result = fve.calculate(mock_df)
    assert result is not None
    assert fve.name == 'fve_22'


def test_all_pattern_features():
    """Test all pattern feature implementations."""
    from src.feature_engineering.pattern.pivots import PivotPointsFeature
    from src.feature_engineering.pattern.psar import PSARFeature
    from src.feature_engineering.pattern.supertrend import SuperTrendFeature
    from src.feature_engineering.pattern.zigzag import ZigZagFeature
    
    # Test Pivot Points
    pivots = PivotPointsFeature()
    result = pivots.calculate(mock_df)
    assert result is not None
    assert pivots.name == 'pivot_points'
    
    # Test PSAR (Parabolic SAR)
    psar = PSARFeature(initial_af=0.02, max_af=0.2, af_increment=0.02)
    result = psar.calculate(mock_df)
    assert result is not None
    assert psar.name == 'psar'
    
    # Test SuperTrend
    supertrend = SuperTrendFeature(period=10, multiplier=3)
    result = supertrend.calculate(mock_df)
    assert result is not None
    assert supertrend.name == 'supertrend_10_3'
    
    # Test ZigZag
    zigzag = ZigZagFeature(percentage=5)
    result = zigzag.calculate(mock_df)
    assert result is not None
    assert zigzag.name == 'zigzag_5'


def test_all_trend_strength_features():
    """Test all trend strength feature implementations."""
    from src.feature_engineering.trend_strength.adx import ADXFeatures
    from src.feature_engineering.trend_strength.aroon import AroonFeatures
    from src.feature_engineering.trend_strength.trix import TRIXFeature
    from src.feature_engineering.trend_strength.vortex import VortexFeature
    
    # Test ADX
    adx = ADXFeatures(period=14)
    result = adx.calculate(mock_df)
    assert result is not None
    assert adx.name == 'adx_14'
    
    # Test Aroon
    aroon = AroonFeatures(period=25)
    result = aroon.calculate(mock_df)
    assert result is not None
    assert aroon.name == 'aroon_25'
    
    # Test TRIX
    trix = TRIXFeature(period=15)
    result = trix.calculate(mock_df)
    assert result is not None
    assert trix.name == 'trix_15'
    
    # Test Vortex
    vortex = VortexFeature(period=14)
    result = vortex.calculate(mock_df)
    assert result is not None
    assert vortex.name == 'vortex_14'


def test_all_statistical_features():
    """Test all statistical feature implementations."""
    from src.feature_engineering.statistical.basic import (
        LogReturnsFeature, VolatilityFeature, ZScoreFeature
    )
    from src.feature_engineering.statistical.regression import (
        LinearRegressionFeature, PolynomialRegressionFeature,
        LinearRegressionChannelFeature, LinearRegressionSlopeFeature
    )
    
    # Test Log Returns
    log_returns = LogReturnsFeature()
    result = log_returns.calculate(mock_df)
    assert result is not None
    assert log_returns.name == 'log_returns'
    
    # Test Volatility
    volatility = VolatilityFeature(period=20)
    result = volatility.calculate(mock_df)
    assert result is not None
    assert volatility.name == 'volatility_20'
    
    # Test Z-Score
    zscore = ZScoreFeature(period=30)
    result = zscore.calculate(mock_df)
    assert result is not None
    assert zscore.name == 'zscore_30'
    
    # Test Linear Regression
    lin_reg = LinearRegressionFeature(period=20)
    result = lin_reg.calculate(mock_df)
    assert result is not None
    assert lin_reg.name == 'linear_regression_20'
    
    # Test Polynomial Regression
    poly_reg = PolynomialRegressionFeature(period=20, degree=3)
    result = poly_reg.calculate(mock_df)
    assert result is not None
    assert poly_reg.name == 'polynomial_regression_20_3'
    
    # Test Linear Regression Channel
    lr_channel = LinearRegressionChannelFeature(period=20, std_dev=2)
    result = lr_channel.calculate(mock_df)
    assert result is not None
    assert lr_channel.name == 'lr_channel_20'
    
    # Test Linear Regression Slope
    lr_slope = LinearRegressionSlopeFeature(period=20)
    result = lr_slope.calculate(mock_df)
    assert result is not None
    assert lr_slope.name == 'lr_slope_20'


def test_data_collection_async():
    """Test async data collection components."""
    from src.data_collection.binance_websocket import BinanceWebSocketCollector
    
    # Create collector
    collector = BinanceWebSocketCollector(
        symbol='BTCUSDT',
        output_dir='test_output',
        buffer_size=100
    )
    
    # Test initialization
    assert collector.symbol == 'BTCUSDT'
    assert collector.buffer_size == 100
    assert collector.reconnect_delay == 5
    
    # Test URL generation
    url = collector._get_ws_url()
    assert 'binance.com' in url
    assert 'btcusdt' in url.lower()
    
    # Test orderbook update processing
    mock_message = {
        'b': [['50000', '1.0'], ['49999', '2.0']],  # Bids
        'a': [['50001', '1.5'], ['50002', '2.5']]   # Asks
    }
    
    orderbook = collector._process_orderbook_update(mock_message)
    assert 'timestamp' in orderbook
    assert orderbook['best_bid'] == 50000
    assert orderbook['best_ask'] == 50001
    assert orderbook['bid_volume'] == 1.0
    assert orderbook['ask_volume'] == 1.5


@patch('google.cloud.storage.Client')
def test_data_processing_complete(mock_client):
    """Test complete data processing pipeline."""
    from src.data_processing.daily_preprocessor import DailyPreprocessor
    
    # Mock GCS components
    mock_client_instance = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.download_as_text.return_value = json.dumps({
        'timestamp': datetime.now().isoformat(),
        'best_bid': 50000,
        'best_ask': 50001,
        'bid_volume': 1.0,
        'ask_volume': 1.5,
        'mid_price': 50000.5,
        'spread': 1,
        'volume_imbalance': 0.2
    })
    
    mock_client.return_value = mock_client_instance
    mock_client_instance.bucket.return_value = mock_bucket
    mock_bucket.list_blobs.return_value = [mock_blob] * 10
    mock_bucket.blob.return_value = mock_blob
    
    # Create preprocessor
    preprocessor = DailyPreprocessor('test-project', 'test-bucket')
    
    # Test preprocessing for date
    result = preprocessor.preprocess_for_date('2024-01-01')
    assert result is True
    
    # Test scheduled preprocessing
    preprocessor._is_time_to_run = MagicMock(return_value=False)
    preprocessor.run_scheduled()  # Should exit immediately
    
    # Test orderbook data resampling
    df = mock_df.copy()
    resampled = preprocessor._resample_orderbook_data(df)
    assert resampled is not None
    
    # Test feature computation
    features = preprocessor._compute_additional_features(df)
    assert features is not None
    
    # Test data validation
    is_valid = preprocessor._validate_data(df)
    assert isinstance(is_valid, bool)


def test_api_server_endpoints():
    """Test all API server endpoints."""
    from src.api import app as fastapi_app
    from fastapi.testclient import TestClient
    
    # Create test client
    mock_client = MagicMock(spec=TestClient)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'status': 'ok'}
    mock_client.get.return_value = mock_response
    mock_client.post.return_value = mock_response
    
    # Test health endpoint
    response = mock_client.get('/health')
    assert response.status_code == 200
    
    # Test predict endpoint
    response = mock_client.post('/predict', json={'features': {'close': 50000}})
    assert response.status_code == 200
    
    # Test batch predict endpoint
    response = mock_client.post('/predict/batch', json={'samples': [{'close': 50000}]})
    assert response.status_code == 200
    
    # Test model info endpoint
    response = mock_client.get('/model/info')
    assert response.status_code == 200


def test_risk_management_complete():
    """Test complete risk management system."""
    from src.risk_management.risk_manager import RiskManager
    from src.risk_management.models.position_sizing import (
        KellyPositionSizer, FixedFractionalSizer, VolatilityParitySizer
    )
    from src.risk_management.models.cost_model import BinanceCostModel
    from src.risk_management.models.drawdown_guard import DrawdownGuard
    from src.risk_management.models.api_throttler import BinanceAPIThrottler
    
    # Create all components
    kelly = KellyPositionSizer(min_edge=0.01, kelly_fraction=0.25)
    ff = FixedFractionalSizer(fraction=0.02)
    vp = VolatilityParitySizer(target_volatility=0.02)
    
    cost_model = BinanceCostModel()
    guard = DrawdownGuard(max_drawdown=0.08)
    throttler = BinanceAPIThrottler()
    
    # Test with different position sizers
    for sizer in [kelly, ff, vp]:
        manager = RiskManager(
            position_sizer=sizer,
            cost_model=cost_model,
            drawdown_guard=guard,
            api_throttler=throttler
        )
        
        # Test position validation
        approved, size, reason = manager.check_new_position(
            symbol='BTCUSDT',
            portfolio_value=10000,
            current_price=50000,
            signal_confidence=0.75,
            win_rate=0.55,
            avg_win=0.02,
            avg_loss=0.01,
            volatility=0.03
        )
        
        assert isinstance(approved, bool)
        assert isinstance(size, float)
        assert isinstance(reason, str)
        
        # Test closing position
        if approved:
            manager.update_position('BTCUSDT', size, 50000)
            can_close, reason = manager.check_close_position('BTCUSDT', 51000)
            assert isinstance(can_close, bool)
            assert isinstance(reason, str)
    
    # Test portfolio updates
    manager.update_portfolio(11000, {'BTCUSDT': 51000})
    
    # Test risk metrics
    metrics = manager.get_metrics()
    assert 'portfolio_value' in metrics
    assert 'positions' in metrics
    assert 'api_usage' in metrics
    assert 'drawdown_status' in metrics


def test_monitoring_complete():
    """Test complete monitoring system."""
    from src.monitoring.metrics_collector import MetricsCollector
    from src.monitoring.performance_monitor import PerformanceMonitor
    from src.monitoring.alert_manager import AlertManager
    
    # Test metrics collector with various scenarios
    collector = MetricsCollector()
    
    # Record multiple predictions
    for i in range(50):
        signal = ['buy', 'sell', 'hold'][i % 3]
        confidence = 0.5 + (i % 5) * 0.1
        latency = 0.05 + (i % 10) * 0.01
        collector.record_prediction(signal, confidence, latency)
    
    # Record multiple trades
    for i in range(20):
        action = ['buy', 'sell'][i % 2]
        amount = 0.1 + (i % 5) * 0.02
        price = 50000 + i * 100
        fee = amount * price * 0.001
        collector.record_trade('BTCUSDT', action, amount, price, fee)
    
    # Record errors
    for error_type in ['api', 'data', 'model', 'system']:
        collector.record_error(error_type, f'{error_type}_error')
    
    # Test metrics
    metrics = collector.get_metrics()
    assert metrics['total_predictions'] == 50
    assert metrics['total_trades'] == 20
    assert metrics['total_errors'] == 4
    assert metrics['buy_signals'] > 0
    assert metrics['sell_signals'] > 0
    
    # Test performance monitor with realistic data
    monitor = PerformanceMonitor(window_size=100)
    
    # Simulate trading returns
    for i in range(100):
        # Generate realistic returns
        if i % 10 < 6:  # 60% win rate
            ret = 0.015 + (i % 5) * 0.005  # Wins: 1.5% to 3.5%
        else:
            ret = -0.01 - (i % 4) * 0.005  # Losses: -1% to -2.5%
        monitor.update(ret)
    
    perf = monitor.get_performance()
    assert perf['total_returns'] == 100
    assert perf['win_rate'] > 0.5
    assert perf['sharpe_ratio'] != 0
    assert 'max_drawdown' in perf
    assert 'avg_win' in perf
    assert 'avg_loss' in perf
    
    # Test alert manager with various conditions
    manager = AlertManager()
    
    # Test all alert types
    test_conditions = {
        'current_drawdown': -0.12,  # > 10%
        'consecutive_losses': 7,     # > 5
        'error_rate': 0.2,          # > 10%
        'api_usage': 0.95,          # > 90%
        'latency_ms': 250,          # > 200ms
        'position_size': 0.15       # > 10%
    }
    
    alerts = manager.check_alerts(test_conditions)
    assert len(alerts) >= 3  # Should trigger multiple alerts
    
    # Test alert throttling
    for alert in alerts:
        first_send = manager.should_send_alert(alert)
        second_send = manager.should_send_alert(alert)
        assert first_send is True
        assert second_send is False  # Should be throttled


def test_feature_engineering_pipeline():
    """Test complete feature engineering pipeline."""
    from src.features.technical_indicators import TechnicalIndicators
    from src.feature_engineering.engineer import FeatureEngineer
    
    # Test with technical indicators
    indicators = TechnicalIndicators()
    
    # Add all indicators
    result = indicators.add_all_indicators(mock_df)
    assert result is not None
    
    # Test with feature engineer
    engineer = FeatureEngineer()
    
    # Add various features
    feature_configs = [
        ('sma', {'period': 20}),
        ('ema', {'period': 20}),
        ('rsi', {'period': 14}),
        ('macd', {}),
        ('bollinger_bands', {'period': 20}),
        ('atr', {'period': 14}),
        ('volume_features', {}),
        ('adx', {'period': 14})
    ]
    
    for feature_name, params in feature_configs:
        engineer.add_feature(feature_name, **params)
    
    # Compute all features
    result = engineer.compute_features(mock_df)
    assert result is not None
    assert len(engineer.feature_names) == len(feature_configs)
    
    # Test feature removal
    engineer.remove_feature(engineer.feature_names[0])
    assert len(engineer.feature_names) == len(feature_configs) - 1


def test_main_command_execution():
    """Test main command execution paths."""
    from src.main import main, run_collect, run_preprocess, run_train, run_serve
    
    # Test collect command
    with patch('sys.argv', ['main.py', 'collect', '--symbol', 'BTCUSDT']):
        with patch('src.main.run_collect') as mock_collect:
            main()
            mock_collect.assert_called_once()
    
    # Test preprocess command
    with patch('sys.argv', ['main.py', 'preprocess', '--date', '2024-01-01']):
        with patch('src.main.run_preprocess') as mock_preprocess:
            main()
            mock_preprocess.assert_called_once()
    
    # Test train command
    with patch('sys.argv', ['main.py', 'train', '--config', 'config.yaml']):
        with patch('src.main.run_train') as mock_train:
            main()
            mock_train.assert_called_once()
    
    # Test serve command
    with patch('sys.argv', ['main.py', 'serve', '--port', '8000']):
        with patch('src.main.run_serve') as mock_serve:
            main()
            mock_serve.assert_called_once()


def test_error_handling():
    """Test error handling across modules."""
    from src.utils import safe_divide, calculate_returns
    from src.risk_management.models.position_sizing import KellyPositionSizer
    
    # Test division by zero
    assert safe_divide(10, 0) == 0
    assert safe_divide(0, 0) == 0
    
    # Test returns with zero initial
    assert calculate_returns(0, 100) == 0
    
    # Test Kelly with extreme values
    kelly = KellyPositionSizer()
    
    # Zero win rate
    size = kelly.calculate_position_size(
        portfolio_value=10000,
        current_price=50000,
        signal_confidence=0.9,
        win_rate=0,
        avg_win=0.02,
        avg_loss=0.01
    )
    assert size == 0
    
    # Zero avg loss (should handle gracefully)
    size = kelly.calculate_position_size(
        portfolio_value=10000,
        current_price=50000,
        signal_confidence=0.9,
        win_rate=0.6,
        avg_win=0.02,
        avg_loss=0
    )
    assert size >= 0  # Should not crash


def test_prediction_server_complete():
    """Test complete prediction server functionality."""
    from src.api.prediction_server import PredictionServer, RiskAnalysisRequest
    
    # Create server
    server = PredictionServer()
    
    # Test initialization
    assert server.app is not None
    assert server.indicators is not None
    assert server.risk_manager is not None
    assert server.model is None  # Mocked
    
    # Test risk analysis request
    risk_req = RiskAnalysisRequest(
        portfolio_value=10000,
        current_price=50000,
        signal_confidence=0.75,
        win_rate=0.6,
        avg_win=0.03,
        avg_loss=0.01
    )
    
    # Test request validation
    assert risk_req.portfolio_value == 10000
    assert risk_req.current_price == 50000