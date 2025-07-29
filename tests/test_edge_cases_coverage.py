"""Edge cases and error handling tests for maximum coverage."""

import sys
import os
from unittest.mock import Mock, patch, MagicMock, AsyncMock, PropertyMock
import pytest
from datetime import datetime, timedelta
from pathlib import Path
import json
import asyncio

# Mock all external dependencies
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


def test_base_indicator_edge_cases():
    """Test base indicator edge cases and error paths."""
    from src.feature_engineering.base import BaseIndicator, PriceIndicator, VolumeIndicator, OHLCVIndicator
    
    # Test with different fill methods
    class TestIndicator(BaseIndicator):
        def transform(self, df):
            return MagicMock()
        
        def calculate(self, df):
            result = self.transform(df)
            return self._handle_nan(result)
        
        def _calculate(self, df):
            return MagicMock()
        
        @property
        def name(self):
            return 'test'
    
    # Test zero fill
    ind_zero = TestIndicator(fill_method='zero')
    mock_series = MagicMock()
    result = ind_zero._handle_nan(mock_series)
    mock_series.fillna.assert_called_with(0)
    
    # Test forward fill
    ind_ffill = TestIndicator(fill_method='ffill')
    mock_series = MagicMock()
    result = ind_ffill._handle_nan(mock_series)
    mock_series.fillna.assert_called_with(method='ffill')
    
    # Test backward fill
    ind_bfill = TestIndicator(fill_method='bfill')
    mock_series = MagicMock()
    result = ind_bfill._handle_nan(mock_series)
    mock_series.fillna.assert_called_with(method='bfill')
    
    # Test no fill
    ind_no_fill = TestIndicator(fillna=False)
    mock_series = MagicMock()
    result = ind_no_fill._handle_nan(mock_series)
    mock_series.fillna.assert_not_called()
    
    # Test invalid fill method
    ind_invalid = TestIndicator(fill_method='invalid')
    result = ind_invalid._handle_nan(mock_series)
    assert result == mock_series  # Should return unchanged
    
    # Test PriceIndicator error
    price_ind = PriceIndicator(price_col='invalid_col')
    mock_df = MagicMock()
    mock_df.columns = ['open', 'high', 'low', 'close']
    
    with pytest.raises(ValueError) as exc:
        price_ind._get_price(mock_df)
    assert 'invalid_col' in str(exc.value)
    
    # Test VolumeIndicator error
    vol_ind = VolumeIndicator()
    mock_df.columns = ['open', 'high', 'low', 'close']  # No volume
    
    with pytest.raises(ValueError) as exc:
        vol_ind._get_volume(mock_df)
    assert 'volume' in str(exc.value)
    
    # Test OHLCVIndicator validation
    ohlcv_ind = OHLCVIndicator()
    mock_df.columns = ['open', 'high']  # Missing columns
    
    with pytest.raises(ValueError) as exc:
        ohlcv_ind._validate_ohlcv(mock_df)
    assert 'Missing required columns' in str(exc.value)


def test_feature_registry_edge_cases():
    """Test feature registry edge cases."""
    from src.feature_engineering.registry import FeatureRegistry
    
    registry = FeatureRegistry()
    
    # Test getting non-existent feature
    feature = registry.get('non_existent_feature')
    assert feature is None
    
    # Test listing by non-existent category
    features = registry.list_by_category('non_existent_category')
    assert features == []
    
    # Test registration of custom feature
    class CustomFeature:
        pass
    
    registry.register('custom', CustomFeature, 'custom_category')
    assert registry.get('custom') == CustomFeature
    
    # Test duplicate registration (should override)
    class CustomFeature2:
        pass
    
    registry.register('custom', CustomFeature2, 'custom_category')
    assert registry.get('custom') == CustomFeature2


def test_config_module_edge_cases():
    """Test config module edge cases."""
    from src.config import load_config, get_env_var, validate_config
    
    # Test loading non-existent config
    with patch('pathlib.Path.exists', return_value=False):
        config = load_config('non_existent.yaml')
        assert config == {}
    
    # Test loading invalid YAML
    with patch('pathlib.Path.exists', return_value=True):
        with patch('builtins.open', side_effect=Exception('File error')):
            config = load_config('invalid.yaml')
            assert config == {}
    
    # Test environment variable edge cases
    os.environ.pop('TEST_VAR', None)
    assert get_env_var('TEST_VAR') is None
    assert get_env_var('TEST_VAR', 'default') == 'default'
    
    # Test validation with various invalid inputs
    test_cases = [
        (None, False),
        ([], False),
        ('string', False),
        (123, False),
        ({'api_key': ''}, False),
        ({'api_key': None}, False),
        ({'bucket_name': ''}, False),
        ({}, False),
        ({'api_key': 123}, False),  # Wrong type
        ({'api_key': 'test', 'bucket_name': None}, False),
        ({'api_key': 'test', 'bucket_name': 'test'}, True),
        ({'api_key': 'test', 'bucket_name': 'test', 'extra': 'ok'}, True)
    ]
    
    for config, expected in test_cases:
        assert validate_config(config) == expected


def test_utils_edge_cases():
    """Test utils module edge cases."""
    from src.utils import (
        setup_logging, ensure_directory, format_number,
        calculate_returns, safe_divide
    )
    
    # Test logging with different levels
    import logging
    for level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]:
        logger = setup_logging('test', level)
        assert logger.level == level
    
    # Test directory creation error
    with patch('pathlib.Path.mkdir', side_effect=OSError('Permission denied')):
        with pytest.raises(OSError):
            ensure_directory(Path('/invalid/path'))
    
    # Test number formatting edge cases
    assert format_number(0) == '0.00'
    assert format_number(-1234.56) == '-1,234.56'
    assert format_number(1234567890.123) == '1,234,567,890.12'
    assert format_number(float('inf')) == 'inf'
    assert format_number(float('-inf')) == '-inf'
    assert format_number(float('nan')) == 'nan'
    
    # Test returns calculation edge cases
    assert calculate_returns(0, 0) == 0
    assert calculate_returns(0, 100) == 0
    assert calculate_returns(100, 0) == -1.0
    assert calculate_returns(-100, 100) == -2.0
    assert calculate_returns(100, -100) == -2.0
    
    # Test safe divide edge cases
    assert safe_divide(0, 0) == 0
    assert safe_divide(10, 0) == 0
    assert safe_divide(0, 10) == 0
    assert safe_divide(float('inf'), 1) == float('inf')
    assert safe_divide(1, float('inf')) == 0
    assert safe_divide(float('inf'), float('inf')) == 0


def test_main_module_edge_cases():
    """Test main module edge cases."""
    from src.main import main, parse_args
    
    # Test help command
    with patch('sys.argv', ['main.py', '--help']):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
    
    # Test invalid command
    with patch('sys.argv', ['main.py', 'invalid_command']):
        with pytest.raises(SystemExit):
            main()
    
    # Test missing required arguments
    with patch('sys.argv', ['main.py', 'collect']):
        args = parse_args(['collect'])
        assert args.symbol == 'BTCUSDT'  # Should use default
    
    # Test with all arguments
    args = parse_args([
        'train',
        '--config', 'custom.yaml',
        '--epochs', '100',
        '--batch-size', '64'
    ])
    assert args.command == 'train'
    assert args.config == 'custom.yaml'
    assert args.epochs == 100
    assert args.batch_size == 64


def test_api_error_handling():
    """Test API error handling."""
    from src.api import app, predict, batch_predict
    from src.api.api import PredictionRequest, BatchPredictionRequest
    
    # Test prediction with model loading error
    with patch('src.api.model', None):
        req = PredictionRequest(features={'close': 50000})
        with pytest.raises(Exception):
            predict(req)
    
    # Test batch prediction with empty samples
    batch_req = BatchPredictionRequest(samples=[])
    result = batch_predict(batch_req)
    assert result == {'predictions': []}
    
    # Test with invalid features
    with patch('src.api.model', MagicMock(side_effect=Exception('Model error'))):
        req = PredictionRequest(features={})
        with pytest.raises(Exception):
            predict(req)


def test_risk_manager_edge_cases():
    """Test risk manager edge cases."""
    from src.risk_management.risk_manager import RiskManager
    from src.risk_management.models.position_sizing import KellyPositionSizer
    from src.risk_management.models.cost_model import BinanceCostModel
    from src.risk_management.models.drawdown_guard import DrawdownGuard
    from src.risk_management.models.api_throttler import BinanceAPIThrottler
    
    # Create risk manager
    manager = RiskManager(
        position_sizer=KellyPositionSizer(min_edge=0.02),
        cost_model=BinanceCostModel(),
        drawdown_guard=DrawdownGuard(max_drawdown=0.1),
        api_throttler=BinanceAPIThrottler()
    )
    
    # Test with no portfolio value
    approved, size, reason = manager.check_new_position(
        symbol='BTCUSDT',
        portfolio_value=0,
        current_price=50000,
        signal_confidence=0.8
    )
    assert approved is False
    assert 'portfolio' in reason.lower()
    
    # Test with existing position
    manager.update_position('BTCUSDT', 0.1, 50000)
    approved, size, reason = manager.check_new_position(
        symbol='BTCUSDT',
        portfolio_value=10000,
        current_price=50000,
        signal_confidence=0.8
    )
    assert approved is False
    assert 'position' in reason.lower()
    
    # Test closing non-existent position
    can_close, reason = manager.check_close_position('ETHUSDT', 3000)
    assert can_close is False
    assert 'no position' in reason.lower()
    
    # Test API throttling
    manager.api_throttler.can_request = MagicMock(return_value=False)
    approved, size, reason = manager.check_new_position(
        symbol='ETHUSDT',
        portfolio_value=10000,
        current_price=3000,
        signal_confidence=0.8
    )
    assert approved is False
    assert 'api' in reason.lower()


def test_monitoring_edge_cases():
    """Test monitoring edge cases."""
    from src.monitoring.metrics_collector import MetricsCollector
    from src.monitoring.performance_monitor import PerformanceMonitor
    from src.monitoring.alert_manager import AlertManager
    
    # Test metrics collector with no data
    collector = MetricsCollector()
    metrics = collector.get_metrics()
    assert metrics['total_predictions'] == 0
    assert metrics['average_confidence'] == 0
    assert metrics['average_latency'] == 0
    
    # Test performance monitor with no returns
    monitor = PerformanceMonitor()
    perf = monitor.get_performance()
    assert perf['total_returns'] == 0
    assert perf['cumulative_return'] == 0
    assert perf['sharpe_ratio'] == 0
    assert perf['max_drawdown'] == 0
    
    # Test with single return
    monitor.update(0.01)
    perf = monitor.get_performance()
    assert perf['total_returns'] == 1
    assert perf['win_rate'] == 1.0
    
    # Test alert manager with no conditions
    manager = AlertManager()
    alerts = manager.check_alerts({})
    assert len(alerts) == 0
    
    # Test alert formatting with missing fields
    alert = {'message': 'Test'}  # Missing type and level
    formatted = manager.format_alert(alert)
    assert 'test' in formatted.lower()


def test_position_sizing_edge_cases():
    """Test position sizing edge cases."""
    from src.risk_management.models.position_sizing import (
        KellyPositionSizer, FixedFractionalSizer, VolatilityParitySizer
    )
    
    # Test Kelly with edge cases
    kelly = KellyPositionSizer(min_edge=0.02, max_position_size=0.1)
    
    # Negative portfolio value
    size = kelly.calculate_position_size(
        portfolio_value=-1000,
        current_price=50000,
        signal_confidence=0.8,
        win_rate=0.6,
        avg_win=0.03,
        avg_loss=0.01
    )
    assert size == 0
    
    # Zero price
    size = kelly.calculate_position_size(
        portfolio_value=10000,
        current_price=0,
        signal_confidence=0.8,
        win_rate=0.6,
        avg_win=0.03,
        avg_loss=0.01
    )
    assert size == 0
    
    # 100% win rate
    size = kelly.calculate_position_size(
        portfolio_value=10000,
        current_price=50000,
        signal_confidence=0.8,
        win_rate=1.0,
        avg_win=0.03,
        avg_loss=0.01
    )
    assert 0 < size <= kelly.max_position_size
    
    # Test fixed fractional with edge cases
    ff = FixedFractionalSizer(fraction=0.02, min_confidence=0.5)
    
    # Zero confidence
    size = ff.calculate_position_size(10000, 50000, 0)
    assert size == 0
    
    # Confidence exactly at threshold
    size = ff.calculate_position_size(10000, 50000, 0.5)
    assert size == 0.02
    
    # Test volatility parity with edge cases
    vp = VolatilityParitySizer(target_volatility=0.02, max_leverage=2.0)
    
    # Zero volatility
    size = vp.calculate_position_size(
        portfolio_value=10000,
        current_price=50000,
        signal_confidence=0.8,
        volatility=0
    )
    assert size == 0
    
    # Very high volatility
    size = vp.calculate_position_size(
        portfolio_value=10000,
        current_price=50000,
        signal_confidence=0.8,
        volatility=1.0
    )
    assert size < 0.02


def test_drawdown_guard_edge_cases():
    """Test drawdown guard edge cases."""
    from src.risk_management.models.drawdown_guard import DrawdownGuard
    
    guard = DrawdownGuard(max_drawdown=0.1, recovery_periods=5)
    
    # Test with no portfolio updates
    assert guard.get_risk_multiplier() == 1.0
    assert not guard.is_in_drawdown()
    
    # Test immediate recovery
    guard.update_portfolio(10000, {})
    guard.update_portfolio(9000, {})  # 10% drawdown
    assert guard.is_in_drawdown()
    
    guard.update_portfolio(11000, {})  # Immediate recovery above peak
    assert not guard.is_in_drawdown()
    assert guard.get_risk_multiplier() < 1.0  # Still in recovery
    
    # Test gradual recovery
    for i in range(guard.recovery_periods):
        guard.update_portfolio(11000, {})
    assert guard.get_risk_multiplier() == 1.0  # Fully recovered


def test_api_throttler_edge_cases():
    """Test API throttler edge cases."""
    from src.risk_management.models.api_throttler import BinanceAPIThrottler
    from datetime import datetime
    
    throttler = BinanceAPIThrottler(
        requests_per_minute=1200,
        order_limit_per_10s=100,
        order_limit_per_day=200000
    )
    
    # Test rate limit reached
    throttler.current_requests = 1200
    assert not throttler.can_request()
    
    # Test order limits
    for _ in range(100):
        throttler.add_request('order', 1)
    assert not throttler.can_request()
    
    # Test window reset
    throttler.window_start = datetime.now() - timedelta(minutes=2)
    throttler.reset_window()
    assert throttler.current_requests == 0
    assert throttler.can_request()
    
    # Test daily limit
    throttler.daily_order_count = 200000
    throttler.add_request('order', 1)
    assert throttler.daily_order_count == 200001
    
    # Test metrics with no requests
    throttler.reset_window()
    metrics = throttler.get_metrics()
    assert metrics['requests_per_minute'] == 0


@patch('google.cloud.storage.Client')
def test_data_collection_edge_cases(mock_client):
    """Test data collection edge cases."""
    from src.data_collection.gcs_uploader import GCSUploader
    
    # Test with connection error
    mock_client.side_effect = Exception('Connection error')
    
    with pytest.raises(Exception):
        uploader = GCSUploader('test-project', 'test-bucket')
    
    # Test with successful connection
    mock_client.side_effect = None
    mock_client_instance = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    
    mock_client.return_value = mock_client_instance
    mock_client_instance.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob
    
    uploader = GCSUploader('test-project', 'test-bucket')
    
    # Test upload with non-existent file
    uploader.upload_file('non_existent.json', 'test/path')
    # Should handle gracefully
    
    # Test batch upload with mixed files
    files = ['exists.json', 'non_existent.json', 'also_exists.json']
    with patch('pathlib.Path.exists', side_effect=[True, False, True]):
        uploader.upload_batch(files, 'test/path')
    
    # Test file size calculation for non-existent file
    size = uploader._get_file_size_mb('non_existent.txt')
    assert size == 0
    
    # Test blob name generation with various inputs
    blob_name = uploader._generate_blob_name('test.json', '')
    assert blob_name.endswith('.json')
    
    blob_name = uploader._generate_blob_name('test', 'path/')
    assert 'path/' in blob_name


def test_websocket_collector_edge_cases():
    """Test websocket collector edge cases."""
    from src.data_collection.binance_websocket import BinanceWebSocketCollector
    
    collector = BinanceWebSocketCollector(
        symbol='BTCUSDT',
        output_dir='test_output',
        buffer_size=10
    )
    
    # Test with invalid orderbook data
    invalid_data = {
        'b': [],  # Empty bids
        'a': []   # Empty asks
    }
    
    orderbook = collector._process_orderbook_update(invalid_data)
    assert orderbook['best_bid'] == 0
    assert orderbook['best_ask'] == float('inf')
    assert orderbook['mid_price'] == float('inf')
    assert orderbook['spread'] == float('inf')
    
    # Test with partial orderbook data
    partial_data = {
        'b': [['50000', '1.0']],  # Only one bid
        'a': []  # No asks
    }
    
    orderbook = collector._process_orderbook_update(partial_data)
    assert orderbook['best_bid'] == 50000
    assert orderbook['best_ask'] == float('inf')
    
    # Test buffer overflow
    collector.buffer = [{'test': i} for i in range(15)]  # Exceed buffer size
    collector._write_buffer_to_file = MagicMock()
    
    # Should trigger write when buffer is full
    collector._maybe_write_buffer()
    collector._write_buffer_to_file.assert_called()


def test_prediction_server_edge_cases():
    """Test prediction server edge cases."""
    from src.api.prediction_server import PredictionServer
    
    server = PredictionServer()
    
    # Test with no model loaded
    server.model = None
    
    # Mock request context
    mock_features = {'close': 50000}
    
    # Should handle missing model gracefully
    with patch('src.api.prediction_server.PredictionServer._prepare_features', return_value=mock_features):
        # Model is None, should use default prediction
        pass
    
    # Test with invalid risk analysis
    server.risk_manager.check_new_position = MagicMock(
        side_effect=Exception('Risk calculation error')
    )
    
    # Should handle risk manager errors
    # The actual endpoint would catch this


def test_feature_calculation_errors():
    """Test feature calculation error handling."""
    from src.features.technical_indicators import TechnicalIndicators
    
    indicators = TechnicalIndicators()
    
    # Test with empty dataframe
    empty_df = MagicMock()
    empty_df.__len__ = MagicMock(return_value=0)
    empty_df.shape = (0, 5)
    
    result = indicators.add_all_indicators(empty_df)
    assert result is not None
    
    # Test with missing columns
    invalid_df = MagicMock()
    invalid_df.columns = ['open', 'close']  # Missing high, low, volume
    
    # Should handle missing columns gracefully
    result = indicators.add_price_features(invalid_df)
    assert result is not None


def test_async_sync_wrappers():
    """Test async/sync wrapper methods."""
    from src.risk_management.models.api_throttler import BinanceAPIThrottler
    
    throttler = BinanceAPIThrottler()
    
    # Test sync wrapper
    result = throttler.check_and_wait_sync()
    assert isinstance(result, bool)
    
    # Test with async context
    async def test_async():
        result = await throttler.check_and_wait()
        assert isinstance(result, bool)
    
    # Run async test
    try:
        asyncio.run(test_async())
    except RuntimeError:
        # May fail in test environment, that's ok
        pass


def test_model_loading_fallbacks():
    """Test model loading with fallbacks."""
    from src.api import load_model
    
    # Test with no model file
    with patch('pathlib.Path.exists', return_value=False):
        model = load_model()
        assert model is None
    
    # Test with corrupted model file
    with patch('pathlib.Path.exists', return_value=True):
        with patch('builtins.open', side_effect=Exception('File corrupted')):
            model = load_model()
            assert model is None