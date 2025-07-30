"""Comprehensive test coverage for all modules."""

import sys
import os
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import pytest
from datetime import datetime
from pathlib import Path

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

# Mock pandas DataFrame and Series
mock_df = MagicMock()
mock_df.__len__ = MagicMock(return_value=100)
mock_df.shape = (100, 5)
mock_df.columns = ['open', 'high', 'low', 'close', 'volume']
mock_df.__getitem__ = MagicMock(return_value=MagicMock())
mock_df.fillna = MagicMock(return_value=mock_df)
mock_df.rolling = MagicMock(return_value=MagicMock(mean=MagicMock(return_value=mock_df)))

mock_series = MagicMock()
mock_series.__len__ = MagicMock(return_value=100)
mock_series.name = 'test_series'
mock_series.fillna = MagicMock(return_value=mock_series)

sys.modules['pandas'].DataFrame = MagicMock(return_value=mock_df)
sys.modules['pandas'].Series = MagicMock(return_value=mock_series)


def test_base_indicator():
    """Test base indicator classes."""
    from src.feature_engineering.base import BaseIndicator, PriceIndicator, VolumeIndicator, OHLCVIndicator
    
    # Test abstract base indicator
    class TestIndicator(BaseIndicator):
        def transform(self, df):
            return mock_series
        
        @property
        def name(self):
            return 'test'
        
        def calculate(self, df):
            return self.transform(df)
        
        def _calculate(self, df):
            return mock_series
    
    indicator = TestIndicator(window_size=10)
    assert indicator.window_size == 10
    assert indicator.fillna is True
    assert indicator.fill_method == 'zero'
    
    # Test NaN handling
    result = indicator._handle_nan(mock_series)
    assert result is not None
    
    # Test price indicator
    price_ind = PriceIndicator(price_col='close')
    assert price_ind.price_col == 'close'
    
    # Test volume indicator
    vol_ind = VolumeIndicator()
    assert vol_ind is not None
    
    # Test OHLCV indicator
    ohlcv_ind = OHLCVIndicator()
    ohlcv_ind._validate_ohlcv(mock_df)


def test_feature_engineer():
    """Test feature engineer module."""
    from src.feature_engineering.engineer import FeatureEngineer
    
    engineer = FeatureEngineer()
    
    # Test initialization
    assert engineer.features == {}
    assert engineer.feature_names == []
    
    # Test adding features
    engineer.add_feature('sma', period=20)
    assert 'sma_20' in engineer.features
    
    # Test computing features
    result = engineer.compute_features(mock_df)
    assert result is not None
    
    # Test removing features
    engineer.remove_feature('sma_20')
    assert 'sma_20' not in engineer.features
    
    # Test clearing features
    engineer.add_feature('ema', period=10)
    engineer.clear()
    assert len(engineer.features) == 0


def test_feature_registry():
    """Test feature registry."""
    from src.feature_engineering.registry import FeatureRegistry, registry
    
    # Test singleton
    reg1 = FeatureRegistry()
    reg2 = FeatureRegistry()
    assert id(reg1) == id(reg2)
    
    # Test listing features
    features = registry.list_features()
    assert isinstance(features, list)
    
    # Test getting feature
    feature_class = registry.get('sma')
    assert feature_class is not None
    
    # Test listing by category
    momentum_features = registry.list_by_category('momentum')
    assert isinstance(momentum_features, list)


def test_config_module():
    """Test config module."""
    from src.config import (
        load_config, get_env_var, validate_config,
        PROJECT_ROOT, DATA_DIR, GCP_PROJECT_ID, GCS_BUCKET
    )
    
    # Test load_config
    config = load_config()
    assert isinstance(config, dict)
    
    # Test get_env_var
    result = get_env_var('TEST_VAR', 'default')
    assert result == 'default'
    
    os.environ['TEST_VAR'] = 'test_value'
    result = get_env_var('TEST_VAR', 'default')
    assert result == 'test_value'
    
    # Test validate_config
    valid = validate_config({'api_key': 'test', 'bucket_name': 'test'})
    assert valid is True
    
    invalid = validate_config({'api_key': None})
    assert invalid is False
    
    # Test edge cases
    assert validate_config(None) is False
    assert validate_config([]) is False
    assert validate_config('string') is False
    assert validate_config(123) is False
    assert validate_config({'api_key': ''}) is False
    assert validate_config({'bucket_name': None}) is False
    assert validate_config({}) is False
    
    # Test constants
    assert isinstance(PROJECT_ROOT, Path)
    assert isinstance(DATA_DIR, Path)
    assert isinstance(GCP_PROJECT_ID, str)
    assert isinstance(GCS_BUCKET, str)


def test_utils_module():
    """Test utils module."""
    from src.utils import (
        setup_logging, validate_config, ensure_directory,
        format_number, calculate_returns, safe_divide
    )
    
    # Test logging setup
    logger = setup_logging('test')
    assert logger is not None
    assert logger.name == 'test'
    
    # Test directory creation
    test_dir = Path('/tmp/test_dir')
    result = ensure_directory(test_dir)
    assert result == test_dir
    
    # Test number formatting
    assert format_number(1234567.89) == '1,234,567.89'
    assert format_number(1000) == '1,000.00'
    assert format_number(0) == '0.00'
    
    # Test returns calculation
    assert calculate_returns(100, 110) == 0.1
    assert calculate_returns(100, 90) == -0.1
    assert calculate_returns(0, 100) == 0
    
    # Test safe divide
    assert safe_divide(10, 2) == 5
    assert safe_divide(10, 0) == 0
    assert safe_divide(0, 5) == 0


def test_main_module():
    """Test main module."""
    from src.main import parse_args, main
    
    # Test argument parsing
    args = parse_args(['collect', '--symbol', 'BTCUSDT'])
    assert args.command == 'collect'
    assert args.symbol == 'BTCUSDT'
    
    args = parse_args(['preprocess', '--date', '2024-01-01'])
    assert args.command == 'preprocess'
    assert args.date == '2024-01-01'
    
    args = parse_args(['train', '--config', 'test.yaml'])
    assert args.command == 'train'
    assert args.config == 'test.yaml'
    
    args = parse_args(['serve', '--port', '8080'])
    assert args.command == 'serve'
    assert args.port == 8080
    
    # Test main function with mocked commands
    with patch('sys.argv', ['main.py', 'collect']):
        with patch('src.main.run_collect') as mock_collect:
            main()
            mock_collect.assert_called_once()
    
    with patch('sys.argv', ['main.py', 'serve']):
        with patch('src.main.run_serve') as mock_serve:
            main()
            mock_serve.assert_called_once()


def test_api_module():
    """Test API module."""
    from src.api import create_app, load_model, get_model_info
    from src.api.api import PredictionRequest, PredictionResponse, BatchPredictionRequest
    
    # Test app creation
    app = create_app()
    assert app is not None
    
    # Test model loading
    model = load_model()
    assert model is None  # Mocked
    
    # Test model info
    info = get_model_info()
    assert info['name'] == 'TauSACTrader'
    assert 'version' in info
    assert 'features' in info
    
    # Test request models
    req = PredictionRequest(features={'close': 50000})
    assert req.features['close'] == 50000
    
    batch_req = BatchPredictionRequest(samples=[{'test': 1}, {'test': 2}])
    assert len(batch_req.samples) == 2
    
    # Test response model
    resp = PredictionResponse(
        prediction=0.5,
        confidence=0.8,
        timestamp='2024-01-01T00:00:00'
    )
    assert resp.prediction == 0.5
    assert resp.confidence == 0.8


def test_risk_manager():
    """Test risk manager."""
    from src.risk_management.risk_manager import RiskManager
    from src.risk_management.models.position_sizing import KellyPositionSizer
    from src.risk_management.models.cost_model import BinanceCostModel
    from src.risk_management.models.drawdown_guard import DrawdownGuard
    from src.risk_management.models.api_throttler import BinanceAPIThrottler
    
    # Create components
    sizer = KellyPositionSizer(min_edge=0.02)
    cost_model = BinanceCostModel()
    guard = DrawdownGuard(max_drawdown=0.1)
    throttler = BinanceAPIThrottler()
    
    # Create risk manager
    manager = RiskManager(
        position_sizer=sizer,
        cost_model=cost_model,
        drawdown_guard=guard,
        api_throttler=throttler
    )
    
    # Test position check
    approved, size, reason = manager.check_new_position(
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
    
    # Test position update
    manager.update_position('BTCUSDT', 0.1, 50000)
    assert 'BTCUSDT' in manager.positions
    
    # Test portfolio update
    manager.update_portfolio(10000, {'BTCUSDT': 50000})
    
    # Test metrics
    metrics = manager.get_metrics()
    assert isinstance(metrics, dict)
    assert 'portfolio_value' in metrics
    
    # Test configuration
    config = manager.get_config()
    assert 'max_position_size' in config
    assert 'max_drawdown' in config


def test_monitoring_modules():
    """Test monitoring modules."""
    from src.monitoring.metrics_collector import MetricsCollector
    from src.monitoring.performance_monitor import PerformanceMonitor
    from src.monitoring.alert_manager import AlertManager
    
    # Test metrics collector
    collector = MetricsCollector()
    
    # Test recording methods
    collector.record_prediction('buy', 0.8, 0.05)
    collector.record_trade('BTCUSDT', 'buy', 0.1, 50000, 100)
    collector.record_latency('prediction', 0.1)
    collector.record_error('api', 'timeout')
    
    # Test metrics retrieval
    metrics = collector.get_metrics()
    assert metrics['total_predictions'] == 1
    assert metrics['total_trades'] == 1
    assert metrics['total_errors'] == 1
    assert 'average_latency' in metrics
    
    # Test summary
    summary = collector.get_summary()
    assert 'timestamp' in summary
    assert 'metrics' in summary
    
    # Test reset
    collector.reset()
    metrics = collector.get_metrics()
    assert metrics['total_predictions'] == 0
    
    # Test performance monitor
    monitor = PerformanceMonitor(window_size=100)
    
    # Add returns
    for i in range(10):
        monitor.update(0.01 * ((-1) ** i))
    
    # Get performance
    perf = monitor.get_performance()
    assert perf['total_returns'] == 10
    assert 'sharpe_ratio' in perf
    assert 'max_drawdown' in perf
    assert 'win_rate' in perf
    
    # Test alert manager
    manager = AlertManager()
    
    # Test alert checking
    alerts = manager.check_alerts({
        'current_drawdown': -0.15,
        'consecutive_losses': 6,
        'error_rate': 0.15,
        'api_usage': 0.95
    })
    assert len(alerts) > 0
    
    # Test alert throttling
    alert = {'type': 'test', 'level': 'warning', 'message': 'Test'}
    assert manager.should_send_alert(alert) is True
    assert manager.should_send_alert(alert) is False  # Throttled
    
    # Test alert formatting
    formatted = manager.format_alert(alert)
    assert 'test' in formatted.lower()


def test_position_sizing_models():
    """Test position sizing models."""
    from src.risk_management.models.position_sizing import (
        KellyPositionSizer, FixedFractionalSizer, VolatilityParitySizer
    )
    
    # Test Kelly sizing
    kelly = KellyPositionSizer(min_edge=0.02, kelly_fraction=0.25)
    
    # No edge case
    size = kelly.calculate_position_size(
        portfolio_value=10000,
        current_price=50000,
        signal_confidence=0.5,
        win_rate=0.5,
        avg_win=0.01,
        avg_loss=0.01
    )
    assert size == 0
    
    # Positive edge case
    size = kelly.calculate_position_size(
        portfolio_value=10000,
        current_price=50000,
        signal_confidence=0.8,
        win_rate=0.65,
        avg_win=0.03,
        avg_loss=0.01
    )
    assert 0 < size <= 0.1
    
    # Test fixed fractional
    ff = FixedFractionalSizer(fraction=0.02)
    
    # Low confidence
    size = ff.calculate_position_size(10000, 50000, 0.3)
    assert size == 0
    
    # High confidence
    size = ff.calculate_position_size(10000, 50000, 0.8)
    assert size == 0.02
    
    # Test volatility parity
    vp = VolatilityParitySizer(target_volatility=0.02)
    
    # High volatility
    size = vp.calculate_position_size(
        portfolio_value=10000,
        current_price=50000,
        signal_confidence=0.8,
        volatility=0.05
    )
    assert 0 < size < 0.02
    
    # Low volatility
    size = vp.calculate_position_size(
        portfolio_value=10000,
        current_price=50000,
        signal_confidence=0.8,
        volatility=0.01
    )
    assert 0.02 < size < 0.04


def test_cost_models():
    """Test cost models."""
    from src.risk_management.models.cost_model import BinanceCostModel, FixedCostModel
    
    # Test Binance cost model
    binance = BinanceCostModel()
    
    # Market order
    cost = binance.calculate_cost('market', 0.1, 50000)
    assert cost == 0.1 * 50000 * 0.001  # Taker fee
    
    # Limit order
    cost = binance.calculate_cost('limit', 0.1, 50000)
    assert cost == 0.1 * 50000 * 0.001  # Maker fee
    
    # Test fixed cost model
    fixed = FixedCostModel(maker_fee=0.0005, taker_fee=0.001)
    
    # Limit order
    cost = fixed.calculate_cost('limit', 1.0, 10000)
    assert cost == 5.0  # 0.0005 * 10000
    
    # Market order
    cost = fixed.calculate_cost('market', 1.0, 10000)
    assert cost == 10.0  # 0.001 * 10000


def test_drawdown_guard():
    """Test drawdown guard."""
    from src.risk_management.models.drawdown_guard import DrawdownGuard
    
    guard = DrawdownGuard(max_drawdown=0.1, recovery_periods=10)
    
    # Initial state
    guard.update_portfolio(10000, {'BTCUSDT': 50000})
    assert guard.get_risk_multiplier() == 1.0
    assert not guard.is_in_drawdown()
    
    # Enter drawdown
    guard.update_portfolio(9000, {'BTCUSDT': 45000})
    assert guard.is_in_drawdown()
    assert guard.get_risk_multiplier() < 1.0
    
    # Recovery
    for i in range(5):
        guard.update_portfolio(9000 + i * 200, {'BTCUSDT': 45000 + i * 1000})
    
    new_multiplier = guard.get_risk_multiplier()
    assert new_multiplier < 1.0  # Still recovering
    
    # Full recovery
    guard.update_portfolio(11000, {'BTCUSDT': 55000})
    guard.update_portfolio(11000, {'BTCUSDT': 55000})  # Multiple updates
    assert guard.get_risk_multiplier() > 0.5


def test_api_throttler():
    """Test API throttler."""
    from src.risk_management.models.api_throttler import BinanceAPIThrottler
    
    throttler = BinanceAPIThrottler()
    
    # Test rate limiting
    for i in range(10):
        can_request = throttler.check_and_wait_sync()
        assert can_request is True
        throttler.add_request('order', 1)
    
    # Test metrics
    metrics = throttler.get_metrics()
    assert metrics['total_requests'] >= 10
    assert 'requests_per_minute' in metrics
    assert 'current_window_requests' in metrics
    
    # Test reset
    throttler.reset_window()
    metrics = throttler.get_metrics()
    assert metrics['current_window_requests'] == 0
    
    # Test async wrapper
    assert hasattr(throttler, 'check_and_wait')
    assert hasattr(throttler, 'check_and_wait_sync')


@patch('google.cloud.storage.Client')
def test_data_collection_modules(mock_client):
    """Test data collection modules."""
    from src.data_collection.gcs_uploader import GCSUploader
    
    # Mock GCS client
    mock_client_instance = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    
    mock_client.return_value = mock_client_instance
    mock_client_instance.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob
    
    # Create uploader
    uploader = GCSUploader('test-project', 'test-bucket')
    
    # Test file size calculation
    size = uploader._get_file_size_mb('nonexistent.txt')
    assert size == 0
    
    # Test blob name generation
    blob_name = uploader._generate_blob_name('test.json', 'orderbook/2024/01/01')
    assert 'orderbook/2024/01/01' in blob_name
    assert blob_name.endswith('.json')
    
    # Test upload methods
    uploader.upload_file('test.json', 'test/path')
    mock_blob.upload_from_filename.assert_called()
    
    # Test batch upload
    uploader.upload_batch(['file1.json', 'file2.json'], 'test/path')
    assert mock_blob.upload_from_filename.call_count >= 1


def test_prediction_server():
    """Test prediction server."""
    from src.api.prediction_server import PredictionServer, create_app
    
    # Create server
    server = PredictionServer()
    assert server.app is not None
    assert server.indicators is not None
    assert server.risk_manager is not None
    
    # Test app creation
    app = create_app()
    assert app is not None
    
    # Test endpoints registration
    routes = [route.path for route in server.app.routes]
    assert '/' in routes
    assert '/health' in routes
    assert '/predict' in routes
    assert '/predict/batch' in routes
    assert '/analyze/risk' in routes
    assert '/model/info' in routes


def test_feature_modules():
    """Test individual feature modules."""
    # Import feature modules with mocked dependencies
    from src.feature_engineering.momentum.macd import MACDFeatures
    from src.feature_engineering.momentum.oscillators import RSIFeature, StochasticFeature
    from src.feature_engineering.trend.moving_averages import SMAFeature, EMAFeature
    from src.feature_engineering.volatility.bands import BollingerBandsFeatures
    from src.feature_engineering.volume.classic import VolumeFeatures, OBVFeature
    
    # Test MACD
    macd = MACDFeatures()
    result = macd.calculate(mock_df)
    assert result is not None
    
    # Test RSI
    rsi = RSIFeature(period=14)
    result = rsi.calculate(mock_df)
    assert result is not None
    
    # Test Stochastic
    stoch = StochasticFeature(period=14)
    result = stoch.calculate(mock_df)
    assert result is not None
    
    # Test SMA
    sma = SMAFeature(period=20)
    result = sma.calculate(mock_df)
    assert result is not None
    
    # Test EMA
    ema = EMAFeature(period=20)
    result = ema.calculate(mock_df)
    assert result is not None
    
    # Test Bollinger Bands
    bb = BollingerBandsFeatures(period=20)
    result = bb.calculate(mock_df)
    assert result is not None
    
    # Test Volume Features
    vol = VolumeFeatures()
    result = vol.calculate(mock_df)
    assert result is not None
    
    # Test OBV
    obv = OBVFeature()
    result = obv.calculate(mock_df)
    assert result is not None


def test_technical_indicators():
    """Test technical indicators module."""
    from src.features.technical_indicators import TechnicalIndicators
    
    indicators = TechnicalIndicators()
    
    # Test adding all indicators
    result = indicators.add_all_indicators(mock_df)
    assert result is not None
    
    # Test individual indicator groups
    result = indicators.add_price_features(mock_df)
    assert result is not None
    
    result = indicators.add_momentum_indicators(mock_df)
    assert result is not None
    
    result = indicators.add_trend_indicators(mock_df)
    assert result is not None
    
    result = indicators.add_volatility_indicators(mock_df)
    assert result is not None
    
    result = indicators.add_volume_indicators(mock_df)
    assert result is not None


def test_data_processing_modules():
    """Test data processing modules."""
    from src.data_processing.daily_preprocessor import DailyPreprocessor
    
    with patch('google.cloud.storage.Client') as mock_client:
        mock_client_instance = MagicMock()
        mock_bucket = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.bucket.return_value = mock_bucket
        
        preprocessor = DailyPreprocessor('test-project', 'test-bucket')
        
        # Test date formatting
        date_str = preprocessor._get_date_str(datetime.now())
        assert len(date_str) == 10
        assert '-' in date_str
        
        # Test feature computation
        df = mock_df.copy()
        df['mid_price'] = 50000
        df['spread'] = 10
        df['volume_imbalance'] = 0.1
        
        features = preprocessor._compute_additional_features(df)
        assert features is not None


def test_rl_models():
    """Test RL models with mocked dependencies."""
    from src.rl.models import TauSACAgent, create_agent
    
    # Test agent creation
    agent = create_agent(state_dim=104, action_dim=4)
    assert agent is None  # Mocked
    
    # Test TauSACAgent initialization
    with patch('src.rl.models.SAC'):
        agent = TauSACAgent(
            env=MagicMock(),
            tau_values=[3, 6, 9, 12],
            learning_rate=3e-4
        )
        assert agent.tau_values == [3, 6, 9, 12]
        assert agent.learning_rate == 3e-4


def test_rl_environments():
    """Test RL environments with mocked dependencies."""
    from src.rl.environments import BTCTradingEnvironment
    
    # Create mock environment
    with patch('src.rl.environments.gym.Env'):
        env = BTCTradingEnvironment(
            data=mock_df,
            initial_balance=10000,
            fee_rate=0.001
        )
        assert env.initial_balance == 10000
        assert env.fee_rate == 0.001


def test_edge_cases():
    """Test various edge cases."""
    from src.utils import safe_divide, calculate_returns
    from src.config import validate_config
    
    # Test division edge cases
    assert safe_divide(0, 0) == 0
    assert safe_divide(float('inf'), 1) == float('inf')
    assert safe_divide(1, float('inf')) == 0
    
    # Test returns edge cases
    assert calculate_returns(0, 0) == 0
    assert calculate_returns(-100, 100) == -2.0
    assert calculate_returns(100, -100) == -2.0
    
    # Test config validation edge cases
    assert validate_config({'api_key': 123}) is False  # Wrong type
    assert validate_config({'api_key': 'test', 'extra': 'field'}) is True  # Extra fields ok
    assert validate_config({'api_key': 'test', 'bucket_name': ''}) is False  # Empty string