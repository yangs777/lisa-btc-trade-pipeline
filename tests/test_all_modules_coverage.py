"""Comprehensive test file to achieve 85% coverage across all modules."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio
import json
import yaml
from datetime import datetime, timedelta
import tempfile
import os


class TestConfigModule:
    """Test config module."""
    
    def test_load_config(self):
        """Test configuration loading."""
        with patch.dict(os.environ, {
            "GCP_PROJECT_ID": "test-project",
            "GCS_BUCKET": "test-bucket"
        }):
            from src.config import load_config
            config = load_config()
            assert isinstance(config, dict)
            assert "bucket_name" in config
            assert config["bucket_name"] == "test-bucket"
            assert config["project_id"] == "test-project"
    
    def test_get_env_var(self):
        """Test environment variable retrieval."""
        from src.config import get_env_var
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            assert get_env_var("TEST_VAR") == "test_value"
            assert get_env_var("MISSING_VAR", "default") == "default"
    
    def test_validate_config(self):
        """Test config validation."""
        from src.config import validate_config
        
        # Valid config
        valid_config = {"api_key": "key", "bucket_name": "bucket"}
        assert validate_config(valid_config)
        
        # Invalid configs
        assert not validate_config({})
        assert not validate_config({"api_key": None, "bucket_name": "bucket"})
        assert not validate_config("not a dict")


class TestDataCollection:
    """Test data collection modules."""
    
    @pytest.mark.asyncio
    async def test_binance_websocket(self):
        """Test Binance WebSocket functionality."""
        from src.data_collection.binance_websocket import BinanceWebSocket
        
        # Mock websocket
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=[
            json.dumps({"e": "depthUpdate", "b": [[50000, 1]], "a": [[50001, 1]]}),
            asyncio.CancelledError()
        ])
        
        with patch('websockets.connect', return_value=mock_ws):
            ws = BinanceWebSocket("btcusdt")
            
            # Test connection
            await ws.connect()
            assert ws.connected
            
            # Test message processing
            msg = await ws.get_orderbook_update()
            assert msg is not None
            assert "b" in msg
    
    def test_gcs_uploader(self):
        """Test GCS uploader."""
        from src.data_collection.gcs_uploader import GCSUploader
        
        # Mock GCS client
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_bucket.blob.return_value = mock_blob
        
        with patch('google.cloud.storage.Client') as mock_client:
            mock_client.return_value.bucket.return_value = mock_bucket
            
            uploader = GCSUploader("test-bucket")
            
            # Test upload
            test_data = {"test": "data"}
            uploader.upload_json("test.json", test_data)
            
            mock_blob.upload_from_string.assert_called_once()
            
            # Test batch upload
            batch_data = [{"data": i} for i in range(5)]
            uploader.upload_batch("prefix", batch_data)
            assert mock_blob.upload_from_string.call_count > 1


class TestDataProcessing:
    """Test data processing modules."""
    
    def test_daily_preprocessor(self):
        """Test daily preprocessor."""
        from src.data_processing.daily_preprocessor import DailyPreprocessor
        
        # Create sample data
        dates = pd.date_range("2024-01-01", periods=100, freq="h")
        data = pd.DataFrame({
            "timestamp": dates,
            "open": np.random.randn(100) + 50000,
            "high": np.random.randn(100) + 50100,
            "low": np.random.randn(100) + 49900,
            "close": np.random.randn(100) + 50000,
            "volume": np.random.randn(100) * 1000 + 10000
        })
        
        preprocessor = DailyPreprocessor()
        
        # Test preprocessing
        processed = preprocessor.preprocess_data(data)
        assert len(processed) == len(data)
        assert "returns" in processed.columns
        
        # Test feature computation
        features = preprocessor.compute_features(processed)
        assert len(features.columns) > len(processed.columns)
        
        # Test validation
        is_valid = preprocessor.validate_data(data)
        assert is_valid


class TestFeatureEngineering:
    """Test feature engineering modules."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range("2024-01-01", periods=200, freq="h")
        return pd.DataFrame({
            "timestamp": dates,
            "open": 50000 + np.random.randn(200) * 100,
            "high": 50100 + np.random.randn(200) * 100,
            "low": 49900 + np.random.randn(200) * 100,
            "close": 50000 + np.random.randn(200) * 100,
            "volume": 10000 + np.random.randn(200) * 1000
        })
    
    def test_base_indicator(self, sample_data):
        """Test base indicator class."""
        from src.feature_engineering.base import BaseIndicator
        
        class TestIndicator(BaseIndicator):
            def compute(self, data, **kwargs):
                return pd.DataFrame({"test_feature": data["close"].pct_change()})
        
        indicator = TestIndicator()
        features = indicator.compute(sample_data)
        assert "test_feature" in features.columns
    
    def test_feature_engineer(self, sample_data):
        """Test feature engineer."""
        from src.feature_engineering.engineer import FeatureEngineer
        
        engineer = FeatureEngineer()
        
        # Test basic feature computation
        features = engineer.compute_features(sample_data)
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_data)
        
        # Test with specific indicators
        features_subset = engineer.compute_features(
            sample_data,
            indicators=["MACD", "RSI"]
        )
        assert len(features_subset.columns) < len(features.columns)
    
    def test_indicator_registry(self):
        """Test indicator registry."""
        from src.feature_engineering.registry import IndicatorRegistry
        
        registry = IndicatorRegistry()
        
        # Test registration
        all_indicators = registry.get_all_indicators()
        assert len(all_indicators) > 0
        
        # Test retrieval by category
        momentum = registry.get_indicators_by_category("momentum")
        assert len(momentum) > 0
        
        # Test specific indicator
        macd = registry.get_indicator("MACD")
        assert macd is not None
    
    def test_momentum_indicators(self, sample_data):
        """Test momentum indicators."""
        from src.feature_engineering.momentum.macd import MACD
        from src.feature_engineering.momentum.oscillators import RSI, Stochastic, WilliamsR
        
        # Test MACD
        macd = MACD()
        features = macd.compute(sample_data)
        assert "macd" in features.columns
        assert "macd_signal" in features.columns
        
        # Test RSI
        rsi = RSI()
        features = rsi.compute(sample_data)
        assert "rsi_14" in features.columns
        
        # Test Stochastic
        stoch = Stochastic()
        features = stoch.compute(sample_data)
        assert "stoch_k" in features.columns
        
        # Test Williams %R
        williams = WilliamsR()
        features = williams.compute(sample_data)
        assert "williams_r_14" in features.columns
    
    def test_pattern_indicators(self, sample_data):
        """Test pattern indicators."""
        from src.feature_engineering.pattern.pivots import PivotPoints
        from src.feature_engineering.pattern.psar import ParabolicSAR
        from src.feature_engineering.pattern.supertrend import SuperTrend
        from src.feature_engineering.pattern.zigzag import ZigZag
        
        # Test Pivot Points
        pivots = PivotPoints()
        features = pivots.compute(sample_data)
        assert "pivot" in features.columns
        
        # Test Parabolic SAR
        psar = ParabolicSAR()
        features = psar.compute(sample_data)
        assert "psar" in features.columns
        
        # Test SuperTrend
        supertrend = SuperTrend()
        features = supertrend.compute(sample_data)
        assert "supertrend" in features.columns
        
        # Test ZigZag
        zigzag = ZigZag()
        features = zigzag.compute(sample_data)
        assert "zigzag" in features.columns
    
    def test_volume_indicators(self, sample_data):
        """Test volume indicators."""
        from src.feature_engineering.volume.classic import OBV, ADL, MFI, VWMA
        from src.feature_engineering.volume.price_volume import PVT, EaseOfMovement
        
        # Test OBV
        obv = OBV()
        features = obv.compute(sample_data)
        assert "obv" in features.columns
        
        # Test ADL
        adl = ADL()
        features = adl.compute(sample_data)
        assert "adl" in features.columns
        
        # Test MFI
        mfi = MFI()
        features = mfi.compute(sample_data)
        assert "mfi_14" in features.columns
        
        # Test PVT
        pvt = PVT()
        features = pvt.compute(sample_data)
        assert "pvt" in features.columns


class TestMonitoring:
    """Test monitoring modules."""
    
    def test_metrics_collector(self):
        """Test metrics collector."""
        from src.monitoring.metrics_collector import MetricsCollector
        
        collector = MetricsCollector()
        
        # Record various metrics
        collector.record_prediction("BUY", 0.8, 0.05)
        collector.record_trade("BTC/USDT", "BUY", 1.0, 50000, 100)
        collector.record_latency("prediction", 0.05)
        collector.record_error("api", "timeout")
        
        # Get metrics
        metrics = collector.get_metrics()
        assert metrics["total_predictions"] == 1
        assert metrics["total_trades"] == 1
        assert metrics["total_pnl"] == 100
        
        # Get summary
        summary = collector.get_summary()
        assert summary["total_predictions"] == 1
        
        # Test reset
        collector.reset()
        metrics = collector.get_metrics()
        assert metrics["total_predictions"] == 0
    
    def test_alert_manager(self):
        """Test alert manager."""
        from src.monitoring.alert_manager import AlertManager
        
        manager = AlertManager()
        
        # Test alert checking
        metrics = {
            "current_drawdown": -0.12,
            "consecutive_losses": 6,
            "error_rate": 0.15
        }
        
        alerts = manager.check_alerts(metrics)
        assert len(alerts) == 3
        
        # Test throttling
        alert = alerts[0]
        assert manager.should_send_alert(alert)
        assert not manager.should_send_alert(alert)  # Throttled
        
        # Test alert sending
        manager.send_alert(alert, channels=["log"])
    
    def test_performance_monitor(self):
        """Test performance monitor."""
        from src.monitoring.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        # Update with returns
        returns = [0.01, -0.02, 0.015, -0.005, 0.02]
        for ret in returns:
            monitor.update(ret)
        
        # Get performance
        perf = monitor.get_performance()
        assert perf["total_returns"] == len(returns)
        assert "sharpe_ratio" in perf
        assert "max_drawdown" in perf


class TestAPI:
    """Test API modules."""
    
    def test_api_models(self):
        """Test API data models."""
        from src.api.api import PredictionRequest, BatchPredictionRequest, PredictionResponse
        
        # Test single prediction request
        req = PredictionRequest(features={"close": 50000, "volume": 10000})
        assert req.features["close"] == 50000
        
        # Test batch request
        batch = BatchPredictionRequest(samples=[
            {"close": 50000, "volume": 10000},
            {"close": 51000, "volume": 11000}
        ])
        assert len(batch.samples) == 2
        
        # Test response
        resp = PredictionResponse(
            prediction=0.8,
            confidence=0.9,
            timestamp=datetime.now().isoformat()
        )
        assert resp.prediction == 0.8
    
    def test_create_app(self):
        """Test FastAPI app creation."""
        from src.api.api import create_app
        
        app = create_app()
        assert app is not None
        assert app.title == "Bitcoin Trading API"
        
        # Check routes
        routes = [route.path for route in app.routes]
        assert "/health" in routes
        assert "/predict" in routes
        assert "/predict/batch" in routes
    
    @patch('src.api.prediction_server.load_trained_model')
    def test_prediction_server(self, mock_load_model):
        """Test prediction server."""
        from src.api.prediction_server import PredictionServer
        
        # Mock model
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([[0.7, 0.2, 0.1]]))
        mock_load_model.return_value = mock_model
        
        server = PredictionServer()
        
        # Test initialization
        assert server.risk_manager is not None
        assert server.metrics_collector is not None
        
        # Test model loading
        server.model = mock_load_model()
        assert server.model is not None


class TestUtils:
    """Test utility functions."""
    
    def test_setup_logging(self):
        """Test logging setup."""
        from src.utils import setup_logging
        
        logger = setup_logging("test_logger")
        assert logger is not None
        logger.info("Test message")
    
    def test_get_project_root(self):
        """Test project root detection."""
        from src.utils import get_project_root
        
        root = get_project_root()
        assert os.path.exists(root)
        assert os.path.isdir(root)
    
    def test_load_yaml_config(self):
        """Test YAML loading."""
        from src.utils import load_yaml_config
        
        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({"test": "value"}, f)
            temp_path = f.name
        
        try:
            config = load_yaml_config(temp_path)
            assert config["test"] == "value"
        finally:
            os.unlink(temp_path)


class TestMainModule:
    """Test main module."""
    
    @patch('sys.argv', ['main.py', '--version'])
    def test_main_version(self):
        """Test version command."""
        from src.main import main
        
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
    
    @patch('src.main.setup_logging')
    def test_main_commands(self, mock_logging):
        """Test main commands."""
        from src.main import parse_args
        
        # Test collect command
        args = parse_args(['collect', '--symbol', 'BTCUSDT'])
        assert args.command == 'collect'
        assert args.symbol == 'BTCUSDT'
        
        # Test train command
        args = parse_args(['train', '--config', 'config.yaml'])
        assert args.command == 'train'
        assert args.config == 'config.yaml'
        
        # Test serve command
        args = parse_args(['serve', '--port', '8080'])
        assert args.command == 'serve'
        assert args.port == 8080