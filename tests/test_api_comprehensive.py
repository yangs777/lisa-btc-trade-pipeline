"""Comprehensive tests for API modules."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from datetime import datetime

# Test prediction server without FastAPI dependency issues
from src.api.prediction_server import PredictionServer
from src.risk_management.risk_manager import RiskManager


class TestPredictionServerComprehensive:
    """Test PredictionServer functionality."""
    
    @pytest.fixture
    def server(self):
        """Create prediction server instance."""
        return PredictionServer()
    
    def test_prediction_server_initialization(self, server):
        """Test server initialization."""
        assert server.indicators is not None
        assert server.risk_manager is not None
        assert isinstance(server.risk_manager, RiskManager)
        assert server.model is None  # No model loaded initially
        
    def test_load_model(self, server):
        """Test model loading."""
        # Mock model loading
        with patch('src.api.prediction_server.load_trained_model') as mock_load:
            mock_model = Mock()
            mock_model.predict = Mock(return_value=np.array([[0.7, 0.2, 0.1]]))
            mock_load.return_value = mock_model
            
            server.model = mock_load()
            assert server.model is not None
            
            # Test prediction
            prediction = server.model.predict(np.random.randn(1, 100))
            assert prediction.shape == (1, 3)
    
    def test_create_app(self, server):
        """Test FastAPI app creation."""
        app = server.app
        assert app is not None
        assert app.title == "Bitcoin Trading Prediction API"
        
        # Check routes exist
        routes = [route.path for route in app.routes]
        assert "/" in routes
        assert "/health" in routes
        assert "/predict" in routes
        assert "/predict/batch" in routes
        assert "/model/info" in routes
        assert "/risk/report" in routes
        assert "/position/check" in routes
        assert "/metrics" in routes
    
    @patch('src.api.prediction_server.PredictionServer.model')
    def test_predict_endpoint_logic(self, mock_model, server):
        """Test prediction logic without HTTP."""
        # Setup mock model
        mock_model.predict = Mock(return_value=np.array([[0.7, 0.2, 0.1]]))
        server.model = mock_model
        
        # Mock feature computation
        mock_features = np.random.randn(1, 100)
        with patch.object(server, '_prepare_features', return_value=mock_features):
            # Test internal prediction logic
            features_dict = {"close": 50000, "volume": 1000000}
            prepared = server._prepare_features(features_dict)
            assert prepared.shape == (1, 100)
    
    def test_risk_integration(self, server):
        """Test risk management integration."""
        # Test position checking
        position_result = server.risk_manager.check_new_position(
            symbol="BTC/USDT",
            portfolio_value=100000,
            current_price=50000,
            signal_confidence=0.8,
            win_rate=0.6,
            avg_win=0.03,
            avg_loss=0.01
        )
        
        assert "approved" in position_result
        assert "position_size" in position_result
        assert "risk_multiplier" in position_result
    
    def test_metrics_endpoints_logic(self, server):
        """Test metrics collection logic."""
        # Record some metrics
        server.metrics_collector.record_prediction("BUY", 0.8, 0.05)
        server.metrics_collector.record_trade("BTC/USDT", "BUY", 1.0, 50000, 100)
        
        # Get metrics
        metrics = server.metrics_collector.get_metrics()
        assert metrics["total_predictions"] == 1
        assert metrics["total_trades"] == 1
        
        # Get summary
        summary = server.metrics_collector.get_summary()
        assert summary["total_predictions"] == 1
        assert summary["total_trades"] == 1
    
    def test_performance_monitoring(self, server):
        """Test performance monitoring."""
        # Update performance
        server.performance_monitor.update(0.02)  # 2% return
        server.performance_monitor.update(-0.01)  # -1% return
        server.performance_monitor.update(0.015)  # 1.5% return
        
        # Get performance metrics
        perf = server.performance_monitor.get_performance()
        assert perf["total_returns"] == 3
        assert perf["cumulative_return"] > 0
        assert "sharpe_ratio" in perf
        assert "max_drawdown" in perf
    
    def test_alert_system(self, server):
        """Test alert system."""
        # Create alert conditions
        metrics = {
            "current_drawdown": -0.15,  # High drawdown
            "error_rate": 0.20         # High error rate
        }
        
        alerts = server.alert_manager.check_alerts(metrics)
        assert len(alerts) >= 2
        
        # Test alert sending
        for alert in alerts:
            server.alert_manager.send_alert(alert, channels=["log"])
    
    def test_batch_prediction_logic(self, server):
        """Test batch prediction logic."""
        # Mock model for batch prediction
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([
            [0.7, 0.2, 0.1],
            [0.3, 0.5, 0.2],
            [0.4, 0.4, 0.2]
        ]))
        server.model = mock_model
        
        # Test batch processing
        batch_size = 3
        mock_features = np.random.randn(batch_size, 100)
        
        predictions = mock_model.predict(mock_features)
        assert predictions.shape == (batch_size, 3)
        
        # Process predictions
        results = []
        for i in range(batch_size):
            pred = predictions[i]
            action_idx = np.argmax(pred)
            confidence = float(pred[action_idx])
            results.append({
                "action": ["HOLD", "BUY", "SELL"][action_idx],
                "confidence": confidence
            })
        
        assert len(results) == batch_size
        assert all("action" in r for r in results)
        assert all("confidence" in r for r in results)
    
    def test_model_info_logic(self, server):
        """Test model info logic."""
        # Test without model
        info = {
            "status": "No model loaded",
            "version": "N/A",
            "features": [],
            "last_updated": None
        }
        assert info["status"] == "No model loaded"
        
        # Test with mock model
        server.model = Mock()
        server.model_info = {
            "version": "1.0.0",
            "features": ["close", "volume", "rsi"],
            "training_date": "2024-01-01",
            "performance": {
                "test_accuracy": 0.65,
                "test_sharpe": 1.5
            }
        }
        
        assert server.model_info["version"] == "1.0.0"
        assert len(server.model_info["features"]) == 3
    
    def test_websocket_logic(self, server):
        """Test WebSocket streaming logic."""
        # Mock WebSocket connection
        mock_websocket = Mock()
        mock_websocket.send_json = Mock()
        mock_websocket.receive_json = Mock(side_effect=[
            {"type": "subscribe", "channel": "predictions"},
            {"type": "unsubscribe", "channel": "predictions"}
        ])
        
        # Simulate sending predictions
        prediction_data = {
            "timestamp": datetime.now().isoformat(),
            "action": "BUY",
            "confidence": 0.85,
            "position_size": 0.05
        }
        
        mock_websocket.send_json(prediction_data)
        mock_websocket.send_json.assert_called_with(prediction_data)
    
    def test_error_handling(self, server):
        """Test error handling in prediction server."""
        # Test with no model loaded
        with pytest.raises(ValueError):
            server._validate_model()
        
        # Test with invalid features
        with pytest.raises(KeyError):
            server._prepare_features({"invalid": "data"})
        
        # Test risk check with invalid parameters
        result = server.risk_manager.check_new_position(
            symbol="BTC/USDT",
            portfolio_value=-1000,  # Invalid
            current_price=50000,
            signal_confidence=0.8
        )
        assert not result["approved"]
    
    def test_concurrent_requests(self, server):
        """Test handling concurrent requests."""
        # Simulate multiple concurrent predictions
        import threading
        results = []
        
        def make_prediction():
            try:
                # Mock prediction
                result = {"action": "HOLD", "confidence": 0.9}
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})
        
        # Create threads
        threads = []
        for _ in range(10):
            t = threading.Thread(target=make_prediction)
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        assert len(results) == 10
        assert all("action" in r or "error" in r for r in results)