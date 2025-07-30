"""Tests for FastAPI prediction server."""

from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


class TestPredictionServer:
    """Test prediction server."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.api.prediction_server import create_app
        
        app = create_app()
        return TestClient(app)

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data
        assert "health" in data

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    def test_predict_endpoint_valid(self, client):
        """Test prediction endpoint with valid data."""
        # Create valid market data
        market_data = []
        for i in range(200):
            market_data.append({
                "open": 50000 + i * 10,
                "high": 50100 + i * 10,
                "low": 49900 + i * 10,
                "close": 50050 + i * 10,
                "volume": 100 + i
            })
        
        request_data = {
            "market_data": market_data,
            "portfolio_value": 10000.0,
            "confidence": 0.6
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "action" in data
        assert data["action"] in ["buy", "sell", "hold"]
        assert "position_size" in data
        assert "confidence" in data
        assert "risk_metrics" in data
        assert "timestamp" in data

    def test_predict_endpoint_insufficient_data(self, client):
        """Test prediction endpoint with insufficient data."""
        # Only 10 candles instead of minimum 200
        market_data = []
        for i in range(10):
            market_data.append({
                "open": 50000,
                "high": 50100,
                "low": 49900,
                "close": 50050,
                "volume": 100
            })
        
        request_data = {
            "market_data": market_data,
            "portfolio_value": 10000.0,
            "confidence": 0.5
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_invalid_prices(self, client):
        """Test prediction endpoint with invalid price data."""
        market_data = []
        for i in range(200):
            market_data.append({
                "open": 50000,
                "high": 49000,  # High less than open - invalid
                "low": 49900,
                "close": 50050,
                "volume": 100
            })
        
        request_data = {
            "market_data": market_data,
            "portfolio_value": 10000.0,
            "confidence": 0.5
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_batch_predict_endpoint(self, client):
        """Test batch prediction endpoint."""
        # Create sample requests
        samples = []
        for _ in range(3):
            market_data = []
            for i in range(200):
                market_data.append({
                    "open": 50000 + i * 10,
                    "high": 50100 + i * 10,
                    "low": 49900 + i * 10,
                    "close": 50050 + i * 10,
                    "volume": 100 + i
                })
            
            samples.append({
                "market_data": market_data,
                "portfolio_value": 10000.0,
                "confidence": 0.5
            })
        
        request_data = {"samples": samples}
        
        response = client.post("/predict/batch", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 3

    def test_batch_predict_too_many_samples(self, client):
        """Test batch prediction with too many samples."""
        samples = []
        for _ in range(101):  # Max is 100
            samples.append({
                "market_data": [{"open": 50000, "high": 50100, "low": 49900, "close": 50050, "volume": 100}] * 200,
                "portfolio_value": 10000.0,
                "confidence": 0.5
            })
        
        request_data = {"samples": samples}
        
        response = client.post("/predict/batch", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_risk_analysis_endpoint(self, client):
        """Test risk analysis endpoint."""
        request_data = {
            "portfolio_value": 10000.0,
            "positions": {
                "BTCUSDT": 0.5
            },
            "recent_returns": [0.01, -0.02, 0.03, -0.01, 0.02]
        }
        
        response = client.post("/analyze/risk", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "current_drawdown" in data
        assert "max_drawdown" in data
        assert "risk_multiplier" in data
        assert "daily_pnl" in data
        assert "position_count" in data
        assert "risk_warnings" in data
        assert isinstance(data["risk_warnings"], list)

    def test_model_info_endpoint(self, client):
        """Test model info endpoint."""
        response = client.get("/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "TauSACTrader"
        assert data["version"] == "1.0.0"
        assert "features" in data
        assert "risk_parameters" in data
        assert "last_updated" in data
        
        # Check risk parameters
        risk_params = data["risk_parameters"]
        assert risk_params["max_drawdown"] == 0.10
        assert risk_params["min_edge"] == 0.02
        assert risk_params["kelly_fraction"] == 0.25

    def test_backtest_endpoint(self, client):
        """Test backtest endpoint."""
        response = client.post("/backtest/run")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Backtest started"
        assert "task_id" in data
        assert data["status"] == "running"

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "api_usage" in data
        assert "risk_metrics" in data
        assert "timestamp" in data

    def test_cors_headers(self, client):
        """Test CORS headers are set."""
        response = client.options("/predict")
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] == "*"

    def test_prediction_with_risk_rejection(self, client):
        """Test prediction when risk manager rejects position."""
        # Create market data that will trigger hold
        market_data = []
        for i in range(200):
            market_data.append({
                "open": 50000,
                "high": 50000,
                "low": 50000,
                "close": 50000,  # No price movement
                "volume": 100
            })
        
        request_data = {
            "market_data": market_data,
            "portfolio_value": 10000.0,
            "confidence": 0.1  # Low confidence
        }
        
        with patch("src.risk_management.risk_manager.RiskManager.check_new_position") as mock_check:
            mock_check.return_value = (False, 0.0, "Risk limit exceeded")
            
            response = client.post("/predict", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert data["action"] == "hold"
            assert data["position_size"] == 0.0

    def test_mock_predict_logic(self):
        """Test the mock prediction logic."""
        from src.api.prediction_server import PredictionServer
        
        server = PredictionServer()
        
        # Test buy signal
        df_buy = pd.DataFrame({
            "close": [50000, 51000]  # 2% increase
        })
        assert server._mock_predict(df_buy) == "buy"
        
        # Test sell signal
        df_sell = pd.DataFrame({
            "close": [50000, 49000]  # 2% decrease
        })
        assert server._mock_predict(df_sell) == "sell"
        
        # Test hold signal
        df_hold = pd.DataFrame({
            "close": [50000, 50200]  # 0.4% increase
        })
        assert server._mock_predict(df_hold) == "hold"
        
        # Test insufficient data
        df_short = pd.DataFrame({
            "close": [50000]
        })
        assert server._mock_predict(df_short) == "hold"

    def test_error_handling_in_predict(self, client):
        """Test error handling in prediction endpoint."""
        market_data = []
        for i in range(200):
            market_data.append({
                "open": 50000,
                "high": 50100,
                "low": 49900,
                "close": 50050,
                "volume": 100
            })
        
        request_data = {
            "market_data": market_data,
            "portfolio_value": 10000.0,
            "confidence": 0.5
        }
        
        with patch("src.features.technical_indicators.TechnicalIndicators.add_all_indicators") as mock_indicators:
            mock_indicators.side_effect = Exception("Indicator calculation failed")
            
            response = client.post("/predict", json=request_data)
            assert response.status_code == 500
            assert "Indicator calculation failed" in response.json()["detail"]

    def test_batch_predict_with_errors(self, client):
        """Test batch prediction with some errors."""
        # Create samples with one invalid
        samples = [
            {
                "market_data": [{"open": 50000, "high": 50100, "low": 49900, "close": 50050, "volume": 100}] * 200,
                "portfolio_value": 10000.0,
                "confidence": 0.5
            },
            {
                "market_data": [{"open": 50000, "high": 50100, "low": 49900, "close": 50050, "volume": 100}] * 10,  # Too few
                "portfolio_value": 10000.0,
                "confidence": 0.5
            }
        ]
        
        request_data = {"samples": samples}
        
        response = client.post("/predict/batch", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        predictions = data["predictions"]
        assert len(predictions) == 2
        
        # First should succeed
        assert "action" in predictions[0]
        
        # Second should have error
        assert "error" in predictions[1]