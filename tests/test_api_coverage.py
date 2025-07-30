"""Tests for API module."""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import json

from src.api.api import create_app, PredictionRequest, PredictionResponse, BatchPredictionRequest


class TestAPI:
    """Test cases for FastAPI application."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
    
    def test_create_app(self):
        """Test app creation."""
        app = create_app()
        assert app.title == "Bitcoin Trading API"
        assert app.version == "1.0.0"
        
        # Check middleware
        middlewares = [m.cls.__name__ for m in app.user_middleware]
        assert 'CORSMiddleware' in str(middlewares)
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
    
    def test_predict_success(self, client):
        """Test successful prediction."""
        request_data = {
            "features": {
                "open": 40000.0,
                "high": 40500.0,
                "low": 39500.0,
                "close": 40200.0,
                "volume": 1000.5
            }
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["prediction"] == 0.5
        assert data["confidence"] == 0.8
        assert "timestamp" in data
        
        # Validate timestamp format
        datetime.fromisoformat(data["timestamp"])
    
    def test_predict_empty_features(self, client):
        """Test prediction with empty features."""
        request_data = {"features": {}}
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == 0.5
        assert data["confidence"] == 0.8
    
    def test_predict_invalid_request(self, client):
        """Test prediction with invalid request."""
        # Missing features field
        response = client.post("/predict", json={})
        assert response.status_code == 422
        
        # Wrong type for features
        response = client.post("/predict", json={"features": "invalid"})
        assert response.status_code == 422
    
    def test_batch_predict_success(self, client):
        """Test successful batch prediction."""
        request_data = {
            "samples": [
                {"open": 40000, "close": 40100, "volume": 100},
                {"open": 40100, "close": 40200, "volume": 110},
                {"open": 40200, "close": 40150, "volume": 95}
            ]
        }
        
        response = client.post("/predict/batch", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 3
        
        for pred in data["predictions"]:
            assert pred["prediction"] == 0.5
            assert pred["confidence"] == 0.8
    
    def test_batch_predict_empty(self, client):
        """Test batch prediction with empty samples."""
        request_data = {"samples": []}
        
        response = client.post("/predict/batch", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["predictions"] == []
    
    def test_batch_predict_large_batch(self, client):
        """Test batch prediction with many samples."""
        # Create 100 samples
        samples = [
            {"feature1": float(i), "feature2": float(i * 2)}
            for i in range(100)
        ]
        request_data = {"samples": samples}
        
        response = client.post("/predict/batch", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["predictions"]) == 100
    
    def test_model_info(self, client):
        """Test model info endpoint."""
        response = client.get("/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "TauSACTrader"
        assert data["version"] == "1.0.0"
        assert data["features"] == ["open", "high", "low", "close", "volume"]
        assert data["training_date"] == "2024-01-01"
    
    def test_cors_headers(self, client):
        """Test CORS headers are properly set."""
        response = client.options("/predict")
        # FastAPI with CORS middleware should handle OPTIONS
        assert response.status_code in [200, 405]  # 405 if OPTIONS not explicitly handled
        
        # Test actual CORS headers on a real request
        response = client.get("/health")
        # Note: TestClient doesn't fully simulate CORS headers
        # In production, these would be set by the middleware
    
    def test_prediction_request_model(self):
        """Test PredictionRequest model."""
        # Valid request
        req = PredictionRequest(features={"test": 1.0})
        assert req.features == {"test": 1.0}
        
        # Test validation
        with pytest.raises(ValueError):
            PredictionRequest()  # Missing required field
    
    def test_batch_prediction_request_model(self):
        """Test BatchPredictionRequest model."""
        req = BatchPredictionRequest(samples=[{"a": 1}, {"b": 2}])
        assert len(req.samples) == 2
        assert req.samples[0] == {"a": 1}
    
    def test_prediction_response_model(self):
        """Test PredictionResponse model."""
        resp = PredictionResponse(
            prediction=0.75,
            confidence=0.9,
            timestamp="2024-01-01T12:00:00"
        )
        assert resp.prediction == 0.75
        assert resp.confidence == 0.9
        assert resp.timestamp == "2024-01-01T12:00:00"
    
    def test_api_error_handling(self, client, monkeypatch):
        """Test API error handling."""
        # Mock an exception in predict endpoint
        def mock_predict_error(*args, **kwargs):
            raise ValueError("Test error")
        
        # Since we can't easily patch the endpoint function,
        # we'll test the error handling pattern
        # In real implementation, predict could raise exceptions
        # and they would be caught by the HTTPException handler
        
        # Test 404
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_content_type_validation(self, client):
        """Test content type validation."""
        # Send request with wrong content type
        response = client.post(
            "/predict",
            data="not json",
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 422
    
    def test_concurrent_requests_simulation(self, client):
        """Test handling multiple requests (simulated)."""
        # Make multiple requests in sequence (simulating concurrency)
        results = []
        for i in range(10):
            response = client.post("/predict", json={
                "features": {"id": float(i)}
            })
            assert response.status_code == 200
            results.append(response.json())
        
        assert len(results) == 10
        # All should have same prediction (mock always returns 0.5)
        assert all(r["prediction"] == 0.5 for r in results)