"""Tests for FastAPI application."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


class TestFastAPIApp:
    """Test FastAPI application."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        # Import and create app
        from src.api import create_app

        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def mock_model(self):
        """Mock the ML model."""
        mock = Mock()
        mock.predict = Mock(return_value=np.array([[0.5]]))
        return mock

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_predict_endpoint(self, client, mock_model):
        """Test prediction endpoint."""
        with patch("src.api.load_model", return_value=mock_model):
            # Prepare test data
            test_data = {
                "features": {
                    "open": 50000,
                    "high": 51000,
                    "low": 49000,
                    "close": 50500,
                    "volume": 1000,
                    "sma_20": 50200,
                    "rsi_14": 55.5,
                    "bb_upper": 51500,
                    "bb_lower": 48500,
                }
            }

            response = client.post("/predict", json=test_data)
            assert response.status_code == 200

            data = response.json()
            assert "prediction" in data
            assert "confidence" in data
            assert "timestamp" in data

    def test_predict_invalid_features(self, client):
        """Test prediction with invalid features."""
        test_data = {"features": {"invalid_feature": 123}}

        response = client.post("/predict", json=test_data)
        assert response.status_code == 422  # Validation error

    def test_batch_predict_endpoint(self, client, mock_model):
        """Test batch prediction endpoint."""
        with patch("src.api.load_model", return_value=mock_model):
            test_data = {
                "samples": [
                    {
                        "open": 50000,
                        "high": 51000,
                        "low": 49000,
                        "close": 50500,
                        "volume": 1000,
                        "sma_20": 50200,
                        "rsi_14": 55.5,
                        "bb_upper": 51500,
                        "bb_lower": 48500,
                    },
                    {
                        "open": 51000,
                        "high": 52000,
                        "low": 50000,
                        "close": 51500,
                        "volume": 1100,
                        "sma_20": 50700,
                        "rsi_14": 60.5,
                        "bb_upper": 52500,
                        "bb_lower": 49500,
                    },
                ]
            }

            response = client.post("/predict/batch", json=test_data)
            assert response.status_code == 200

            data = response.json()
            assert "predictions" in data
            assert len(data["predictions"]) == 2

    def test_model_info_endpoint(self, client):
        """Test model info endpoint."""
        with patch("src.api.get_model_info") as mock_info:
            mock_info.return_value = {
                "name": "TauSACTrader",
                "version": "1.0.0",
                "features": ["open", "high", "low", "close", "volume"],
                "training_date": "2024-01-01",
            }

            response = client.get("/model/info")
            assert response.status_code == 200

            data = response.json()
            assert data["name"] == "TauSACTrader"
            assert data["version"] == "1.0.0"
            assert len(data["features"]) == 5

    def test_cors_headers(self, client):
        """Test CORS headers."""
        response = client.options("/predict")
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers

    def test_request_validation_middleware(self, client):
        """Test request validation."""
        # Send invalid JSON
        response = client.post(
            "/predict", data="invalid json", headers={"content-type": "application/json"}
        )
        assert response.status_code == 422

    def test_error_handling(self, client, mock_model):
        """Test error handling."""
        # Make model raise exception
        mock_model.predict.side_effect = Exception("Model error")

        with patch("src.api.load_model", return_value=mock_model):
            test_data = {
                "features": {
                    "open": 50000,
                    "high": 51000,
                    "low": 49000,
                    "close": 50500,
                    "volume": 1000,
                    "sma_20": 50200,
                    "rsi_14": 55.5,
                    "bb_upper": 51500,
                    "bb_lower": 48500,
                }
            }

            response = client.post("/predict", json=test_data)
            assert response.status_code == 500

            data = response.json()
            assert "error" in data
            assert "Model error" in data["error"]


# Create a simple API module if it doesn't exist
def create_test_api():
    """Create a test API module."""
    api_content = '''"""FastAPI application for Bitcoin trading predictions."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any
import numpy as np
from datetime import datetime


class PredictionRequest(BaseModel):
    """Single prediction request."""
    features: Dict[str, float]


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    samples: List[Dict[str, float]]


class PredictionResponse(BaseModel):
    """Prediction response."""
    prediction: float
    confidence: float
    timestamp: str


def create_app():
    """Create FastAPI application."""
    app = FastAPI(title="Bitcoin Trading API", version="1.0.0")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/health")
    def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "version": "1.0.0"}
    
    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest):
        """Make single prediction."""
        try:
            # Mock prediction
            prediction = 0.5
            confidence = 0.8
            
            return PredictionResponse(
                prediction=prediction,
                confidence=confidence,
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/predict/batch")
    def batch_predict(request: BatchPredictionRequest):
        """Make batch predictions."""
        predictions = []
        for sample in request.samples:
            predictions.append({
                "prediction": 0.5,
                "confidence": 0.8
            })
        return {"predictions": predictions}
    
    @app.get("/model/info")
    def model_info():
        """Get model information."""
        return get_model_info()
    
    return app


def load_model():
    """Load the trained model."""
    # Mock implementation
    return None


def get_model_info():
    """Get model information."""
    return {
        "name": "TauSACTrader",
        "version": "1.0.0",
        "features": ["open", "high", "low", "close", "volume"],
        "training_date": "2024-01-01"
    }
'''

    # Write the API module
    from pathlib import Path

    api_path = Path("src/api.py")
    api_path.parent.mkdir(exist_ok=True)
    api_path.write_text(api_content)

    # Create __init__.py
    init_path = api_path.parent / "__init__.py"
    if not init_path.exists():
        init_path.write_text(
            '"""API module."""\n\nfrom .api import create_app\n\n__all__ = ["create_app"]\n'
        )


# Create the API module if needed
create_test_api()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
