"""Shared test fixtures and configuration."""

import pytest
from unittest.mock import MagicMock, Mock
try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None
from pathlib import Path


@pytest.fixture
def mock_gcs_client():
    """Mock Google Cloud Storage client."""
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    
    mock_client.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob
    mock_blob.upload_from_string.return_value = None
    mock_blob.download_as_text.return_value = "{}"
    
    return mock_client


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range('2024-01-01', periods=100, freq='h')
    return pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(40000, 50000, 100),
        'high': np.random.uniform(40100, 50100, 100),
        'low': np.random.uniform(39900, 49900, 100),
        'close': np.random.uniform(40000, 50000, 100),
        'volume': np.random.uniform(100, 200, 100)
    })


@pytest.fixture
def valid_ohlcv_data():
    """Generate valid OHLCV data with proper constraints."""
    n = 50
    opens = np.random.uniform(40000, 50000, n)
    closes = np.random.uniform(40000, 50000, n)
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    highs = np.maximum(opens, closes) + np.random.uniform(0, 100, n)
    lows = np.minimum(opens, closes) - np.random.uniform(0, 100, n)
    
    return pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': np.random.uniform(100, 200, n)
    })


@pytest.fixture
def mock_optuna_study():
    """Mock Optuna study for hyperparameter optimization tests."""
    mock_study = MagicMock()
    mock_trial = MagicMock()
    
    # Configure trial suggestions
    mock_trial.suggest_float.return_value = 0.01
    mock_trial.suggest_int.return_value = 64
    mock_trial.suggest_categorical.return_value = 'adam'
    
    # Configure study behavior
    mock_study.optimize.return_value = None
    mock_study.best_params = {
        'learning_rate': 0.01,
        'batch_size': 64,
        'optimizer': 'adam'
    }
    mock_study.best_value = 0.95
    mock_study.best_trial = mock_trial
    
    return mock_study


@pytest.fixture
def mock_websocket():
    """Mock websocket for Binance tests."""
    mock_ws = MagicMock()
    mock_ws.recv.return_value = '{"e":"depthUpdate","s":"BTCUSDT","b":[[40000,1.5]],"a":[[40001,2.0]]}'
    mock_ws.send.return_value = None
    mock_ws.close.return_value = None
    return mock_ws


@pytest.fixture
def test_config():
    """Test configuration dictionary."""
    return {
        'api_key': 'test_key',
        'bucket_name': 'test-bucket',
        'project_id': 'test-project',
        'symbol': 'BTCUSDT',
        'max_position_size': 0.1,
        'risk_per_trade': 0.02,
        'learning_rate': 0.001,
        'batch_size': 32
    }


@pytest.fixture(autouse=True)
def mock_heavy_imports(monkeypatch):
    """Automatically mock heavy imports for all tests."""
    # Mock gymnasium
    mock_gym = MagicMock()
    mock_gym.spaces.Box.return_value = MagicMock()
    mock_gym.spaces.Discrete.return_value = MagicMock()
    monkeypatch.setattr('gymnasium.spaces', mock_gym.spaces)
    
    # Mock torch if imported
    mock_torch = MagicMock()
    mock_torch.nn.Module = MagicMock
    mock_torch.device.return_value = 'cpu'
    monkeypatch.setattr('torch', mock_torch)
    
    # Mock requests
    mock_requests = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {'status': 'ok'}
    mock_response.status_code = 200
    mock_requests.get.return_value = mock_response
    mock_requests.post.return_value = mock_response
    monkeypatch.setattr('requests', mock_requests)


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for file operations."""
    return tmp_path