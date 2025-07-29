"""Test coverage for batch prediction module."""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from src.pipeline.batch_predict import BatchPredictor, main


class TestBatchPredictor:
    """Test BatchPredictor class."""
    
    def test_initialization(self):
        """Test predictor initialization."""
        predictor = BatchPredictor(model_path="gs://bucket/model.pkl")
        
        assert predictor.model_path == "gs://bucket/model.pkl"
        assert predictor.model is None
    
    def test_load_model(self):
        """Test model loading."""
        predictor = BatchPredictor(model_path="gs://bucket/model.pkl")
        
        # Mock logger to verify logging
        with patch('src.pipeline.batch_predict.logger') as mock_logger:
            predictor.load_model()
            
            # Verify model loaded
            assert predictor.model is not None
            assert predictor.model["type"] == "TauSAC"
            assert predictor.model["loaded"] == True
            
            # Verify logging
            mock_logger.info.assert_called_with("Loading model from gs://bucket/model.pkl")
    
    def test_load_data(self):
        """Test data loading."""
        predictor = BatchPredictor(model_path="test.pkl")
        
        with patch('src.pipeline.batch_predict.logger') as mock_logger:
            data = predictor.load_data("gs://bucket/input.json")
            
            # Verify data structure
            assert isinstance(data, list)
            assert len(data) == 3
            assert all("timestamp" in item for item in data)
            assert all("features" in item for item in data)
            assert all(len(item["features"]) == 200 for item in data)
            
            # Verify logging
            mock_logger.info.assert_called_with("Loading data from gs://bucket/input.json")
    
    def test_predict(self):
        """Test prediction generation."""
        predictor = BatchPredictor(model_path="test.pkl")
        predictor.model = {"type": "TauSAC", "loaded": True}
        
        # Create test data
        test_data = [
            {"timestamp": "2024-01-01T00:00:00", "features": [0.1] * 200},
            {"timestamp": "2024-01-01T01:00:00", "features": [0.2] * 200},
        ]
        
        predictions = predictor.predict(test_data)
        
        # Verify predictions structure
        assert len(predictions) == 2
        for i, pred in enumerate(predictions):
            assert pred["timestamp"] == test_data[i]["timestamp"]
            assert pred["action"] in [0, 1, 2]  # Buy/Hold/Sell
            assert 0 <= pred["confidence"] <= 1
            assert isinstance(pred["expected_return"], float)
            assert 0 <= pred["risk_score"] <= 1
    
    def test_save_predictions(self):
        """Test saving predictions."""
        predictor = BatchPredictor(model_path="test.pkl")
        
        predictions = [
            {
                "timestamp": "2024-01-01T00:00:00",
                "action": 1,
                "confidence": 0.8,
                "expected_return": 0.02,
                "risk_score": 0.3
            }
        ]
        
        with patch('src.pipeline.batch_predict.logger') as mock_logger:
            predictor.save_predictions(predictions, "gs://bucket/output.json")
            
            # Verify logging calls
            assert mock_logger.info.call_count == 2
            mock_logger.info.assert_any_call("Saving 1 predictions to gs://bucket/output.json")
    
    def test_run_success(self):
        """Test successful batch prediction run."""
        predictor = BatchPredictor(model_path="gs://bucket/model.pkl")
        
        # Mock methods
        predictor.load_model = Mock()
        predictor.load_data = Mock(return_value=[{"timestamp": "2024-01-01", "features": [1, 2, 3]}])
        predictor.predict = Mock(return_value=[{"timestamp": "2024-01-01", "action": 1}])
        predictor.save_predictions = Mock()
        
        result = predictor.run("input.json", "output.json")
        
        # Verify result
        assert result["status"] == "success"
        assert result["input_path"] == "input.json"
        assert result["output_path"] == "output.json"
        assert result["num_predictions"] == 1
        assert result["model_path"] == "gs://bucket/model.pkl"
        
        # Verify method calls
        predictor.load_model.assert_called_once()
        predictor.load_data.assert_called_once_with("input.json")
        predictor.predict.assert_called_once()
        predictor.save_predictions.assert_called_once()
    
    def test_run_failure(self):
        """Test batch prediction run with failure."""
        predictor = BatchPredictor(model_path="gs://bucket/model.pkl")
        
        # Mock load_model to raise exception
        predictor.load_model = Mock(side_effect=Exception("Model not found"))
        
        result = predictor.run("input.json", "output.json")
        
        # Verify failure result
        assert result["status"] == "failed"
        assert result["error"] == "Model not found"
        assert result["input_path"] == "input.json"
        assert result["output_path"] == "output.json"
    
    def test_run_with_logging(self):
        """Test run with proper logging."""
        predictor = BatchPredictor(model_path="test.pkl")
        
        with patch('src.pipeline.batch_predict.logger') as mock_logger:
            result = predictor.run("input.json", "output.json")
            
            # Should have multiple info logs
            assert mock_logger.info.call_count >= 4
            
            # Check for specific log messages
            log_messages = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any("Loading model" in msg for msg in log_messages)
            assert any("Loading data" in msg for msg in log_messages)
            assert any("Loaded 3 data points" in msg for msg in log_messages)
            assert any("Generated 3 predictions" in msg for msg in log_messages)


class TestBatchPredictMain:
    """Test main entry point."""
    
    def test_main_success(self):
        """Test successful main execution."""
        test_args = [
            "batch_predict.py",
            "--model", "gs://bucket/model.pkl",
            "--input", "gs://bucket/input.json",
            "--output", "gs://bucket/output.json"
        ]
        
        with patch('sys.argv', test_args):
            with patch('src.pipeline.batch_predict.BatchPredictor') as mock_predictor_class:
                with patch('builtins.print') as mock_print:
                    with patch('builtins.exit') as mock_exit:
                        # Setup mock
                        mock_predictor = Mock()
                        mock_predictor.run.return_value = {
                            "status": "success",
                            "num_predictions": 10
                        }
                        mock_predictor_class.return_value = mock_predictor
                        
                        # Run main
                        main()
                        
                        # Verify predictor created with correct model
                        mock_predictor_class.assert_called_once_with("gs://bucket/model.pkl")
                        
                        # Verify run called with correct args
                        mock_predictor.run.assert_called_once_with(
                            "gs://bucket/input.json",
                            "gs://bucket/output.json"
                        )
                        
                        # Verify success exit
                        mock_exit.assert_called_once_with(0)
    
    def test_main_failure(self):
        """Test main execution with failure."""
        test_args = [
            "batch_predict.py",
            "--model", "bad_model.pkl",
            "--input", "input.json",
            "--output", "output.json"
        ]
        
        with patch('sys.argv', test_args):
            with patch('src.pipeline.batch_predict.BatchPredictor') as mock_predictor_class:
                with patch('builtins.print') as mock_print:
                    with patch('builtins.exit') as mock_exit:
                        # Setup mock to fail
                        mock_predictor = Mock()
                        mock_predictor.run.return_value = {
                            "status": "failed",
                            "error": "Model not found"
                        }
                        mock_predictor_class.return_value = mock_predictor
                        
                        # Run main
                        main()
                        
                        # Verify failure exit
                        mock_exit.assert_called_once_with(1)
    
    def test_main_with_log_level(self):
        """Test main with custom log level."""
        test_args = [
            "batch_predict.py",
            "--model", "model.pkl",
            "--input", "input.json",
            "--output", "output.json",
            "--log-level", "DEBUG"
        ]
        
        with patch('sys.argv', test_args):
            with patch('src.pipeline.batch_predict.logging.basicConfig') as mock_logging:
                with patch('src.pipeline.batch_predict.BatchPredictor') as mock_predictor_class:
                    with patch('builtins.print'):
                        with patch('builtins.exit'):
                            # Setup mock
                            mock_predictor = Mock()
                            mock_predictor.run.return_value = {"status": "success"}
                            mock_predictor_class.return_value = mock_predictor
                            
                            # Run main
                            main()
                            
                            # Verify logging configured with DEBUG
                            mock_logging.assert_called_once()
                            assert mock_logging.call_args[1]["level"] == logging.DEBUG
    
    def test_main_missing_args(self):
        """Test main with missing required arguments."""
        test_args = ["batch_predict.py", "--model", "model.pkl"]
        
        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit):
                main()


# Import logging for test
import logging