"""Test coverage for pipeline module."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from src.pipeline.integration import run_live_trading, PipelineOrchestrator
from src.pipeline.vertex_orchestrator import VertexPipelineOrchestrator


class TestRunLiveTrading:
    """Test run_live_trading function."""
    
    @pytest.mark.asyncio
    async def test_successful_trading_with_model(self):
        """Test successful live trading with model."""
        with patch('src.pipeline.integration.BinanceWebSocket') as mock_ws_class:
            with patch('src.pipeline.integration.TauSAC') as mock_model_class:
                # Setup mocks
                mock_ws = AsyncMock()
                mock_ws_class.return_value = mock_ws
                mock_ws.get_orderbook_update = AsyncMock(side_effect=[
                    {"bid": 50000, "ask": 50100},
                    {"bid": 50100, "ask": 50200},
                    None
                ])
                
                mock_model = Mock()
                mock_model_class.load.return_value = mock_model
                
                # Run function
                result = await run_live_trading(
                    symbol="BTCUSDT",
                    duration_seconds=0.1,
                    model_path="test_model.pkl"
                )
                
                # Verify
                assert result["status"] == "completed"
                assert result["data_collected"] == 2
                assert result["predictions_made"] == 2
                assert len(result["errors"]) == 0
                mock_ws.connect.assert_called_once()
                mock_ws.disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_trading_without_model(self):
        """Test live trading without model (creates default)."""
        with patch('src.pipeline.integration.BinanceWebSocket') as mock_ws_class:
            with patch('src.pipeline.integration.TauSAC') as mock_model_class:
                # Setup mocks
                mock_ws = AsyncMock()
                mock_ws_class.return_value = mock_ws
                mock_ws.get_orderbook_update = AsyncMock(return_value=None)
                
                # Run function
                result = await run_live_trading(duration_seconds=0.1)
                
                # Verify default model created
                mock_model_class.assert_called_with(
                    observation_dim=10,
                    action_dim=3,
                    tau_values=[3, 6, 9]
                )
    
    @pytest.mark.asyncio
    async def test_trading_with_errors(self):
        """Test handling errors during trading."""
        with patch('src.pipeline.integration.BinanceWebSocket') as mock_ws_class:
            # Setup mock to raise error
            mock_ws = AsyncMock()
            mock_ws_class.return_value = mock_ws
            mock_ws.get_orderbook_update = AsyncMock(
                side_effect=Exception("Connection error")
            )
            
            # Run function
            result = await run_live_trading(duration_seconds=0.1)
            
            # Verify error handling
            assert result["status"] == "completed"
            assert "Connection error" in result["errors"]
    
    @pytest.mark.asyncio
    async def test_connection_failure(self):
        """Test handling connection failure."""
        with patch('src.pipeline.integration.BinanceWebSocket') as mock_ws_class:
            # Setup mock to fail connection
            mock_ws = AsyncMock()
            mock_ws_class.return_value = mock_ws
            mock_ws.connect = AsyncMock(side_effect=Exception("Connection failed"))
            
            # Run function
            result = await run_live_trading()
            
            # Verify failure
            assert result["status"] == "failed"
            assert "Connection failed" in result["errors"]


class TestPipelineOrchestrator:
    """Test PipelineOrchestrator class."""
    
    def test_initialization(self):
        """Test orchestrator initialization."""
        config = {"symbol": "BTCUSDT", "bucket": "test-bucket"}
        orchestrator = PipelineOrchestrator(config)
        
        assert orchestrator.config == config
        assert orchestrator.components == {}
    
    def test_setup_data_collection(self):
        """Test data collection setup."""
        with patch('src.pipeline.integration.BinanceWebSocket') as mock_ws:
            with patch('src.pipeline.integration.GCSUploader') as mock_uploader:
                config = {"symbol": "btcusdt", "bucket": "test-bucket"}
                orchestrator = PipelineOrchestrator(config)
                
                orchestrator.setup_data_collection()
                
                # Verify components created
                assert "websocket" in orchestrator.components
                assert "uploader" in orchestrator.components
                mock_ws.assert_called_with("btcusdt")
                mock_uploader.assert_called_with("test-bucket")
    
    def test_setup_feature_engineering(self):
        """Test feature engineering setup."""
        with patch('src.pipeline.integration.FeatureEngineer') as mock_fe:
            orchestrator = PipelineOrchestrator({})
            
            orchestrator.setup_feature_engineering()
            
            assert "feature_engineer" in orchestrator.components
            mock_fe.assert_called_once()
    
    def test_setup_model(self):
        """Test model setup."""
        with patch('src.pipeline.integration.TauSAC') as mock_model:
            config = {
                "model": {
                    "observation_dim": 100,
                    "action_dim": 5,
                    "tau_values": [1, 2, 3]
                }
            }
            orchestrator = PipelineOrchestrator(config)
            
            orchestrator.setup_model()
            
            assert "model" in orchestrator.components
            mock_model.assert_called_with(
                observation_dim=100,
                action_dim=5,
                tau_values=[1, 2, 3]
            )
    
    def test_setup_model_defaults(self):
        """Test model setup with defaults."""
        with patch('src.pipeline.integration.TauSAC') as mock_model:
            orchestrator = PipelineOrchestrator({})
            
            orchestrator.setup_model()
            
            mock_model.assert_called_with(
                observation_dim=200,
                action_dim=3,
                tau_values=[3, 6, 9, 12]
            )
    
    def test_setup_risk_management(self):
        """Test risk management setup."""
        with patch('src.pipeline.integration.RiskManager') as mock_rm:
            with patch('src.pipeline.integration.KellyPositionSizer') as mock_kelly:
                orchestrator = PipelineOrchestrator({})
                
                orchestrator.setup_risk_management()
                
                assert "risk_manager" in orchestrator.components
                mock_kelly.assert_called_once()
                mock_rm.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_pipeline(self):
        """Test running full pipeline."""
        # Create mocks
        mock_ws = AsyncMock()
        mock_ws.get_orderbook_update = AsyncMock(side_effect=[
            {"data": "test1"},
            {"data": "test2"},
            {"data": "test3"}
        ] + [None] * 10)  # Return None after 3 data points
        
        mock_fe = Mock()
        mock_fe.compute_features = Mock(return_value={"feature": 1})
        
        mock_model = Mock()
        mock_model.predict = Mock(return_value=1)
        
        mock_rm = Mock()
        mock_rm.check_new_position = Mock(return_value={
            "approved": True,
            "position_size": 0.1
        })
        
        # Create orchestrator with mocked components
        orchestrator = PipelineOrchestrator({"symbol": "BTCUSDT"})
        orchestrator.components = {
            "websocket": mock_ws,
            "feature_engineer": mock_fe,
            "model": mock_model,
            "risk_manager": mock_rm
        }
        
        # Patch setup methods
        with patch.object(orchestrator, 'setup_data_collection'):
            with patch.object(orchestrator, 'setup_feature_engineering'):
                with patch.object(orchestrator, 'setup_model'):
                    with patch.object(orchestrator, 'setup_risk_management'):
                        result = await orchestrator.run()
        
        # Verify results
        assert result["trades_executed"] == 11  # Breaks after 10 trades
        assert len(result["errors"]) == 0
        mock_ws.connect.assert_called_once()
        mock_ws.disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_pipeline_with_risk_rejection(self):
        """Test pipeline with risk rejections."""
        # Create mocks
        mock_ws = AsyncMock()
        mock_ws.get_orderbook_update = AsyncMock(return_value={"data": "test"})
        
        mock_fe = Mock()
        mock_fe.compute_features = Mock(return_value={"feature": 1})
        
        mock_model = Mock()
        mock_model.predict = Mock(return_value=1)
        
        mock_rm = Mock()
        mock_rm.check_new_position = Mock(side_effect=[
            {"approved": False},  # First 5 rejected
            {"approved": False},
            {"approved": False},
            {"approved": False},
            {"approved": False},
            {"approved": True, "position_size": 0.1},  # Then approved
        ] * 3)
        
        # Create orchestrator
        orchestrator = PipelineOrchestrator({"symbol": "BTCUSDT"})
        orchestrator.components = {
            "websocket": mock_ws,
            "feature_engineer": mock_fe,
            "model": mock_model,
            "risk_manager": mock_rm
        }
        
        # Patch setup methods
        with patch.object(orchestrator, 'setup_data_collection'):
            with patch.object(orchestrator, 'setup_feature_engineering'):
                with patch.object(orchestrator, 'setup_model'):
                    with patch.object(orchestrator, 'setup_risk_management'):
                        result = await orchestrator.run()
        
        # Should still complete
        assert result["trades_executed"] == 11
    
    @pytest.mark.asyncio
    async def test_run_pipeline_error_handling(self):
        """Test pipeline error handling."""
        # Create mock that raises error
        mock_ws = AsyncMock()
        mock_ws.connect = AsyncMock(side_effect=Exception("Connection error"))
        
        orchestrator = PipelineOrchestrator({})
        orchestrator.components = {"websocket": mock_ws}
        
        # Patch setup methods
        with patch.object(orchestrator, 'setup_data_collection'):
            with patch.object(orchestrator, 'setup_feature_engineering'):
                with patch.object(orchestrator, 'setup_model'):
                    with patch.object(orchestrator, 'setup_risk_management'):
                        result = await orchestrator.run()
        
        # Verify error captured
        assert "Connection error" in result["errors"]
        mock_ws.disconnect.assert_called_once()


class TestVertexPipelineOrchestrator:
    """Test Vertex AI pipeline orchestrator."""
    
    def test_initialization(self):
        """Test Vertex orchestrator initialization."""
        orchestrator = VertexPipelineOrchestrator(
            project_id="test-project",
            location="us-east1"
        )
        
        assert orchestrator.project_id == "test-project"
        assert orchestrator.location == "us-east1"
    
    def test_initialization_default_location(self):
        """Test initialization with default location."""
        orchestrator = VertexPipelineOrchestrator(project_id="test-project")
        
        assert orchestrator.project_id == "test-project"
        assert orchestrator.location == "us-central1"
    
    def test_create_training_pipeline(self):
        """Test creating training pipeline spec."""
        orchestrator = VertexPipelineOrchestrator(project_id="test-project")
        
        pipeline_spec = orchestrator.create_training_pipeline(
            dataset_uri="gs://bucket/dataset",
            model_uri="gs://bucket/model",
            config={"learning_rate": 0.001}
        )
        
        # Verify structure
        assert pipeline_spec["displayName"] == "Bitcoin Trading Model Training"
        assert "pipelineSpec" in pipeline_spec
        assert "components" in pipeline_spec["pipelineSpec"]
        assert "deploymentSpec" in pipeline_spec["pipelineSpec"]
        assert "pipelineInfo" in pipeline_spec["pipelineSpec"]
        assert "root" in pipeline_spec["pipelineSpec"]
        
        # Verify components
        components = pipeline_spec["pipelineSpec"]["components"]
        assert "data-preprocessing" in components
        assert "model-training" in components
        assert "model-evaluation" in components
        
        # Verify executors
        executors = pipeline_spec["pipelineSpec"]["deploymentSpec"]["executors"]
        assert "exec-preprocess" in executors
        assert "exec-train" in executors
        assert "exec-evaluate" in executors
        
        # Verify images
        assert f"gcr.io/test-project/btc-preprocess:latest" in str(pipeline_spec)
        assert f"gcr.io/test-project/btc-train:latest" in str(pipeline_spec)
        assert f"gcr.io/test-project/btc-evaluate:latest" in str(pipeline_spec)
        
        # Verify DAG
        dag = pipeline_spec["pipelineSpec"]["root"]["dag"]["tasks"]
        assert "preprocess" in dag
        assert "train" in dag
        assert "evaluate" in dag
        assert dag["train"]["dependentTasks"] == ["preprocess"]
        assert dag["evaluate"]["dependentTasks"] == ["train"]
    
    def test_create_training_pipeline_no_config(self):
        """Test creating training pipeline without config."""
        orchestrator = VertexPipelineOrchestrator(project_id="test-project")
        
        pipeline_spec = orchestrator.create_training_pipeline(
            dataset_uri="gs://bucket/dataset",
            model_uri="gs://bucket/model"
        )
        
        # Should use empty config
        dag = pipeline_spec["pipelineSpec"]["root"]["dag"]["tasks"]
        config_value = dag["train"]["inputs"]["parameters"]["config"]["runtimeValue"]["constantValue"]["stringValue"]
        assert config_value == "{}"
    
    def test_create_batch_prediction_pipeline(self):
        """Test creating batch prediction pipeline."""
        orchestrator = VertexPipelineOrchestrator(project_id="test-project")
        
        pipeline_spec = orchestrator.create_batch_prediction_pipeline(
            model_uri="gs://bucket/model",
            input_uri="gs://bucket/input",
            output_uri="gs://bucket/output"
        )
        
        # Verify structure
        assert pipeline_spec["displayName"] == "Bitcoin Trading Batch Prediction"
        assert "pipelineSpec" in pipeline_spec
        
        # Verify components
        components = pipeline_spec["pipelineSpec"]["components"]
        assert "batch-predict" in components
        
        # Verify executor
        executors = pipeline_spec["pipelineSpec"]["deploymentSpec"]["executors"]
        assert "exec-predict" in executors
        
        # Verify container
        container = executors["exec-predict"]["container"]
        assert container["image"] == f"gcr.io/test-project/btc-predict:latest"
        assert container["command"] == ["python", "-m", "src.pipeline.batch_predict"]
        
        # Verify args
        assert "--model" in str(container["args"])
        assert "--input" in str(container["args"])
        assert "--output" in str(container["args"])