"""Test coverage for pipeline integration module."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List
import numpy as np
from datetime import datetime

from src.pipeline.integration import run_live_trading, PipelineOrchestrator


class TestPipelineIntegration:
    """Test pipeline integration functions."""
    
    @pytest.mark.asyncio
    async def test_run_live_trading_success(self) -> None:
        """Test successful live trading run."""
        # Mock dependencies
        with patch('src.pipeline.integration.BinanceWebSocket') as mock_ws_class:
            with patch('src.pipeline.integration.TauSAC') as mock_model_class:
                # Setup websocket mock
                mock_ws = AsyncMock()
                mock_ws.connect = AsyncMock()
                mock_ws.disconnect = AsyncMock()
                mock_ws.get_orderbook_update = AsyncMock(return_value={"bid": 50000, "ask": 50100})
                mock_ws_class.return_value = mock_ws
                
                # Setup model mock
                mock_model = Mock()
                mock_model.predict = Mock(return_value=np.array([0.8, 0.1, 0.1]))
                mock_model_class.return_value = mock_model
                
                # Run with short duration
                result = await run_live_trading(
                    symbol="BTCUSDT",
                    duration_seconds=0.1,
                    model_path=None
                )
                
                # Verify result
                assert result["status"] == "completed"
                assert "predictions_made" in result
                assert "data_collected" in result
                assert isinstance(result["errors"], list)
                
                # Verify calls
                mock_ws.connect.assert_called_once()
                mock_ws.disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_live_trading_with_model_path(self) -> None:
        """Test live trading with model path."""
        with patch('src.pipeline.integration.BinanceWebSocket') as mock_ws_class:
            with patch('src.pipeline.integration.TauSAC') as mock_model_class:
                # Setup mocks
                mock_ws = AsyncMock()
                mock_ws.connect = AsyncMock()
                mock_ws.disconnect = AsyncMock()
                mock_ws.get_orderbook_update = AsyncMock(return_value={"bid": 50000})
                mock_ws_class.return_value = mock_ws
                
                mock_model_class.load = Mock(return_value=Mock())
                
                # Run with model path
                result = await run_live_trading(
                    symbol="BTCUSDT",
                    duration_seconds=0.1,
                    model_path="model.pkl"
                )
                
                # Verify model loading
                mock_model_class.load.assert_called_once_with("model.pkl")
                assert result["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_run_live_trading_error(self) -> None:
        """Test live trading with error."""
        with patch('src.pipeline.integration.BinanceWebSocket') as mock_ws_class:
            # Setup websocket to raise error
            mock_ws = AsyncMock()
            mock_ws.connect = AsyncMock(side_effect=Exception("Connection failed"))
            mock_ws_class.return_value = mock_ws
            
            # Run
            result = await run_live_trading(duration_seconds=0.1)
            
            # Verify error handling
            assert result["status"] == "failed"
            assert len(result["errors"]) > 0
            assert "Connection failed" in result["errors"][0]
    
    @pytest.mark.asyncio
    async def test_run_live_trading_data_collection(self) -> None:
        """Test data collection during live trading."""
        with patch('src.pipeline.integration.BinanceWebSocket') as mock_ws_class:
            with patch('src.pipeline.integration.TauSAC') as mock_model_class:
                # Setup mocks
                mock_ws = AsyncMock()
                mock_ws.connect = AsyncMock()
                mock_ws.disconnect = AsyncMock()
                
                # Return data on first call, then None
                data_sequence = [{"bid": 50000}, {"bid": 50100}, None]
                mock_ws.get_orderbook_update = AsyncMock(side_effect=data_sequence)
                mock_ws_class.return_value = mock_ws
                
                # Run
                result = await run_live_trading(duration_seconds=0.1)
                
                # Verify data collection
                assert result["data_collected"] >= 2  # At least 2 data points
                assert result["predictions_made"] >= 2


class TestPipelineOrchestrator:
    """Test PipelineOrchestrator class."""
    
    def test_orchestrator_initialization(self) -> None:
        """Test orchestrator initialization."""
        config = {
            "symbol": "BTCUSDT",
            "bucket": "test-bucket",
            "model": {
                "observation_dim": 100,
                "action_dim": 3,
                "tau_values": [3, 6, 9]
            }
        }
        
        orchestrator = PipelineOrchestrator(config)
        
        assert orchestrator.config == config
        assert orchestrator.components == {}
    
    def test_setup_data_collection(self) -> None:
        """Test data collection setup."""
        with patch('src.pipeline.integration.BinanceWebSocket') as mock_ws_class:
            with patch('src.pipeline.integration.GCSUploader') as mock_uploader_class:
                config = {"symbol": "BTCUSDT", "bucket": "test-bucket"}
                orchestrator = PipelineOrchestrator(config)
                
                orchestrator.setup_data_collection()
                
                # Verify components created
                assert "websocket" in orchestrator.components
                assert "uploader" in orchestrator.components
                
                # Verify initialization
                mock_ws_class.assert_called_once_with("btcusdt")
                mock_uploader_class.assert_called_once_with("test-bucket")
    
    def test_setup_feature_engineering(self) -> None:
        """Test feature engineering setup."""
        with patch('src.pipeline.integration.FeatureEngineer') as mock_fe_class:
            orchestrator = PipelineOrchestrator({})
            
            orchestrator.setup_feature_engineering()
            
            # Verify component created
            assert "feature_engineer" in orchestrator.components
            mock_fe_class.assert_called_once()
    
    def test_setup_model(self) -> None:
        """Test model setup."""
        with patch('src.pipeline.integration.TauSAC') as mock_model_class:
            config = {
                "model": {
                    "observation_dim": 150,
                    "action_dim": 4,
                    "tau_values": [5, 10, 15]
                }
            }
            orchestrator = PipelineOrchestrator(config)
            
            orchestrator.setup_model()
            
            # Verify component created
            assert "model" in orchestrator.components
            
            # Verify initialization parameters
            mock_model_class.assert_called_once_with(
                observation_dim=150,
                action_dim=4,
                tau_values=[5, 10, 15]
            )
    
    def test_setup_model_defaults(self) -> None:
        """Test model setup with defaults."""
        with patch('src.pipeline.integration.TauSAC') as mock_model_class:
            orchestrator = PipelineOrchestrator({})
            
            orchestrator.setup_model()
            
            # Verify default parameters
            mock_model_class.assert_called_once_with(
                observation_dim=200,
                action_dim=3,
                tau_values=[3, 6, 9, 12]
            )
    
    def test_setup_risk_management(self) -> None:
        """Test risk management setup."""
        with patch('src.pipeline.integration.RiskManager') as mock_rm_class:
            with patch('src.pipeline.integration.KellyPositionSizer') as mock_ps_class:
                orchestrator = PipelineOrchestrator({})
                
                orchestrator.setup_risk_management()
                
                # Verify component created
                assert "risk_manager" in orchestrator.components
                
                # Verify initialization
                mock_ps_class.assert_called_once()
                mock_rm_class.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_orchestrator_run(self) -> None:
        """Test orchestrator run method."""
        with patch.object(PipelineOrchestrator, 'setup_data_collection') as mock_setup_dc:
            with patch.object(PipelineOrchestrator, 'setup_feature_engineering') as mock_setup_fe:
                with patch.object(PipelineOrchestrator, 'setup_model') as mock_setup_model:
                    with patch.object(PipelineOrchestrator, 'setup_risk_management') as mock_setup_rm:
                        # Create mock components
                        mock_ws = AsyncMock()
                        mock_ws.connect = AsyncMock()
                        mock_ws.disconnect = AsyncMock()
                        mock_ws.get_orderbook_update = AsyncMock(return_value={"bid": 50000})
                        
                        mock_fe = Mock()
                        mock_fe.compute_features = Mock(return_value=np.random.rand(200))
                        
                        mock_model = Mock()
                        mock_model.predict = Mock(return_value=np.array([0.8, 0.1, 0.1]))
                        
                        mock_rm = Mock()
                        mock_rm.check_new_position = Mock(return_value={"approved": True})
                        
                        # Setup orchestrator
                        config = {"symbol": "BTCUSDT"}
                        orchestrator = PipelineOrchestrator(config)
                        
                        # Manually set components
                        orchestrator.components = {
                            "websocket": mock_ws,
                            "feature_engineer": mock_fe,
                            "model": mock_model,
                            "risk_manager": mock_rm
                        }
                        
                        # Run
                        result = await orchestrator.run()
                        
                        # Verify setup methods called
                        mock_setup_dc.assert_called_once()
                        mock_setup_fe.assert_called_once()
                        mock_setup_model.assert_called_once()
                        mock_setup_rm.assert_called_once()
                        
                        # Verify result
                        assert result["trades_executed"] > 0
                        assert "total_pnl" in result
                        assert isinstance(result["errors"], list)
    
    @pytest.mark.asyncio
    async def test_orchestrator_run_error_handling(self) -> None:
        """Test orchestrator error handling."""
        with patch.object(PipelineOrchestrator, 'setup_data_collection') as mock_setup:
            mock_setup.side_effect = Exception("Setup failed")
            
            orchestrator = PipelineOrchestrator({})
            
            # Run should handle error
            result = await orchestrator.run()
            
            # Verify error captured
            assert len(result["errors"]) > 0
            assert "Setup failed" in result["errors"][0]


def test_module_imports() -> None:
    """Test that all required modules can be imported."""
    from src.pipeline import integration
    
    assert hasattr(integration, 'run_live_trading')
    assert hasattr(integration, 'PipelineOrchestrator')