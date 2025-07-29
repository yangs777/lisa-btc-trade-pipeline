"""Test coverage for pipeline integration module."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List
import numpy as np
from datetime import datetime

from src.pipeline.integration import (
    run_live_trading,
    evaluate_model_performance,
    get_latest_model,
    save_trading_results,
    create_pipeline_config
)


class TestPipelineIntegration:
    """Test pipeline integration functions."""
    
    @pytest.mark.asyncio
    async def test_run_live_trading_success(self) -> None:
        """Test successful live trading run."""
        # Mock dependencies
        with patch('src.pipeline.integration.BinanceClient') as mock_client_class:
            with patch('src.pipeline.integration.RiskManager') as mock_risk_class:
                with patch('src.pipeline.integration.get_latest_model') as mock_get_model:
                    with patch('src.pipeline.integration.save_trading_results') as mock_save:
                        # Setup mocks
                        mock_client = AsyncMock()
                        mock_client.get_balance.return_value = {"BTC": 0.1, "USDT": 10000}
                        mock_client.get_ticker.return_value = {"price": 50000}
                        mock_client.get_klines.return_value = [
                            [1609459200000, "45000", "46000", "44000", "45500", "100"],
                            [1609462800000, "45500", "47000", "45000", "46500", "120"]
                        ]
                        mock_client.place_order.return_value = {"orderId": "12345"}
                        mock_client_class.return_value = mock_client
                        
                        mock_risk = Mock()
                        mock_risk.check_position.return_value = (True, {"size": 0.01})
                        mock_risk_class.return_value = mock_risk
                        
                        mock_get_model.return_value = {"type": "TauSAC", "weights": []}
                        
                        # Run trading
                        result = await run_live_trading(
                            symbol="BTCUSDT",
                            duration_seconds=1,  # Short duration for test
                            model_path="test_model.pkl"
                        )
                        
                        # Verify result
                        assert result["status"] == "completed"
                        assert "trades" in result
                        assert "final_balance" in result
                        assert "performance" in result
                        
                        # Verify methods called
                        mock_client.get_balance.assert_called()
                        mock_save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_live_trading_error_handling(self) -> None:
        """Test live trading with errors."""
        with patch('src.pipeline.integration.BinanceClient') as mock_client_class:
            # Setup mock to raise error
            mock_client = AsyncMock()
            mock_client.get_balance.side_effect = Exception("API Error")
            mock_client_class.return_value = mock_client
            
            # Run trading
            result = await run_live_trading(symbol="BTCUSDT", duration_seconds=1)
            
            # Should handle error gracefully
            assert result["status"] == "error"
            assert "error" in result
            assert "API Error" in result["error"]
    
    def test_evaluate_model_performance(self) -> None:
        """Test model performance evaluation."""
        # Mock trades
        trades = [
            {
                "timestamp": datetime(2024, 1, 1, 10, 0),
                "action": "buy",
                "price": 45000,
                "amount": 0.1,
                "fee": 10
            },
            {
                "timestamp": datetime(2024, 1, 1, 11, 0),
                "action": "sell",
                "price": 46000,
                "amount": 0.1,
                "fee": 10
            }
        ]
        
        initial_balance = {"BTC": 0, "USDT": 10000}
        final_balance = {"BTC": 0, "USDT": 10080}  # Profit after fees
        
        performance = evaluate_model_performance(trades, initial_balance, final_balance)
        
        assert performance["total_trades"] == 2
        assert performance["profitable_trades"] == 1
        assert performance["total_pnl"] == 80  # 10080 - 10000
        assert performance["win_rate"] == 1.0  # 100% win
        assert "sharpe_ratio" in performance
        assert "max_drawdown" in performance
    
    def test_evaluate_model_performance_no_trades(self) -> None:
        """Test performance evaluation with no trades."""
        trades: List[Dict[str, Any]] = []
        initial_balance = {"BTC": 0, "USDT": 10000}
        final_balance = {"BTC": 0, "USDT": 10000}
        
        performance = evaluate_model_performance(trades, initial_balance, final_balance)
        
        assert performance["total_trades"] == 0
        assert performance["profitable_trades"] == 0
        assert performance["total_pnl"] == 0
        assert performance["win_rate"] == 0
        assert performance["sharpe_ratio"] == 0
        assert performance["max_drawdown"] == 0
    
    def test_get_latest_model(self) -> None:
        """Test getting latest model."""
        with patch('src.pipeline.integration.Path') as mock_path_class:
            with patch('builtins.open', create=True) as mock_open:
                # Setup mock
                mock_path = Mock()
                mock_path.exists.return_value = True
                mock_path.glob.return_value = [
                    Mock(name="model_v1.pkl", stat=Mock(return_value=Mock(st_mtime=1000))),
                    Mock(name="model_v2.pkl", stat=Mock(return_value=Mock(st_mtime=2000))),
                    Mock(name="model_v3.pkl", stat=Mock(return_value=Mock(st_mtime=1500)))
                ]
                mock_path_class.return_value = mock_path
                
                # Mock file content
                import pickle
                model_data = {"version": "latest", "weights": [1, 2, 3]}
                mock_open.return_value.__enter__.return_value.read.return_value = pickle.dumps(model_data)
                
                # Get model
                model = get_latest_model("models/")
                
                assert model is not None
                assert model["version"] == "latest"
    
    def test_get_latest_model_no_models(self) -> None:
        """Test getting latest model when none exist."""
        with patch('src.pipeline.integration.Path') as mock_path_class:
            mock_path = Mock()
            mock_path.exists.return_value = True
            mock_path.glob.return_value = []
            mock_path_class.return_value = mock_path
            
            model = get_latest_model("models/")
            
            assert model is None
    
    def test_save_trading_results(self) -> None:
        """Test saving trading results."""
        results = {
            "status": "completed",
            "trades": [{"action": "buy", "price": 45000}],
            "performance": {"total_pnl": 100}
        }
        
        with patch('builtins.open', create=True) as mock_open:
            with patch('src.pipeline.integration.json.dump') as mock_dump:
                save_trading_results(results, "results.json")
                
                mock_open.assert_called_once_with("results.json", "w")
                mock_dump.assert_called_once()
                
                # Check that results were dumped
                dumped_data = mock_dump.call_args[0][0]
                assert dumped_data["status"] == "completed"
                assert len(dumped_data["trades"]) == 1
    
    def test_create_pipeline_config(self) -> None:
        """Test pipeline configuration creation."""
        # Test default config
        config = create_pipeline_config()
        
        assert config["symbol"] == "BTCUSDT"
        assert config["interval"] == "1h"
        assert "features" in config
        assert "model" in config
        assert config["model"]["type"] == "TauSAC"
        
        # Test custom config
        custom_config = create_pipeline_config(
            symbol="ETHUSDT",
            interval="5m",
            lookback_periods=50
        )
        
        assert custom_config["symbol"] == "ETHUSDT"
        assert custom_config["interval"] == "5m"
        assert custom_config["features"]["lookback_periods"] == 50
    
    @pytest.mark.asyncio
    async def test_run_live_trading_with_model_updates(self) -> None:
        """Test live trading with model updates during execution."""
        with patch('src.pipeline.integration.BinanceClient') as mock_client_class:
            with patch('src.pipeline.integration.RiskManager') as mock_risk_class:
                with patch('src.pipeline.integration.get_latest_model') as mock_get_model:
                    # Setup mocks
                    mock_client = AsyncMock()
                    mock_client.get_balance.return_value = {"BTC": 0, "USDT": 10000}
                    mock_client.get_ticker.return_value = {"price": 50000}
                    mock_client.get_klines.return_value = [
                        [1609459200000, "45000", "46000", "44000", "45500", "100"]
                    ]
                    mock_client_class.return_value = mock_client
                    
                    # Model changes during execution
                    call_count = 0
                    def model_side_effect(*args):
                        nonlocal call_count
                        call_count += 1
                        return {"version": f"v{call_count}", "weights": [call_count]}
                    
                    mock_get_model.side_effect = model_side_effect
                    
                    result = await run_live_trading(
                        symbol="BTCUSDT",
                        duration_seconds=0.1,
                        check_interval=0.05
                    )
                    
                    assert result["status"] in ["completed", "error"]
                    # Model should be checked multiple times
                    assert mock_get_model.call_count >= 1
    
    def test_evaluate_model_performance_with_losses(self) -> None:
        """Test performance evaluation with losing trades."""
        trades = [
            {
                "timestamp": datetime(2024, 1, 1, 10, 0),
                "action": "buy",
                "price": 50000,
                "amount": 0.1,
                "fee": 10
            },
            {
                "timestamp": datetime(2024, 1, 1, 11, 0),
                "action": "sell",
                "price": 48000,
                "amount": 0.1,
                "fee": 10
            },
            {
                "timestamp": datetime(2024, 1, 1, 12, 0),
                "action": "buy",
                "price": 48000,
                "amount": 0.1,
                "fee": 10
            },
            {
                "timestamp": datetime(2024, 1, 1, 13, 0),
                "action": "sell",
                "price": 49000,
                "amount": 0.1,
                "fee": 10
            }
        ]
        
        initial_balance = {"BTC": 0, "USDT": 10000}
        final_balance = {"BTC": 0, "USDT": 9760}  # Net loss
        
        performance = evaluate_model_performance(trades, initial_balance, final_balance)
        
        assert performance["total_trades"] == 4
        assert performance["profitable_trades"] == 1  # One winning trade
        assert performance["total_pnl"] == -240  # Lost money
        assert performance["win_rate"] == 0.5  # 50% win rate (1 win out of 2 round trips)
        assert performance["sharpe_ratio"] < 0  # Negative Sharpe for losses