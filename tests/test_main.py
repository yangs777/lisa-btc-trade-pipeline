"""Test coverage for main entry point."""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import sys
import asyncio
from src.main import parse_args, main, run_collection, run_training, run_server


class TestMainModule:
    """Test main module functions."""
    
    def test_parse_args_collect(self) -> None:
        """Test parsing collect command arguments."""
        test_args = ["collect", "--symbol", "ETHUSDT", "--duration", "7200"]
        args = parse_args(test_args)
        
        assert args.command == "collect"
        assert args.symbol == "ETHUSDT"
        assert args.duration == 7200
    
    def test_parse_args_train(self) -> None:
        """Test parsing train command arguments."""
        test_args = ["train", "--config", "config.yaml", "--data", "data.csv", "--output", "model.pkl"]
        args = parse_args(test_args)
        
        assert args.command == "train"
        assert args.config == "config.yaml"
        assert args.data == "data.csv"
        assert args.output == "model.pkl"
    
    def test_parse_args_serve(self) -> None:
        """Test parsing serve command arguments."""
        test_args = ["serve", "--port", "8080", "--host", "localhost", "--workers", "4"]
        args = parse_args(test_args)
        
        assert args.command == "serve"
        assert args.port == 8080
        assert args.host == "localhost"
        assert args.workers == 4
    
    def test_parse_args_defaults(self) -> None:
        """Test default argument values."""
        # Collect defaults
        args = parse_args(["collect"])
        assert args.symbol == "BTCUSDT"
        assert args.duration == 3600
        
        # Serve defaults
        args = parse_args(["serve"])
        assert args.port == 8000
        assert args.host == "0.0.0.0"
        assert args.workers == 2
    
    def test_parse_args_version(self) -> None:
        """Test version argument."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["--version"])
        assert exc_info.value.code == 0
    
    def test_run_collection(self) -> None:
        """Test data collection execution."""
        args = Mock()
        args.symbol = "BTCUSDT"
        args.duration = 1  # Short duration for test
        
        with patch('src.main.BinanceWebSocket') as mock_ws_class:
            with patch('src.main.GCSUploader') as mock_uploader_class:
                with patch('src.main.asyncio.run') as mock_asyncio_run:
                    # Setup mocks
                    mock_ws = AsyncMock()
                    mock_ws.connect = AsyncMock()
                    mock_ws.disconnect = AsyncMock()
                    mock_ws.get_orderbook_update = AsyncMock(return_value={"bid": 50000})
                    mock_ws_class.return_value = mock_ws
                    
                    mock_uploader = Mock()
                    mock_uploader.upload_json = Mock()
                    mock_uploader_class.return_value = mock_uploader
                    
                    # Run collection
                    run_collection(args)
                    
                    # Verify execution
                    mock_asyncio_run.assert_called_once()
                    mock_ws_class.assert_called_once_with("btcusdt")
                    mock_uploader_class.assert_called_once_with("btc-orderbook-data")
    
    def test_run_training(self) -> None:
        """Test model training execution."""
        args = Mock()
        args.config = "test_config.yaml"
        args.data = None
        args.output = "model.pkl"
        
        config_content = """
learning_rate: 0.001
batch_size: 128
n_epochs: 10
"""
        
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = config_content
            
            with patch('src.main.yaml.safe_load') as mock_yaml:
                mock_yaml.return_value = {
                    "learning_rate": 0.001,
                    "batch_size": 128,
                    "n_epochs": 10
                }
                
                with patch('src.main.TauSACTrainer') as mock_trainer_class:
                    mock_trainer = Mock()
                    mock_trainer.train = Mock()
                    mock_trainer.save_model = Mock()
                    mock_trainer_class.return_value = mock_trainer
                    
                    # Run training
                    run_training(args)
                    
                    # Verify execution
                    mock_open.assert_called_once_with("test_config.yaml", 'r')
                    mock_trainer.train.assert_called_once()
                    mock_trainer.save_model.assert_called_once_with("model.pkl")
    
    def test_run_server(self) -> None:
        """Test server execution."""
        args = Mock()
        args.host = "localhost"
        args.port = 8080
        args.workers = 2
        
        with patch('src.main.uvicorn') as mock_uvicorn:
            with patch('src.main.create_app') as mock_create_app:
                mock_app = Mock()
                mock_create_app.return_value = mock_app
                
                # Run server
                run_server(args)
                
                # Verify execution
                mock_create_app.assert_called_once()
                mock_uvicorn.run.assert_called_once_with(
                    mock_app,
                    host="localhost",
                    port=8080,
                    workers=2,
                    log_level="info"
                )
    
    def test_main_no_command(self) -> None:
        """Test main with no command."""
        with patch('sys.argv', ['main.py']):
            with patch('src.main.logger') as mock_logger:
                with pytest.raises(SystemExit) as exc_info:
                    main()
                
                assert exc_info.value.code == 1
                mock_logger.error.assert_called_with("No command specified. Use --help for usage.")
    
    def test_main_collect_command(self) -> None:
        """Test main with collect command."""
        with patch('sys.argv', ['main.py', 'collect']):
            with patch('src.main.run_collection') as mock_run:
                main()
                mock_run.assert_called_once()
    
    def test_main_train_command(self) -> None:
        """Test main with train command."""
        with patch('sys.argv', ['main.py', 'train', '--config', 'config.yaml']):
            with patch('src.main.run_training') as mock_run:
                main()
                mock_run.assert_called_once()
    
    def test_main_serve_command(self) -> None:
        """Test main with serve command."""
        with patch('sys.argv', ['main.py', 'serve']):
            with patch('src.main.run_server') as mock_run:
                main()
                mock_run.assert_called_once()
    
    def test_main_unknown_command(self) -> None:
        """Test main with unknown command."""
        with patch('sys.argv', ['main.py', 'unknown']):
            with patch('src.main.logger') as mock_logger:
                with pytest.raises(SystemExit) as exc_info:
                    main()
                
                assert exc_info.value.code == 1
                mock_logger.error.assert_called_with("Unknown command: unknown")
    
    def test_main_keyboard_interrupt(self) -> None:
        """Test main with keyboard interrupt."""
        with patch('sys.argv', ['main.py', 'collect']):
            with patch('src.main.run_collection') as mock_run:
                mock_run.side_effect = KeyboardInterrupt()
                
                with patch('src.main.logger') as mock_logger:
                    with pytest.raises(SystemExit) as exc_info:
                        main()
                    
                    assert exc_info.value.code == 0
                    mock_logger.info.assert_called_with("Interrupted by user")
    
    def test_main_general_exception(self) -> None:
        """Test main with general exception."""
        with patch('sys.argv', ['main.py', 'train', '--config', 'config.yaml']):
            with patch('src.main.run_training') as mock_run:
                mock_run.side_effect = Exception("Test error")
                
                with patch('src.main.logger') as mock_logger:
                    with pytest.raises(SystemExit) as exc_info:
                        main()
                    
                    assert exc_info.value.code == 1
                    mock_logger.error.assert_called_with("Error: Test error")
EOF < /dev/null
