"""Tests for main entry point."""

from unittest.mock import patch, MagicMock
import sys

import pytest


class TestMain:
    """Test main entry point."""

    def test_main_help(self):
        """Test main help command."""
        from src.main import main
        
        with patch("sys.argv", ["main.py", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_main_version(self):
        """Test main version command."""
        from src.main import main
        
        with patch("sys.argv", ["main.py", "--version"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    @patch("src.main.subprocess.run")
    def test_collect_command(self, mock_run):
        """Test collect command."""
        from src.main import main
        
        mock_run.return_value = MagicMock(returncode=0)
        
        with patch("sys.argv", ["main.py", "collect", "--symbol", "BTCUSDT"]):
            result = main()
            assert result == 0
        
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "python" in call_args[0]
        assert "data_collection/binance_websocket.py" in call_args[1]

    @patch("src.main.subprocess.run")
    def test_preprocess_command(self, mock_run):
        """Test preprocess command."""
        from src.main import main
        
        mock_run.return_value = MagicMock(returncode=0)
        
        with patch("sys.argv", ["main.py", "preprocess", "--date", "2024-01-01"]):
            result = main()
            assert result == 0
        
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "python" in call_args[0]
        assert "data_processing/daily_preprocessor.py" in call_args[1]

    @patch("src.main.subprocess.run")
    def test_train_command(self, mock_run):
        """Test train command."""
        from src.main import main
        
        mock_run.return_value = MagicMock(returncode=0)
        
        with patch("sys.argv", ["main.py", "train", "--config", "test.yaml"]):
            result = main()
            assert result == 0
        
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "python" in call_args[0]
        assert "rl/train.py" in call_args[1]

    @patch("src.main.subprocess.run")
    def test_serve_command(self, mock_run):
        """Test serve command."""
        from src.main import main
        
        mock_run.return_value = MagicMock(returncode=0)
        
        with patch("sys.argv", ["main.py", "serve", "--port", "8080"]):
            result = main()
            assert result == 0
        
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "uvicorn" in call_args[0]
        assert "src.api.prediction_server:create_app" in call_args[1]

    @patch("src.main.subprocess.run")
    def test_backtest_command(self, mock_run):
        """Test backtest command."""
        from src.main import main
        
        mock_run.return_value = MagicMock(returncode=0)
        
        with patch("sys.argv", ["main.py", "backtest", "--model", "model.pt"]):
            result = main()
            assert result == 0
        
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "python" in call_args[0]
        assert "rl/backtest.py" in call_args[1]

    def test_unknown_command(self):
        """Test unknown command."""
        from src.main import main
        
        with patch("sys.argv", ["main.py", "unknown"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2

    @patch("src.main.subprocess.run")
    def test_command_failure(self, mock_run):
        """Test command failure handling."""
        from src.main import main
        
        mock_run.return_value = MagicMock(returncode=1)
        
        with patch("sys.argv", ["main.py", "collect"]):
            result = main()
            assert result == 1

    def test_no_command(self):
        """Test no command provided."""
        from src.main import main
        
        with patch("sys.argv", ["main.py"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2

    @patch("src.main.subprocess.run")
    def test_collect_with_all_args(self, mock_run):
        """Test collect command with all arguments."""
        from src.main import main
        
        mock_run.return_value = MagicMock(returncode=0)
        
        with patch("sys.argv", [
            "main.py", "collect",
            "--symbol", "ETHUSDT",
            "--interval", "5m",
            "--duration", "3600"
        ]):
            result = main()
            assert result == 0
        
        call_args = mock_run.call_args[0][0]
        assert "--symbol" in call_args
        assert "ETHUSDT" in call_args
        assert "--interval" in call_args
        assert "5m" in call_args

    @patch("src.main.subprocess.run")
    def test_train_with_all_args(self, mock_run):
        """Test train command with all arguments."""
        from src.main import main
        
        mock_run.return_value = MagicMock(returncode=0)
        
        with patch("sys.argv", [
            "main.py", "train",
            "--config", "config.yaml",
            "--epochs", "100",
            "--batch-size", "64",
            "--lr", "0.001"
        ]):
            result = main()
            assert result == 0
        
        call_args = mock_run.call_args[0][0]
        assert "--config" in call_args
        assert "--epochs" in call_args
        assert "--batch-size" in call_args

    @patch("src.main.subprocess.run")
    def test_serve_with_custom_host(self, mock_run):
        """Test serve command with custom host."""
        from src.main import main
        
        mock_run.return_value = MagicMock(returncode=0)
        
        with patch("sys.argv", [
            "main.py", "serve",
            "--host", "127.0.0.1",
            "--port", "9000",
            "--workers", "4"
        ]):
            result = main()
            assert result == 0
        
        call_args = mock_run.call_args[0][0]
        assert "--host" in call_args
        assert "127.0.0.1" in call_args
        assert "--port" in call_args
        assert "9000" in call_args

    def test_main_as_module(self):
        """Test running main as module."""
        import src.main
        
        with patch("sys.argv", ["main.py", "--help"]):
            with patch.object(src.main, "__name__", "__main__"):
                with pytest.raises(SystemExit):
                    exec(compile(open("src/main.py").read(), "src/main.py", "exec"))