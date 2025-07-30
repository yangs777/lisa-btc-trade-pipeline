"""Tests for monitoring modules."""

from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json

import numpy as np
import pandas as pd
import pytest


class TestMetricsCollector:
    """Test metrics collector."""

    def test_metrics_collector_initialization(self) -> None:
        """Test metrics collector initialization."""
        from src.monitoring.metrics_collector import MetricsCollector
        
        collector = MetricsCollector()
        assert hasattr(collector, "record_prediction")
        assert hasattr(collector, "record_trade")
        assert hasattr(collector, "record_latency")
        assert hasattr(collector, "get_metrics")

    def test_record_prediction(self) -> None:
        """Test recording predictions."""
        from src.monitoring.metrics_collector import MetricsCollector
        
        collector = MetricsCollector()
        
        # Record some predictions
        collector.record_prediction("buy", 0.8, 0.05)
        collector.record_prediction("sell", 0.6, 0.03)
        collector.record_prediction("hold", 0.9, 0.0)
        
        metrics = collector.get_metrics()
        assert metrics["total_predictions"] == 3
        assert metrics["predictions_by_action"]["buy"] == 1
        assert metrics["predictions_by_action"]["sell"] == 1
        assert metrics["predictions_by_action"]["hold"] == 1

    def test_record_trade(self) -> None:
        """Test recording trades."""
        from src.monitoring.metrics_collector import MetricsCollector
        
        collector = MetricsCollector()
        
        # Record trades
        collector.record_trade("BTCUSDT", "buy", 0.5, 50000, 100)
        collector.record_trade("BTCUSDT", "sell", 0.5, 51000, 150)
        
        metrics = collector.get_metrics()
        assert metrics["total_trades"] == 2
        assert metrics["total_pnl"] == 150  # 150 profit from sell

    def test_record_latency(self) -> None:
        """Test recording latency."""
        from src.monitoring.metrics_collector import MetricsCollector
        
        collector = MetricsCollector()
        
        # Record latencies
        collector.record_latency("prediction", 0.1)
        collector.record_latency("prediction", 0.2)
        collector.record_latency("data_fetch", 0.5)
        
        metrics = collector.get_metrics()
        assert metrics["avg_latency"]["prediction"] == 0.15
        assert metrics["avg_latency"]["data_fetch"] == 0.5

    def test_record_error(self) -> None:
        """Test recording errors."""
        from src.monitoring.metrics_collector import MetricsCollector
        
        collector = MetricsCollector()
        
        # Record errors
        collector.record_error("prediction", "Model timeout")
        collector.record_error("prediction", "Invalid input")
        collector.record_error("data_fetch", "API error")
        
        metrics = collector.get_metrics()
        assert metrics["errors"]["prediction"] == 2
        assert metrics["errors"]["data_fetch"] == 1

    def test_get_metrics_summary(self) -> None:
        """Test getting metrics summary."""
        from src.monitoring.metrics_collector import MetricsCollector
        
        collector = MetricsCollector()
        
        # Record various metrics
        collector.record_prediction("buy", 0.8, 0.05)
        collector.record_trade("BTCUSDT", "buy", 0.5, 50000, 0)
        collector.record_trade("BTCUSDT", "sell", 0.5, 51000, 500)
        collector.record_latency("prediction", 0.1)
        
        summary = collector.get_summary()
        assert "timestamp" in summary
        assert summary["total_predictions"] == 1
        assert summary["total_trades"] == 2
        assert summary["total_pnl"] == 500
        assert summary["avg_latency_ms"]["prediction"] == 100  # 0.1s = 100ms

    def test_reset_metrics(self) -> None:
        """Test resetting metrics."""
        from src.monitoring.metrics_collector import MetricsCollector
        
        collector = MetricsCollector()
        
        # Record and reset
        collector.record_prediction("buy", 0.8, 0.05)
        collector.reset()
        
        metrics = collector.get_metrics()
        assert metrics["total_predictions"] == 0


class TestPerformanceMonitor:
    """Test performance monitor."""

    def test_performance_monitor_initialization(self) -> None:
        """Test performance monitor initialization."""
        from src.monitoring.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor(window_size=100)
        assert monitor.window_size == 100
        assert hasattr(monitor, "update")
        assert hasattr(monitor, "get_performance")

    def test_update_returns(self) -> None:
        """Test updating returns."""
        from src.monitoring.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        # Add returns
        returns = [0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.01, 0.02]
        for ret in returns:
            monitor.update(ret)
        
        perf = monitor.get_performance()
        assert perf["total_returns"] == len(returns)
        assert abs(perf["cumulative_return"] - 0.0494) < 0.001  # Compound return
        assert perf["win_rate"] == 0.625  # 5/8 positive returns

    def test_calculate_sharpe_ratio(self) -> None:
        """Test Sharpe ratio calculation."""
        from src.monitoring.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        # Add consistent returns
        for _ in range(252):  # One year of daily returns
            monitor.update(0.001)  # 0.1% daily return
        
        perf = monitor.get_performance()
        assert perf["sharpe_ratio"] > 10  # Very high due to no volatility

    def test_calculate_max_drawdown(self) -> None:
        """Test max drawdown calculation."""
        from src.monitoring.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        # Simulate drawdown
        returns = [0.05, 0.03, -0.10, -0.05, 0.02, 0.01]
        for ret in returns:
            monitor.update(ret)
        
        perf = monitor.get_performance()
        assert perf["max_drawdown"] < -0.10  # Should show significant drawdown

    def test_rolling_window(self) -> None:
        """Test rolling window behavior."""
        from src.monitoring.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor(window_size=5)
        
        # Add more than window size
        for i in range(10):
            monitor.update(0.01 * (i % 2))  # Alternating 0 and 0.01
        
        perf = monitor.get_performance()
        assert perf["total_returns"] == 5  # Only last 5 returns

    def test_empty_performance(self) -> None:
        """Test performance with no data."""
        from src.monitoring.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        perf = monitor.get_performance()
        
        assert perf["total_returns"] == 0
        assert perf["cumulative_return"] == 0
        assert perf["sharpe_ratio"] == 0
        assert perf["max_drawdown"] == 0


class TestAlertManager:
    """Test alert manager."""

    def test_alert_manager_initialization(self) -> None:
        """Test alert manager initialization."""
        from src.monitoring.alert_manager import AlertManager
        
        manager = AlertManager()
        assert hasattr(manager, "check_alerts")
        assert hasattr(manager, "send_alert")

    def test_drawdown_alert(self) -> None:
        """Test drawdown alert."""
        from src.monitoring.alert_manager import AlertManager
        
        manager = AlertManager()
        
        # Check drawdown alert
        alerts = manager.check_alerts({
            "current_drawdown": -0.15,
            "max_drawdown": -0.15
        })
        
        assert len(alerts) > 0
        assert any("drawdown" in alert["message"].lower() for alert in alerts)

    def test_loss_streak_alert(self) -> None:
        """Test loss streak alert."""
        from src.monitoring.alert_manager import AlertManager
        
        manager = AlertManager()
        
        # Check loss streak
        alerts = manager.check_alerts({
            "consecutive_losses": 6,
            "recent_trades": [{"pnl": -100} for _ in range(6)]
        })
        
        assert len(alerts) > 0
        assert any("loss" in alert["message"].lower() for alert in alerts)

    def test_api_error_alert(self) -> None:
        """Test API error rate alert."""
        from src.monitoring.alert_manager import AlertManager
        
        manager = AlertManager()
        
        # Check error rate
        alerts = manager.check_alerts({
            "error_rate": 0.15,  # 15% error rate
            "total_requests": 1000,
            "failed_requests": 150
        })
        
        assert len(alerts) > 0
        assert any("error" in alert["message"].lower() for alert in alerts)

    @patch("src.monitoring.alert_manager.send_email")
    @patch("src.monitoring.alert_manager.send_slack")
    def test_send_alert(self, mock_slack: Mock, mock_email: Mock) -> None:
        """Test sending alerts."""
        from src.monitoring.alert_manager import AlertManager
        
        manager = AlertManager()
        
        # Send alert
        alert = {
            "level": "critical",
            "message": "System down",
            "timestamp": datetime.now()
        }
        
        manager.send_alert(alert, channels=["email", "slack"])
        
        mock_email.assert_called_once()
        mock_slack.assert_called_once()

    def test_alert_throttling(self) -> None:
        """Test alert throttling."""
        from src.monitoring.alert_manager import AlertManager
        
        manager = AlertManager()
        
        # Send same alert multiple times
        alert = {
            "type": "drawdown",
            "message": "High drawdown",
            "level": "warning"
        }
        
        # First should send
        assert manager.should_send_alert(alert) is True
        
        # Second within throttle period should not
        assert manager.should_send_alert(alert) is False
        
        # After throttle period should send again
        manager.last_alert_time["drawdown"] = datetime.now() - timedelta(hours=2)
        assert manager.should_send_alert(alert) is True


# Create monitoring modules if needed
def create_monitoring_modules() -> None:
    """Create monitoring modules."""
    from pathlib import Path
    
    # Create metrics collector
    metrics_content = '''"""Metrics collection for monitoring."""

from collections import defaultdict
from datetime import datetime
from typing import Dict, Any, List


class MetricsCollector:
    """Collect and aggregate metrics."""
    
    def __init__(self):
        """Initialize collector."""
        self.predictions = []
        self.trades = []
        self.latencies = defaultdict(list)
        self.errors = defaultdict(int)
        self.start_time = datetime.now()
    
    def record_prediction(self, action: str, confidence: float, position_size: float):
        """Record a prediction."""
        self.predictions.append({
            "action": action,
            "confidence": confidence,
            "position_size": position_size,
            "timestamp": datetime.now()
        })
    
    def record_trade(self, symbol: str, side: str, size: float, price: float, pnl: float):
        """Record a trade."""
        self.trades.append({
            "symbol": symbol,
            "side": side,
            "size": size,
            "price": price,
            "pnl": pnl,
            "timestamp": datetime.now()
        })
    
    def record_latency(self, operation: str, latency: float):
        """Record operation latency."""
        self.latencies[operation].append(latency)
    
    def record_error(self, operation: str, error: str):
        """Record an error."""
        self.errors[operation] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        predictions_by_action = defaultdict(int)
        for pred in self.predictions:
            predictions_by_action[pred["action"]] += 1
        
        avg_latency = {}
        for op, latencies in self.latencies.items():
            if latencies:
                avg_latency[op] = sum(latencies) / len(latencies)
        
        total_pnl = sum(trade["pnl"] for trade in self.trades)
        
        return {
            "total_predictions": len(self.predictions),
            "predictions_by_action": dict(predictions_by_action),
            "total_trades": len(self.trades),
            "total_pnl": total_pnl,
            "avg_latency": avg_latency,
            "errors": dict(self.errors),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        metrics = self.get_metrics()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_predictions": metrics["total_predictions"],
            "total_trades": metrics["total_trades"],
            "total_pnl": metrics["total_pnl"],
            "avg_latency_ms": {k: v * 1000 for k, v in metrics["avg_latency"].items()},
            "error_count": sum(metrics["errors"].values()),
            "uptime_hours": metrics["uptime_seconds"] / 3600
        }
    
    def reset(self):
        """Reset all metrics."""
        self.predictions.clear()
        self.trades.clear()
        self.latencies.clear()
        self.errors.clear()
        self.start_time = datetime.now()
'''
    
    # Create performance monitor
    performance_content = '''"""Performance monitoring."""

from collections import deque
import numpy as np
from typing import Dict, Any


class PerformanceMonitor:
    """Monitor trading performance metrics."""
    
    def __init__(self, window_size: int = 1000):
        """Initialize monitor."""
        self.window_size = window_size
        self.returns = deque(maxlen=window_size)
        self.equity_curve = [1.0]
    
    def update(self, return_pct: float):
        """Update with new return."""
        self.returns.append(return_pct)
        
        # Update equity curve
        new_equity = self.equity_curve[-1] * (1 + return_pct)
        self.equity_curve.append(new_equity)
        
        # Limit equity curve size
        if len(self.equity_curve) > self.window_size:
            self.equity_curve = self.equity_curve[-self.window_size:]
    
    def get_performance(self) -> Dict[str, Any]:
        """Get performance metrics."""
        if not self.returns:
            return {
                "total_returns": 0,
                "cumulative_return": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0
            }
        
        returns_array = np.array(self.returns)
        
        # Calculate metrics
        cumulative_return = self.equity_curve[-1] - 1
        
        # Sharpe ratio (annualized)
        if len(returns_array) > 1:
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            sharpe = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0
        else:
            sharpe = 0
        
        # Max drawdown
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        positive_returns = returns_array > 0
        win_rate = np.mean(positive_returns) if len(returns_array) > 0 else 0
        
        return {
            "total_returns": len(self.returns),
            "cumulative_return": cumulative_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "current_equity": self.equity_curve[-1]
        }
'''
    
    # Create alert manager
    alert_content = '''"""Alert management for monitoring."""

from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class AlertManager:
    """Manage system alerts."""
    
    def __init__(self):
        """Initialize alert manager."""
        self.alert_thresholds = {
            "max_drawdown": -0.10,
            "consecutive_losses": 5,
            "error_rate": 0.10,
            "latency_ms": 1000
        }
        self.last_alert_time = {}
        self.alert_throttle_minutes = 60
    
    def check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check if any alerts should be triggered."""
        alerts = []
        
        # Check drawdown
        if "current_drawdown" in metrics:
            if metrics["current_drawdown"] <= self.alert_thresholds["max_drawdown"]:
                alerts.append({
                    "type": "drawdown",
                    "level": "critical",
                    "message": f"High drawdown detected: {metrics['current_drawdown']:.1%}",
                    "timestamp": datetime.now()
                })
        
        # Check loss streak
        if "consecutive_losses" in metrics:
            if metrics["consecutive_losses"] >= self.alert_thresholds["consecutive_losses"]:
                alerts.append({
                    "type": "loss_streak",
                    "level": "warning",
                    "message": f"Loss streak: {metrics['consecutive_losses']} consecutive losses",
                    "timestamp": datetime.now()
                })
        
        # Check error rate
        if "error_rate" in metrics:
            if metrics["error_rate"] >= self.alert_thresholds["error_rate"]:
                alerts.append({
                    "type": "error_rate",
                    "level": "critical",
                    "message": f"High error rate: {metrics['error_rate']:.1%}",
                    "timestamp": datetime.now()
                })
        
        return alerts
    
    def should_send_alert(self, alert: Dict[str, Any]) -> bool:
        """Check if alert should be sent (throttling)."""
        alert_type = alert.get("type", "unknown")
        
        # Check last alert time
        if alert_type in self.last_alert_time:
            time_since_last = datetime.now() - self.last_alert_time[alert_type]
            if time_since_last < timedelta(minutes=self.alert_throttle_minutes):
                return False
        
        # Update last alert time
        self.last_alert_time[alert_type] = datetime.now()
        return True
    
    def send_alert(self, alert: Dict[str, Any], channels: List[str] = None):
        """Send alert through specified channels."""
        if channels is None:
            channels = ["log"]
        
        # Check throttling
        if not self.should_send_alert(alert):
            return
        
        # Send through channels
        for channel in channels:
            if channel == "log":
                logger.warning(f"ALERT: {alert['message']}")
            elif channel == "email":
                send_email(alert)
            elif channel == "slack":
                send_slack(alert)


def send_email(alert: Dict[str, Any]):
    """Send email alert (placeholder)."""
    pass


def send_slack(alert: Dict[str, Any]):
    """Send Slack alert (placeholder)."""
    pass
'''
    
    # Write modules
    monitoring_dir = Path("src/monitoring")
    monitoring_dir.mkdir(exist_ok=True)
    
    (monitoring_dir / "__init__.py").write_text(
        '"""Monitoring and alerting modules."""\n\n'
        'from .metrics_collector import MetricsCollector\n'
        'from .performance_monitor import PerformanceMonitor\n'
        'from .alert_manager import AlertManager\n\n'
        '__all__ = ["MetricsCollector", "PerformanceMonitor", "AlertManager"]\n'
    )
    
    (monitoring_dir / "metrics_collector.py").write_text(metrics_content)
    (monitoring_dir / "performance_monitor.py").write_text(performance_content)
    (monitoring_dir / "alert_manager.py").write_text(alert_content)


# create_monitoring_modules()  # Commented out - modules already exist