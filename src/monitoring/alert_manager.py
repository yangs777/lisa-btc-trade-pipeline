"""Alert management for monitoring."""

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
