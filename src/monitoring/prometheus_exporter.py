"""Prometheus metrics exporter."""

from typing import Dict, Any
import time
from collections import defaultdict


class PrometheusExporter:
    """Export metrics in Prometheus format."""
    
    def __init__(self, port: int = 8000):
        """Initialize exporter.
        
        Args:
            port: Port to expose metrics
        """
        self.port = port
        self.metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "value": 0,
            "labels": {},
            "type": "counter"
        })
        self.start_time = time.time()
    
    def record_prediction(self, action: str, confidence: float) -> None:
        """Record prediction metric.
        
        Args:
            action: Predicted action
            confidence: Prediction confidence
        """
        # Increment prediction counter
        self.metrics["prediction_count"]["value"] += 1
        
        # Record confidence histogram
        bucket = int(confidence * 10) / 10  # Round to nearest 0.1
        metric_name = f"prediction_confidence_bucket_{bucket}"
        self.metrics[metric_name]["value"] += 1
        self.metrics[metric_name]["type"] = "histogram"
    
    def record_trade(self, symbol: str, amount: float, price: float, pnl: float) -> None:
        """Record trade metric.
        
        Args:
            symbol: Trading symbol
            amount: Trade amount
            price: Trade price
            pnl: Profit/loss
        """
        self.metrics["trade_count"]["value"] += 1
        self.metrics["trade_volume"]["value"] += abs(amount * price)
        self.metrics["trade_volume"]["type"] = "gauge"
        
        if pnl > 0:
            self.metrics["profitable_trades"]["value"] += 1
        else:
            self.metrics["losing_trades"]["value"] += 1
    
    def record_latency(self, operation: str, latency_seconds: float) -> None:
        """Record latency metric.
        
        Args:
            operation: Operation name
            latency_seconds: Latency in seconds
        """
        metric_name = f"latency_seconds_{operation}"
        
        # Update sum and count for average calculation
        if metric_name not in self.metrics:
            self.metrics[metric_name] = {
                "sum": 0,
                "count": 0,
                "type": "summary"
            }
        
        self.metrics[metric_name]["sum"] += latency_seconds
        self.metrics[metric_name]["count"] += 1
    
    def get_metrics_string(self) -> str:
        """Get metrics in Prometheus format.
        
        Returns:
            Metrics string
        """
        lines = []
        
        # Add header
        lines.append("# HELP btc_trading Bitcoin trading system metrics")
        lines.append("# TYPE btc_trading gauge")
        
        # Add uptime
        uptime = time.time() - self.start_time
        lines.append(f"btc_trading_uptime_seconds {uptime}")
        
        # Add all metrics
        for metric_name, metric_data in self.metrics.items():
            if metric_data["type"] == "counter":
                lines.append(f"# TYPE {metric_name} counter")
                lines.append(f"{metric_name} {metric_data['value']}")
            elif metric_data["type"] == "gauge":
                lines.append(f"# TYPE {metric_name} gauge")
                lines.append(f"{metric_name} {metric_data['value']}")
            elif metric_data["type"] == "histogram":
                lines.append(f"# TYPE {metric_name} histogram")
                lines.append(f"{metric_name} {metric_data['value']}")
            elif metric_data["type"] == "summary":
                lines.append(f"# TYPE {metric_name} summary")
                avg = metric_data["sum"] / metric_data["count"] if metric_data["count"] > 0 else 0
                lines.append(f"{metric_name}_sum {metric_data['sum']}")
                lines.append(f"{metric_name}_count {metric_data['count']}")
                lines.append(f"{metric_name}_avg {avg}")
        
        return "\n".join(lines)
    
    def start_server(self) -> None:
        """Start Prometheus metrics server."""
        # Mock implementation - would use prometheus_client library
        print(f"Prometheus metrics server started on port {self.port}")
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.start_time = time.time()