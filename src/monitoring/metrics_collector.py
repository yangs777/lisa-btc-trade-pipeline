"""Metrics collection for monitoring."""

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
