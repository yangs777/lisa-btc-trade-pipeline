"""Performance monitoring."""

from collections import deque
import numpy as np
from typing import Dict, Any


class PerformanceMonitor:
    """Monitor trading performance metrics."""
    
    def __init__(self, window_size: int = 1000):
        """Initialize monitor."""
        self.window_size = window_size
        self.returns: deque[float] = deque(maxlen=window_size)
        self.equity_curve = [1.0]
    
    def update(self, return_pct: float) -> None:
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
