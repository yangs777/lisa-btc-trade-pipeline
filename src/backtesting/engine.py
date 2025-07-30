"""Backtesting engine for strategy evaluation."""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional


class BacktestEngine:
    """Engine for backtesting trading strategies."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        strategy: Any,
        initial_capital: float = 100000,
        fee: float = 0.001,
        slippage: float = 0.0001
    ):
        """Initialize backtest engine.
        
        Args:
            data: Historical price data
            strategy: Trading strategy object
            initial_capital: Starting capital
            fee: Trading fee rate
            slippage: Slippage rate
        """
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.fee = fee
        self.slippage = slippage
        
        # Initialize state
        self.capital = initial_capital
        self.position = 0
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = [initial_capital]
    
    def run(self) -> Dict[str, Any]:
        """Run backtest.
        
        Returns:
            Dictionary with backtest results
        """
        # Generate signals
        signals = self.strategy.generate_signals(self.data)
        
        # Execute trades based on signals
        for i in range(len(self.data)):
            price = self.data.iloc[i]["close"]
            signal = signals[i] if i < len(signals) else 0
            
            # Execute trade if signal changed
            if signal != 0 and signal != np.sign(self.position):
                self._execute_trade(i, price, signal)
            
            # Update equity
            equity = self._calculate_equity(price)
            self.equity_curve.append(equity)
        
        # Close final position
        if self.position != 0:
            final_price = self.data.iloc[-1]["close"]
            self._execute_trade(len(self.data) - 1, final_price, 0)
        
        # Calculate metrics
        results = self._calculate_metrics()
        return results
    
    def _execute_trade(self, index: int, price: float, signal: int) -> None:
        """Execute a trade.
        
        Args:
            index: Data index
            price: Trade price
            signal: Trade signal (-1, 0, 1)
        """
        # Apply slippage
        if signal > self.position:  # Buying
            execution_price = price * (1 + self.slippage)
        else:  # Selling
            execution_price = price * (1 - self.slippage)
        
        # Calculate trade size
        if self.position != 0:
            # Close existing position
            pnl = (execution_price - self.entry_price) * self.position * self.position_size
            pnl -= abs(self.position_size) * execution_price * self.fee  # Fee
            
            self.capital += pnl
            
            self.trades.append({
                "index": index,
                "type": "close",
                "price": execution_price,
                "size": -self.position_size,
                "pnl": pnl
            })
        
        # Open new position
        if signal != 0:
            # Calculate position size
            self.position_size = (self.capital * 0.95) / execution_price  # Use 95% of capital
            self.position = signal
            self.entry_price = execution_price
            
            # Deduct fee
            self.capital -= abs(self.position_size) * execution_price * self.fee
            
            self.trades.append({
                "index": index,
                "type": "open",
                "price": execution_price,
                "size": self.position_size * signal,
                "pnl": 0
            })
        else:
            self.position = 0
    
    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current equity.
        
        Args:
            current_price: Current market price
            
        Returns:
            Current equity value
        """
        if self.position == 0:
            return self.capital
        
        # Unrealized P&L
        unrealized_pnl = (current_price - self.entry_price) * self.position * self.position_size
        return self.capital + unrealized_pnl
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        equity_array = np.array(self.equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        # Basic metrics
        total_return = (equity_array[-1] - equity_array[0]) / equity_array[0]
        
        # Risk metrics
        if len(returns) > 0:
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252 * 24)  # Hourly data
            
            # Maximum drawdown
            cumulative = equity_array / equity_array[0]
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdown)
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Trade statistics
        winning_trades = [t for t in self.trades if t["type"] == "close" and t["pnl"] > 0]
        losing_trades = [t for t in self.trades if t["type"] == "close" and t["pnl"] < 0]
        
        win_rate = len(winning_trades) / len([t for t in self.trades if t["type"] == "close"]) if self.trades else 0
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "num_trades": len([t for t in self.trades if t["type"] == "open"]),
            "final_equity": equity_array[-1],
            "equity_curve": self.equity_curve
        }