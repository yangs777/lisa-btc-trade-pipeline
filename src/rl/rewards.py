"""Risk-Balanced Sharpe Reward (RBSR) implementation for τ-SAC trading."""

import numpy as np
from typing import Optional


class RBSRReward:
    """Risk-Balanced Sharpe Reward function for trading environment.
    
    Balances profit maximization with risk control through a modified
    Sharpe ratio that penalizes excessive volatility and drawdowns.
    """
    
    def __init__(
        self,
        lookback_window: int = 100,
        risk_free_rate: float = 0.0,
        drawdown_penalty: float = 2.0,
        volatility_penalty: float = 1.5,
        min_sharpe: float = -2.0,
        max_sharpe: float = 5.0,
    ):
        """Initialize RBSR reward calculator.
        
        Args:
            lookback_window: Number of steps for return calculation
            risk_free_rate: Risk-free rate for Sharpe calculation (annualized)
            drawdown_penalty: Penalty multiplier for drawdowns
            volatility_penalty: Penalty multiplier for excessive volatility
            min_sharpe: Minimum allowed Sharpe ratio (clipping)
            max_sharpe: Maximum allowed Sharpe ratio (clipping)
        """
        self.lookback_window = lookback_window
        self.risk_free_rate = risk_free_rate / 252 / 24  # Convert to hourly
        self.drawdown_penalty = drawdown_penalty
        self.volatility_penalty = volatility_penalty
        self.min_sharpe = min_sharpe
        self.max_sharpe = max_sharpe
        
        # State tracking
        self.returns_buffer: list[float] = []
        self.equity_curve: list[float] = []
        self.peak_equity: float = 0.0
    
    def reset(self) -> None:
        """Reset reward calculator state."""
        self.returns_buffer.clear()
        self.equity_curve.clear()
        self.peak_equity = 0.0
    
    def calculate_reward(
        self,
        current_equity: float,
        position_pnl: float,
        trade_cost: float,
        holding_period: int,
    ) -> tuple[float, dict[str, float]]:
        """Calculate RBSR reward for current step.
        
        Args:
            current_equity: Current account equity
            position_pnl: Unrealized PnL from open position
            trade_cost: Transaction costs for current step
            holding_period: Number of steps position has been held
            
        Returns:
            Tuple of (reward, metrics_dict)
        """
        # Update tracking
        self.equity_curve.append(current_equity)
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Calculate return
        if len(self.equity_curve) > 1:
            step_return = (current_equity - self.equity_curve[-2]) / self.equity_curve[-2]
        else:
            step_return = 0.0
        
        self.returns_buffer.append(step_return)
        
        # Keep buffer size limited
        if len(self.returns_buffer) > self.lookback_window:
            self.returns_buffer.pop(0)
        
        # Calculate base Sharpe ratio
        if len(self.returns_buffer) >= 20:  # Minimum samples for meaningful calculation
            returns_array = np.array(self.returns_buffer)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            
            if std_return > 0:
                sharpe_ratio = (mean_return - self.risk_free_rate) / std_return
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        # Calculate drawdown
        drawdown = 0.0
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - current_equity) / self.peak_equity
        
        # Calculate volatility penalty
        volatility_penalty = 0.0
        if len(self.returns_buffer) >= 20:
            recent_vol = np.std(self.returns_buffer[-20:])
            long_vol = np.std(self.returns_buffer)
            if long_vol > 0:
                vol_ratio = recent_vol / long_vol
                if vol_ratio > 1.5:  # Recent volatility 50% higher than average
                    volatility_penalty = (vol_ratio - 1.5) * self.volatility_penalty
        
        # Calculate holding penalty (encourage closing positions)
        holding_penalty = 0.0
        if holding_period > 100:  # Positions held too long
            holding_penalty = (holding_period - 100) * 0.001
        
        # Combine into final reward
        reward = sharpe_ratio
        reward -= drawdown * self.drawdown_penalty
        reward -= volatility_penalty
        reward -= holding_penalty
        reward -= trade_cost * 10  # Transaction cost penalty
        
        # Clip reward
        reward = np.clip(reward, self.min_sharpe, self.max_sharpe)
        
        # Prepare metrics
        metrics = {
            "sharpe_ratio": sharpe_ratio,
            "drawdown": drawdown,
            "volatility_penalty": volatility_penalty,
            "holding_penalty": holding_penalty,
            "trade_cost_penalty": trade_cost * 10,
            "final_reward": reward,
            "mean_return": mean_return if 'mean_return' in locals() else 0.0,
            "std_return": std_return if 'std_return' in locals() else 0.0,
        }
        
        return reward, metrics
    
    def get_episode_metrics(self) -> dict[str, float]:
        """Get episode-level performance metrics."""
        if len(self.equity_curve) < 2:
            return {
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
            }
        
        # Calculate total return
        total_return = (self.equity_curve[-1] - self.equity_curve[0]) / self.equity_curve[0]
        
        # Calculate max drawdown
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdowns = (running_max - equity_array) / running_max
        max_drawdown = np.max(drawdowns)
        
        # Calculate episode Sharpe
        returns = np.diff(equity_array) / equity_array[:-1]
        if len(returns) > 0 and np.std(returns) > 0:
            episode_sharpe = (np.mean(returns) - self.risk_free_rate) / np.std(returns)
        else:
            episode_sharpe = 0.0
        
        # Calculate win rate
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        if len(returns) > 0:
            win_rate = len(positive_returns) / len(returns)
        else:
            win_rate = 0.0
        
        # Calculate profit factor
        if len(negative_returns) > 0 and np.sum(np.abs(negative_returns)) > 0:
            profit_factor = np.sum(positive_returns) / np.sum(np.abs(negative_returns))
        else:
            profit_factor = 0.0 if len(positive_returns) == 0 else float('inf')
        
        return {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": episode_sharpe,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
        }