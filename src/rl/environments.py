"""Trading environment implementation for τ-SAC reinforcement learning."""

import numpy as np
import pandas as pd
from typing import Any, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces

from .rewards import RBSRReward


class BTCTradingEnvironment(gym.Env):
    """Bitcoin trading environment for τ-SAC agent training.
    
    Simulates realistic trading conditions including:
    - Transaction costs (maker/taker fees)
    - Slippage based on order size
    - Position limits
    - Margin requirements
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 100000.0,
        max_position_size: float = 0.95,  # Max 95% of balance
        maker_fee: float = 0.0002,  # 0.02%
        taker_fee: float = 0.0004,  # 0.04%
        slippage_factor: float = 0.0001,  # 0.01% per unit size
        leverage: float = 1.0,  # No leverage by default
        lookback_window: int = 100,
        render_mode: Optional[str] = None,
    ):
        """Initialize trading environment.
        
        Args:
            df: DataFrame with OHLCV + indicators
            initial_balance: Starting capital in USDT
            max_position_size: Maximum position as fraction of balance
            maker_fee: Maker fee rate
            taker_fee: Taker fee rate
            slippage_factor: Slippage per unit of position size
            leverage: Maximum allowed leverage
            lookback_window: Number of past observations for state
            render_mode: Rendering mode (human or None)
        """
        super().__init__()
        
        # Validate and prepare data
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        self.df = df.copy()
        self.prices = df['close'].values
        self.n_steps = len(df)
        
        # Trading parameters
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_factor = slippage_factor
        self.leverage = leverage
        self.lookback_window = lookback_window
        self.render_mode = render_mode
        
        # Feature columns (all except OHLCV)
        self.feature_columns = [col for col in df.columns 
                               if col not in required_cols]
        self.n_features = len(self.feature_columns) + 7  # +7 for position info
        
        # Action space: [position_size] where -1 = full short, 0 = neutral, 1 = full long
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features,),
            dtype=np.float32
        )
        
        # Initialize reward calculator
        self.reward_calculator = RBSRReward(lookback_window=lookback_window)
        
        # Episode variables
        self.reset()
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Reset indices
        self.current_step = self.lookback_window
        self.done = False
        
        # Reset account state
        self.balance = self.initial_balance
        self.position = 0.0  # BTC position size
        self.entry_price = 0.0
        self.holding_period = 0
        
        # Reset tracking
        self.trades = []
        self.equity_curve = [self.balance]
        self.reward_calculator.reset()
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one trading step."""
        if self.done:
            raise ValueError("Episode is done. Call reset() to start a new episode.")
        
        # Extract action (desired position)
        desired_position = float(action[0])
        desired_position = np.clip(desired_position, -1.0, 1.0)
        
        # Calculate actual position size based on balance
        max_btc = (self.balance * self.max_position_size) / self.prices[self.current_step]
        target_position = desired_position * max_btc
        
        # Execute trade if position change needed
        position_change = target_position - self.position
        trade_value = 0.0
        trade_cost = 0.0
        
        if abs(position_change) > 0.0001:  # Minimum trade size
            # Calculate execution price with slippage
            base_price = self.prices[self.current_step]
            slippage = self.slippage_factor * abs(position_change) / max_btc
            
            if position_change > 0:  # Buying
                exec_price = base_price * (1 + slippage)
                trade_cost = abs(position_change * exec_price * self.taker_fee)
            else:  # Selling
                exec_price = base_price * (1 - slippage)
                trade_cost = abs(position_change * exec_price * self.taker_fee)
            
            # Update balance and position
            trade_value = -position_change * exec_price
            self.balance += trade_value - trade_cost
            self.position = target_position
            
            # Track trade
            self.trades.append({
                'step': self.current_step,
                'action': desired_position,
                'position_change': position_change,
                'price': exec_price,
                'value': trade_value,
                'cost': trade_cost,
                'balance': self.balance,
            })
            
            # Update entry price and reset holding period
            if abs(self.position) > 0.0001:
                if abs(position_change) > abs(self.position) * 0.5:  # Significant change
                    self.entry_price = exec_price
                    self.holding_period = 0
            else:
                self.entry_price = 0.0
                self.holding_period = 0
        
        # Update holding period
        if abs(self.position) > 0.0001:
            self.holding_period += 1
        
        # Calculate current equity and PnL
        current_price = self.prices[self.current_step]
        position_value = self.position * current_price
        current_equity = self.balance + position_value
        self.equity_curve.append(current_equity)
        
        # Calculate unrealized PnL
        if abs(self.position) > 0.0001 and self.entry_price > 0:
            position_pnl = self.position * (current_price - self.entry_price)
        else:
            position_pnl = 0.0
        
        # Calculate reward
        reward, reward_metrics = self.reward_calculator.calculate_reward(
            current_equity=current_equity,
            position_pnl=position_pnl,
            trade_cost=trade_cost,
            holding_period=self.holding_period,
        )
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        terminated = False
        truncated = False
        
        if self.current_step >= self.n_steps - 1:
            truncated = True
        elif current_equity <= self.initial_balance * 0.2:  # 80% drawdown
            terminated = True
            reward -= 10.0  # Large penalty for blowing up
        
        self.done = terminated or truncated
        
        # Close position if episode ending
        if self.done and abs(self.position) > 0.0001:
            close_price = self.prices[self.current_step - 1]
            close_value = self.position * close_price
            close_cost = abs(close_value * self.taker_fee)
            self.balance += close_value - close_cost
            self.position = 0.0
            
            # Final equity
            final_equity = self.balance
            self.equity_curve[-1] = final_equity
        
        # Get next observation
        obs = self._get_observation()
        info = self._get_info()
        info['reward_metrics'] = reward_metrics
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        # Get current features
        idx = self.current_step
        features = self.df.iloc[idx][self.feature_columns].values
        
        # Normalize price-based features by current price
        current_price = self.prices[idx]
        
        # Add position information
        position_features = np.array([
            self.position / (self.balance / current_price) if self.balance > 0 else 0,  # Position ratio
            self.balance / self.initial_balance,  # Balance ratio
            (current_price - self.entry_price) / current_price if self.entry_price > 0 else 0,  # Unrealized PnL %
            self.holding_period / 100.0,  # Normalized holding period
            self.position > 0,  # Is long
            self.position < 0,  # Is short
            abs(self.position) < 0.0001,  # Is neutral
        ], dtype=np.float32)
        
        # Combine all features
        obs = np.concatenate([features, position_features]).astype(np.float32)
        
        # Handle any NaN values
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return obs
    
    def _get_info(self) -> dict[str, Any]:
        """Get current environment info."""
        current_equity = self.balance
        if abs(self.position) > 0.0001:
            current_equity += self.position * self.prices[self.current_step]
        
        return {
            'step': self.current_step,
            'balance': self.balance,
            'position': self.position,
            'equity': current_equity,
            'entry_price': self.entry_price,
            'current_price': self.prices[self.current_step],
            'holding_period': self.holding_period,
            'n_trades': len(self.trades),
        }
    
    def render(self) -> Optional[str]:
        """Render current state."""
        if self.render_mode == "human":
            info = self._get_info()
            output = f"\nStep: {info['step']}/{self.n_steps}"
            output += f"\nBalance: ${info['balance']:,.2f}"
            output += f"\nPosition: {info['position']:.4f} BTC"
            output += f"\nEquity: ${info['equity']:,.2f}"
            output += f"\nPrice: ${info['current_price']:,.2f}"
            if info['entry_price'] > 0:
                pnl_pct = (info['current_price'] - info['entry_price']) / info['entry_price'] * 100
                output += f"\nEntry: ${info['entry_price']:,.2f} ({pnl_pct:+.2f}%)"
            output += f"\nTrades: {info['n_trades']}"
            return output
        return None
    
    def get_episode_summary(self) -> dict[str, Any]:
        """Get summary statistics for the episode."""
        metrics = self.reward_calculator.get_episode_metrics()
        
        # Add trading-specific metrics
        if len(self.trades) > 0:
            trades_df = pd.DataFrame(self.trades)
            metrics['n_trades'] = len(self.trades)
            metrics['avg_trade_value'] = trades_df['value'].abs().mean()
            metrics['total_fees'] = trades_df['cost'].sum()
        else:
            metrics['n_trades'] = 0
            metrics['avg_trade_value'] = 0.0
            metrics['total_fees'] = 0.0
        
        # Final equity
        final_equity = self.equity_curve[-1] if self.equity_curve else self.initial_balance
        metrics['final_equity'] = final_equity
        metrics['total_return_pct'] = (final_equity - self.initial_balance) / self.initial_balance * 100
        
        return metrics