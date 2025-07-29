"""Backtesting script for τ-SAC trading agent."""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import json
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.rl.environments import BTCTradingEnvironment
from src.rl.wrappers import TradingEnvWrapper
from src.rl.models import TauSACTrader
from src.feature_engineering import FeatureEngineer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BacktestAnalyzer:
    """Analyze backtest results with detailed metrics and visualizations."""
    
    def __init__(self, initial_balance: float = 100000):
        """Initialize analyzer.
        
        Args:
            initial_balance: Starting capital
        """
        self.initial_balance = initial_balance
        self.trades = []
        self.equity_curve = []
        self.positions = []
        self.prices = []
    
    def record_step(
        self,
        timestamp: int,
        price: float,
        position: float,
        balance: float,
        equity: float,
        action: float,
    ) -> None:
        """Record a single backtest step."""
        self.prices.append(price)
        self.positions.append(position)
        self.equity_curve.append(equity)
        
        # Check for trades
        if len(self.positions) > 1:
            position_change = position - self.positions[-2]
            if abs(position_change) > 0.0001:
                self.trades.append({
                    'timestamp': timestamp,
                    'price': price,
                    'position_change': position_change,
                    'new_position': position,
                    'balance': balance,
                    'equity': equity,
                    'action': action,
                })
    
    def calculate_metrics(self) -> dict:
        """Calculate comprehensive backtest metrics."""
        if len(self.equity_curve) < 2:
            return {}
        
        equity_array = np.array(self.equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        # Basic metrics
        total_return = (equity_array[-1] - equity_array[0]) / equity_array[0]
        
        # Risk metrics
        running_max = np.maximum.accumulate(equity_array)
        drawdowns = (running_max - equity_array) / running_max
        max_drawdown = np.max(drawdowns)
        
        # Find drawdown periods
        dd_start = np.argmax(drawdowns)
        dd_peak = np.argmax(equity_array[:dd_start]) if dd_start > 0 else 0
        
        # Sharpe ratio (assuming hourly data)
        hours_per_year = 24 * 365
        if len(returns) > 0 and np.std(returns) > 0:
            annualized_return = (1 + np.mean(returns)) ** hours_per_year - 1
            annualized_vol = np.std(returns) * np.sqrt(hours_per_year)
            sharpe_ratio = annualized_return / annualized_vol
        else:
            sharpe_ratio = 0.0
        
        # Sortino ratio (downside deviation)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_vol = np.std(negative_returns) * np.sqrt(hours_per_year)
            sortino_ratio = annualized_return / downside_vol if downside_vol > 0 else 0
        else:
            sortino_ratio = float('inf') if annualized_return > 0 else 0
        
        # Trading metrics
        n_trades = len(self.trades)
        if n_trades > 0:
            trades_df = pd.DataFrame(self.trades)
            
            # Calculate trade returns
            trade_returns = []
            for i, trade in trades_df.iterrows():
                if i < len(trades_df) - 1:
                    entry_price = trade['price']
                    exit_price = trades_df.iloc[i + 1]['price']
                    position = trade['new_position']
                    if position != 0:
                        trade_return = position * (exit_price - entry_price) / entry_price
                        trade_returns.append(trade_return)
            
            if trade_returns:
                winning_trades = [r for r in trade_returns if r > 0]
                losing_trades = [r for r in trade_returns if r < 0]
                
                win_rate = len(winning_trades) / len(trade_returns)
                avg_win = np.mean(winning_trades) if winning_trades else 0
                avg_loss = np.mean(losing_trades) if losing_trades else 0
                
                profit_factor = (sum(winning_trades) / -sum(losing_trades)) if losing_trades else float('inf')
            else:
                win_rate = avg_win = avg_loss = profit_factor = 0
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
        
        # Compile metrics
        metrics = {
            'total_return': total_return * 100,  # Percentage
            'annualized_return': annualized_return * 100 if 'annualized_return' in locals() else 0,
            'max_drawdown': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'n_trades': n_trades,
            'win_rate': win_rate * 100,
            'avg_win': avg_win * 100,
            'avg_loss': avg_loss * 100,
            'profit_factor': profit_factor,
            'final_equity': equity_array[-1],
            'best_equity': np.max(equity_array),
            'worst_equity': np.min(equity_array),
        }
        
        return metrics
    
    def plot_results(self, save_path: str) -> None:
        """Create comprehensive backtest visualizations."""
        fig, axes = plt.subplots(4, 1, figsize=(12, 16))
        
        # 1. Equity curve
        ax = axes[0]
        ax.plot(self.equity_curve, label='Equity', linewidth=2)
        ax.axhline(y=self.initial_balance, color='gray', linestyle='--', alpha=0.7)
        ax.set_title('Equity Curve', fontsize=14)
        ax.set_ylabel('Equity (USDT)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax = axes[1]
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown_pct = (running_max - equity_array) / running_max * 100
        ax.fill_between(range(len(drawdown_pct)), 0, drawdown_pct, 
                       color='red', alpha=0.3, label='Drawdown')
        ax.plot(drawdown_pct, color='red', linewidth=1)
        ax.set_title('Drawdown %', fontsize=14)
        ax.set_ylabel('Drawdown %')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Position and Price
        ax = axes[2]
        ax2 = ax.twinx()
        
        # Plot price
        ax.plot(self.prices, color='black', alpha=0.7, linewidth=1, label='BTC Price')
        ax.set_ylabel('BTC Price (USDT)')
        
        # Plot position
        position_array = np.array(self.positions)
        colors = ['red' if p < 0 else 'green' if p > 0 else 'gray' for p in position_array]
        ax2.scatter(range(len(position_array)), position_array, c=colors, alpha=0.5, s=1)
        ax2.set_ylabel('Position (BTC)')
        
        ax.set_title('Price and Position', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        
        # 4. Trade distribution
        ax = axes[3]
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trade_actions = trades_df['action'].values
            ax.hist(trade_actions, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax.set_title('Trade Action Distribution', fontsize=14)
            ax.set_xlabel('Action Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_trades(self, save_path: str) -> None:
        """Save trade history to CSV."""
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv(save_path, index=False)
            logger.info(f"Saved {len(trades_df)} trades to {save_path}")


def run_backtest(
    model_path: str,
    data_path: str,
    feature_config: str,
    output_dir: str,
) -> dict:
    """Run full backtest on historical data.
    
    Args:
        model_path: Path to trained model
        data_path: Path to data file
        feature_config: Path to feature configuration
        output_dir: Directory for results
        
    Returns:
        Dictionary of backtest metrics
    """
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    
    # Engineer features
    logger.info("Engineering features...")
    engineer = FeatureEngineer(config_path=feature_config)
    df_features = engineer.transform(df)
    df_features = df_features.dropna()
    
    # Create environment
    env_config = {
        'initial_balance': 100000,
        'max_position_size': 0.95,
        'maker_fee': 0.0002,
        'taker_fee': 0.0004,
        'slippage_factor': 0.0001,
        'leverage': 1.0,
        'lookback_window': 100,
    }
    
    base_env = BTCTradingEnvironment(df=df_features, **env_config)
    env = TradingEnvWrapper(
        base_env,
        normalize_obs=True,
        clip_obs=10.0,
        action_repeat=1,
        reward_scale=0.1,
    )
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    trader = TauSACTrader(env=env)
    trader.load(model_path)
    
    # Initialize analyzer
    analyzer = BacktestAnalyzer(initial_balance=env_config['initial_balance'])
    
    # Run backtest
    logger.info("Running backtest...")
    obs, _ = env.reset()
    done = False
    step = 0
    
    while not done:
        # Get action from model
        action, _ = trader.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Record step
        base_info = info.get('wrapper', info)
        analyzer.record_step(
            timestamp=step,
            price=base_env.prices[base_env.current_step - 1],
            position=base_env.position,
            balance=base_env.balance,
            equity=base_info.get('equity', base_env.balance),
            action=float(action[0]),
        )
        
        step += 1
    
    # Calculate metrics
    metrics = analyzer.calculate_metrics()
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    with open(output_path / 'backtest_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save plots
    analyzer.plot_results(str(output_path / 'backtest_plots.png'))
    
    # Save trades
    analyzer.save_trades(str(output_path / 'trades.csv'))
    
    # Print summary
    logger.info("\nBacktest Results:")
    logger.info(f"  Total Return: {metrics['total_return']:.2f}%")
    logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
    logger.info(f"  Win Rate: {metrics['win_rate']:.2f}%")
    logger.info(f"  Number of Trades: {metrics['n_trades']}")
    
    return metrics


def main():
    """Main backtesting function."""
    parser = argparse.ArgumentParser(description='Backtest τ-SAC trading agent')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data', type=str, required=True, help='Path to data file')
    parser.add_argument('--config', type=str, default='indicators.yaml', help='Feature config')
    parser.add_argument('--output', type=str, default='backtest_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Run backtest
    metrics = run_backtest(
        model_path=args.model,
        data_path=args.data,
        feature_config=args.config,
        output_dir=args.output,
    )
    
    logger.info(f"\nBacktest complete! Results saved to {args.output}")


if __name__ == "__main__":
    main()