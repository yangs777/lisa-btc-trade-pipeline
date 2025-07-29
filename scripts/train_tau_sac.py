"""Training script for τ-SAC Bitcoin trading agent."""

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
from sklearn.model_selection import train_test_split

from src.rl.environments import BTCTradingEnvironment
from src.rl.wrappers import TradingEnvWrapper, EpisodeMonitor, ActionNoiseWrapper
from src.rl.models import TauSACTrader
from src.feature_engineering import FeatureEngineer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_prepare_data(
    data_path: str,
    feature_config: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load data and engineer features.
    
    Args:
        data_path: Path to parquet data file
        feature_config: Path to feature configuration
        train_ratio: Training data ratio
        val_ratio: Validation data ratio
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info(f"Loading data from {data_path}")
    
    # Load data
    df = pd.read_parquet(data_path)
    
    # Engineer features
    logger.info("Engineering features...")
    engineer = FeatureEngineer(config_path=feature_config)
    df_features = engineer.transform(df)
    
    # Remove any NaN values
    df_features = df_features.dropna()
    
    # Split data
    n_samples = len(df_features)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    train_df = df_features[:train_size]
    val_df = df_features[train_size:train_size + val_size]
    test_df = df_features[train_size + val_size:]
    
    logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df


def create_training_env(df: pd.DataFrame, config: dict) -> TradingEnvWrapper:
    """Create wrapped training environment.
    
    Args:
        df: DataFrame with features
        config: Environment configuration
        
    Returns:
        Wrapped training environment
    """
    # Create base environment
    base_env = BTCTradingEnvironment(
        df=df,
        initial_balance=config.get('initial_balance', 100000),
        max_position_size=config.get('max_position_size', 0.95),
        maker_fee=config.get('maker_fee', 0.0002),
        taker_fee=config.get('taker_fee', 0.0004),
        slippage_factor=config.get('slippage_factor', 0.0001),
        leverage=config.get('leverage', 1.0),
        lookback_window=config.get('lookback_window', 100),
    )
    
    # Wrap with monitoring
    env = EpisodeMonitor(base_env)
    
    # Add noise for exploration
    env = ActionNoiseWrapper(
        env,
        noise_scale=config.get('noise_scale', 0.1),
        noise_decay=config.get('noise_decay', 0.999),
    )
    
    # Add normalization wrapper
    env = TradingEnvWrapper(
        env,
        normalize_obs=True,
        clip_obs=10.0,
        action_repeat=1,
        reward_scale=config.get('reward_scale', 0.1),
    )
    
    return env


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train τ-SAC trading agent')
    parser.add_argument('--data', type=str, required=True, help='Path to data file')
    parser.add_argument('--config', type=str, default='indicators.yaml', help='Feature config')
    parser.add_argument('--output', type=str, default='models/tau_sac', help='Output directory')
    parser.add_argument('--timesteps', type=int, default=1000000, help='Training timesteps')
    parser.add_argument('--eval-freq', type=int, default=10000, help='Evaluation frequency')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cpu/cuda)')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    train_df, val_df, test_df = load_and_prepare_data(
        args.data,
        args.config,
        train_ratio=0.7,
        val_ratio=0.15,
    )
    
    # Environment configuration
    env_config = {
        'initial_balance': 100000,
        'max_position_size': 0.95,
        'maker_fee': 0.0002,
        'taker_fee': 0.0004,
        'slippage_factor': 0.0001,
        'leverage': 1.0,
        'lookback_window': 100,
        'noise_scale': 0.1,
        'noise_decay': 0.999,
        'reward_scale': 0.1,
    }
    
    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump({
            'env_config': env_config,
            'data_path': args.data,
            'feature_config': args.config,
            'timesteps': args.timesteps,
            'seed': args.seed,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)
    
    # Create environments
    logger.info("Creating environments...")
    train_env = create_training_env(train_df, env_config)
    val_env = create_training_env(val_df, env_config)
    
    # Create model
    logger.info("Initializing τ-SAC model...")
    trader = TauSACTrader(
        env=train_env,
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=int(1e6),
        tau=0.005,
        gamma=0.99,
        initial_temperature=1.0,
        use_sde=True,
        device=args.device,
    )
    
    # Train model
    logger.info(f"Starting training for {args.timesteps} timesteps...")
    trader.train(
        total_timesteps=args.timesteps,
        log_interval=100,
        eval_env=val_env,
        eval_freq=args.eval_freq,
        n_eval_episodes=5,
        save_path=str(output_dir),
    )
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_env = create_training_env(test_df, env_config)
    test_metrics = trader.get_trading_metrics(test_env, n_episodes=10)
    
    # Save test results
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    logger.info("Test results:")
    for metric, value in test_metrics.items():
        if 'mean_' in metric:
            logger.info(f"  {metric}: {value:.4f}")
    
    logger.info(f"Training complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()