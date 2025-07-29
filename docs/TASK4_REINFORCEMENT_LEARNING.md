# Task 4: τ-SAC Reinforcement Learning Environment

## Overview

Task 4 implements a sophisticated reinforcement learning trading system using τ-SAC (Temperature-aware Soft Actor-Critic) algorithm. The system learns optimal trading strategies through interaction with a realistic Bitcoin trading environment.

## Architecture

### 1. Trading Environment (`src/rl/environments.py`)
- **BTCTradingEnvironment**: Gymnasium-compatible trading environment
  - Realistic market simulation with transaction costs and slippage
  - Position management with leverage limits
  - Episode termination on excessive drawdown
  - Comprehensive observation space including market features and position info

### 2. Reward System (`src/rl/rewards.py`)
- **RBSRReward**: Risk-Balanced Sharpe Reward function
  - Balances profit maximization with risk control
  - Penalizes excessive volatility and drawdowns
  - Encourages efficient position management
  - Provides detailed performance metrics

### 3. Environment Wrappers (`src/rl/wrappers.py`)
- **TradingEnvWrapper**: Main wrapper with observation normalization
- **EpisodeMonitor**: Tracks and logs episode statistics  
- **ActionNoiseWrapper**: Adds exploration noise during training

### 4. τ-SAC Model (`src/rl/models.py`)
- **TradingFeatureExtractor**: Custom neural network for feature extraction
  - Separate pathways for market and position features
  - Attention mechanism for feature integration
- **TemperatureCallback**: Dynamic temperature adjustment
- **TauSACTrader**: Main trading agent with SAC algorithm

## Key Features

### Trading Environment
- **Action Space**: Continuous [-1, 1] representing position size
  - -1: Maximum short position
  - 0: Neutral (no position)
  - 1: Maximum long position
- **Observation Space**: Combines all technical indicators with position information
- **Transaction Costs**: Realistic maker/taker fees and slippage modeling
- **Risk Management**: Position limits and leverage constraints

### RBSR Reward Function
```python
reward = sharpe_ratio - drawdown_penalty - volatility_penalty - holding_penalty - transaction_cost_penalty
```

### τ-SAC Enhancements
- **Temperature Adaptation**: Automatically adjusts exploration based on performance
- **State-Dependent Exploration**: More sophisticated exploration strategy
- **Custom Feature Extraction**: Specialized neural architecture for trading

## Usage

### Training
```bash
python scripts/train_tau_sac.py \
    --data data/processed/btcusdt_20240101_1min.parquet \
    --config indicators.yaml \
    --output models/tau_sac \
    --timesteps 1000000 \
    --device cuda
```

### Backtesting
```bash
python scripts/backtest_tau_sac.py \
    --model models/tau_sac/final_model.zip \
    --data data/processed/btcusdt_20240201_1min.parquet \
    --config indicators.yaml \
    --output backtest_results
```

## Configuration

### Environment Parameters
```python
env_config = {
    'initial_balance': 100000,      # Starting capital in USDT
    'max_position_size': 0.95,      # Max 95% of balance
    'maker_fee': 0.0002,            # 0.02% maker fee
    'taker_fee': 0.0004,            # 0.04% taker fee
    'slippage_factor': 0.0001,      # 0.01% per unit size
    'leverage': 1.0,                # No leverage
    'lookback_window': 100,         # Historical context
}
```

### Model Hyperparameters
```python
model_config = {
    'learning_rate': 3e-4,
    'batch_size': 256,
    'buffer_size': 1_000_000,
    'tau': 0.005,                   # Soft update coefficient
    'gamma': 0.99,                  # Discount factor
    'initial_temperature': 1.0,     # Exploration temperature
    'use_sde': True,                # State-dependent exploration
}
```

## Performance Metrics

The system tracks comprehensive performance metrics:
- **Returns**: Total return, annualized return
- **Risk**: Maximum drawdown, Sharpe ratio, Sortino ratio
- **Trading**: Win rate, profit factor, average trade metrics
- **Execution**: Transaction costs, slippage impact

## Integration with Pipeline

1. **Data Flow**: Processed data with 104 indicators → RL Environment
2. **Training**: Historical data for agent training
3. **Inference**: Trained model generates trading signals
4. **Risk Management**: Integration with Task 5 risk system

## Next Steps

- Task 5: Implement risk management layer on top of RL signals
- Task 6: Deploy model in FastAPI prediction server
- Performance optimization and hyperparameter tuning
- Multi-asset portfolio extension