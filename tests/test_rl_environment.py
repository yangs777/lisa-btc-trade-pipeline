"""Tests for reinforcement learning environment."""

import numpy as np
import pandas as pd

from src.rl.environments import BTCTradingEnvironment
from src.rl.rewards import RBSRReward


def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=n_samples, freq="1h")

    # Generate realistic price data
    np.random.seed(42)
    price = 40000
    prices = []

    for _ in range(n_samples):
        change = np.random.normal(0, 0.002)  # 0.2% volatility
        price *= 1 + change
        prices.append(price)

    # Create OHLCV data
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": [p * 1.001 for p in prices],
            "low": [p * 0.999 for p in prices],
            "close": [p * 1.0005 for p in prices],
            "volume": np.random.uniform(100, 1000, n_samples),
        }
    )

    # Add some basic indicators
    df["sma_20"] = df["close"].rolling(20).mean()
    df["rsi"] = 50 + np.random.normal(0, 10, n_samples)  # Simplified RSI
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    return df.dropna()


class TestBTCTradingEnvironment:
    """Test trading environment functionality."""

    def test_environment_initialization(self):
        """Test environment can be initialized properly."""
        df = create_sample_data()
        env = BTCTradingEnvironment(df)

        assert env.n_steps == len(df)
        assert env.initial_balance == 100000
        assert env.max_position_size == 0.95
        assert env.action_space.shape == (1,)
        assert env.observation_space.shape[0] > 0

    def test_reset(self):
        """Test environment reset."""
        df = create_sample_data()
        env = BTCTradingEnvironment(df)

        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert env.balance == env.initial_balance
        assert env.position == 0.0
        assert env.current_step == env.lookback_window

    def test_step_buy_action(self):
        """Test buying action."""
        df = create_sample_data()
        env = BTCTradingEnvironment(df)
        obs, _ = env.reset()

        # Buy action
        action = np.array([0.5])  # 50% long
        obs, reward, terminated, truncated, info = env.step(action)

        assert env.position > 0  # Should have bought
        assert env.balance < env.initial_balance  # Money spent
        assert isinstance(reward, float)
        assert not terminated

    def test_step_sell_action(self):
        """Test selling action."""
        df = create_sample_data()
        env = BTCTradingEnvironment(df)
        obs, _ = env.reset()

        # First buy
        env.step(np.array([0.5]))

        # Then sell
        action = np.array([-0.5])  # 50% short
        obs, reward, terminated, truncated, info = env.step(action)

        assert env.position < 0  # Should be short
        assert len(env.trades) == 2  # Two trades executed

    def test_transaction_costs(self):
        """Test that transaction costs are applied."""
        df = create_sample_data()
        env = BTCTradingEnvironment(df, taker_fee=0.001)  # 0.1% fee
        obs, _ = env.reset()

        initial_balance = env.balance

        # Execute trade
        env.step(np.array([1.0]))  # Full long

        # Check that fees were deducted
        position_value = env.position * env.prices[env.current_step - 1]
        # expected_fee = position_value * env.taker_fee

        assert env.balance < initial_balance - position_value  # Fees deducted

    def test_episode_termination(self):
        """Test episode termination conditions."""
        df = create_sample_data(200)
        env = BTCTradingEnvironment(df)
        obs, _ = env.reset()

        # Run until end
        done = False
        steps = 0
        while not done and steps < 1000:
            obs, reward, terminated, truncated, info = env.step(np.array([0.0]))
            done = terminated or truncated
            steps += 1

        assert done
        assert env.current_step >= env.n_steps - 1

    def test_drawdown_termination(self):
        """Test termination on large drawdown."""
        df = create_sample_data()
        env = BTCTradingEnvironment(df)
        obs, _ = env.reset()

        # Simulate losses by shorting in uptrend
        env.position = -10.0  # Large short position
        env.balance = env.initial_balance * 0.19  # Just below 80% drawdown

        obs, reward, terminated, truncated, info = env.step(np.array([0.0]))

        assert terminated  # Should terminate due to drawdown
        assert reward < -5  # Large penalty

    def test_position_closing_on_termination(self):
        """Test that positions are closed when episode ends."""
        df = create_sample_data(200)
        env = BTCTradingEnvironment(df)
        obs, _ = env.reset()

        # Open position
        env.step(np.array([1.0]))
        assert env.position > 0

        # Run until end
        while env.current_step < env.n_steps - 2:
            env.step(np.array([1.0]))  # Stay long

        # Final step should close position
        obs, reward, terminated, truncated, info = env.step(np.array([1.0]))

        assert env.position == 0.0  # Position closed


class TestRBSRReward:
    """Test RBSR reward calculation."""

    def test_reward_initialization(self):
        """Test reward calculator initialization."""
        reward_calc = RBSRReward()

        assert reward_calc.lookback_window == 100
        assert reward_calc.risk_free_rate > 0
        assert len(reward_calc.returns_buffer) == 0

    def test_positive_returns_reward(self):
        """Test reward for positive returns."""
        reward_calc = RBSRReward()

        # Simulate profitable trades
        equities = [100000 * (1 + 0.001) ** i for i in range(50)]

        rewards = []
        for equity in equities:
            reward, metrics = reward_calc.calculate_reward(
                current_equity=equity, position_pnl=100, trade_cost=0, holding_period=10
            )
            rewards.append(reward)

        # Average reward should be positive
        assert np.mean(rewards[-20:]) > 0

    def test_drawdown_penalty(self):
        """Test penalty for drawdowns."""
        reward_calc = RBSRReward(drawdown_penalty=2.0)

        # Build up equity
        for i in range(20):
            reward_calc.calculate_reward(
                current_equity=100000 + i * 1000, position_pnl=0, trade_cost=0, holding_period=0
            )

        # Then drawdown
        reward, metrics = reward_calc.calculate_reward(
            current_equity=110000,  # Down from 119000
            position_pnl=0,
            trade_cost=0,
            holding_period=0,
        )

        assert metrics["drawdown"] > 0
        assert reward < 0  # Should be penalized

    def test_volatility_penalty(self):
        """Test penalty for excessive volatility."""
        reward_calc = RBSRReward(volatility_penalty=1.5)

        # Stable returns first
        for i in range(50):
            equity = 100000 + i * 100
            reward_calc.calculate_reward(equity, 0, 0, 0)

        # Then volatile returns
        volatile_equities = [100000, 105000, 95000, 110000, 90000]
        rewards = []

        for equity in volatile_equities:
            reward, metrics = reward_calc.calculate_reward(equity, 0, 0, 0)
            rewards.append(reward)

        # Should have volatility penalty
        assert any(
            metrics["volatility_penalty"] > 0
            for metrics in [reward_calc.calculate_reward(e, 0, 0, 0)[1] for e in volatile_equities]
        )

    def test_episode_metrics(self):
        """Test episode-level metrics calculation."""
        reward_calc = RBSRReward()

        # Simulate an episode
        equities = [100000]
        for _i in range(100):
            change = np.random.normal(0.001, 0.01)
            equities.append(equities[-1] * (1 + change))
            reward_calc.calculate_reward(equities[-1], 0, 0, 0)

        metrics = reward_calc.get_episode_metrics()

        assert "total_return" in metrics
        assert "max_drawdown" in metrics
        assert "sharpe_ratio" in metrics
        assert "win_rate" in metrics
        assert metrics["max_drawdown"] >= 0
