"""Tests for RL trading environment."""

from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest

from src.rl.environments import BTCTradingEnvironment


class TestBTCTradingEnvironment:
    """Test cases for BTCTradingEnvironment."""

    @pytest.fixture
    def sample_df(self):
        """Create sample OHLCV data with indicators."""
        n = 200
        dates = pd.date_range('2024-01-01', periods=n, freq='h')

        # Generate price data
        np.random.seed(42)
        price_base = 40000
        price_noise = np.random.randn(n).cumsum() * 100
        prices = price_base + price_noise

        df = pd.DataFrame({
            'open': prices + np.random.uniform(-50, 50, n),
            'high': prices + np.random.uniform(0, 100, n),
            'low': prices - np.random.uniform(0, 100, n),
            'close': prices,
            'volume': np.random.uniform(100, 200, n),
            # Add some indicators
            'sma_20': pd.Series(prices).rolling(20).mean().fillna(prices[0]),
            'rsi': np.random.uniform(30, 70, n),
            'macd': np.random.uniform(-50, 50, n),
        })

        return df

    @pytest.fixture
    def env(self, sample_df):
        """Create environment instance."""
        return BTCTradingEnvironment(
            df=sample_df,
            initial_balance=10000.0,
            max_position_size=0.95,
            lookback_window=10
        )

    def test_init(self, sample_df):
        """Test environment initialization."""
        env = BTCTradingEnvironment(
            df=sample_df,
            initial_balance=50000.0,
            max_position_size=0.8,
            maker_fee=0.001,
            taker_fee=0.002,
            slippage_factor=0.0005,
            leverage=2.0,
            lookback_window=50
        )

        assert env.initial_balance == 50000.0
        assert env.max_position_size == 0.8
        assert env.maker_fee == 0.001
        assert env.taker_fee == 0.002
        assert env.slippage_factor == 0.0005
        assert env.leverage == 2.0
        assert env.lookback_window == 50
        assert env.n_steps == len(sample_df)
        assert len(env.feature_columns) == 3  # sma_20, rsi, macd
        assert env.n_features == 10  # 3 features + 7 position info

    def test_init_missing_columns(self):
        """Test initialization with missing required columns."""
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'close': [100, 101, 102],
            # Missing high, low, volume
        })

        with pytest.raises(ValueError, match="DataFrame must contain columns"):
            BTCTradingEnvironment(df=df)

    def test_action_space(self, env):
        """Test action space configuration."""
        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (1,)
        assert env.action_space.low[0] == -1.0
        assert env.action_space.high[0] == 1.0

    def test_observation_space(self, env):
        """Test observation space configuration."""
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (10,)  # 3 features + 7 position info

    def test_reset(self, env):
        """Test environment reset."""
        obs, info = env.reset()

        # Check observation
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (10,)
        assert not np.any(np.isnan(obs))

        # Check initial state
        assert env.current_step == env.lookback_window
        assert env.balance == env.initial_balance
        assert env.position == 0.0
        assert env.entry_price == 0.0
        assert env.holding_period == 0
        assert not env.done
        assert len(env.trades) == 0
        assert len(env.equity_curve) == 1
        assert env.equity_curve[0] == env.initial_balance

        # Check info
        assert info['step'] == env.current_step
        assert info['balance'] == env.initial_balance
        assert info['position'] == 0.0
        assert info['equity'] == env.initial_balance

    def test_reset_with_seed(self, env):
        """Test deterministic reset with seed."""
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)

        # Should produce same initial observation
        np.testing.assert_array_equal(obs1, obs2)

    def test_step_neutral_action(self, env):
        """Test step with neutral action (no position)."""
        env.reset()

        obs, reward, terminated, truncated, info = env.step(np.array([0.0]))

        assert not terminated
        assert not truncated
        assert env.position == 0.0
        assert env.balance == env.initial_balance
        assert len(env.trades) == 0
        assert info['n_trades'] == 0

    def test_step_long_position(self, env):
        """Test opening a long position."""
        env.reset()
        initial_balance = env.balance
        current_price = env.prices[env.current_step]

        # Buy action (0.5 = 50% of max position)
        obs, reward, terminated, truncated, info = env.step(np.array([0.5]))

        # Check position opened
        assert env.position > 0
        assert env.balance < initial_balance  # Spent money
        assert len(env.trades) == 1
        assert env.entry_price > current_price  # Should include slippage

        # Check trade record
        trade = env.trades[0]
        assert trade['action'] == 0.5
        assert trade['position_change'] > 0
        assert trade['cost'] > 0  # Transaction cost

    def test_step_short_position(self, env):
        """Test opening a short position."""
        env.reset()

        # Short action (-0.5 = 50% short)
        obs, reward, terminated, truncated, info = env.step(np.array([-0.5]))

        assert env.position < 0
        assert len(env.trades) == 1

        trade = env.trades[0]
        assert trade['action'] == -0.5
        assert trade['position_change'] < 0

    def test_step_position_change(self, env):
        """Test changing positions."""
        env.reset()

        # Open long
        env.step(np.array([0.5]))
        long_position = env.position

        # Switch to short
        obs, reward, terminated, truncated, info = env.step(np.array([-0.5]))

        assert env.position < 0
        assert len(env.trades) == 2
        # Should have closed long and opened short

    def test_slippage_calculation(self, env):
        """Test slippage impact on execution price."""
        env.reset()
        base_price = env.prices[env.current_step]

        # Large position should have more slippage
        env.step(np.array([1.0]))  # Max long

        trade = env.trades[0]
        exec_price = trade['price']

        # Execution price should be higher than base (buying pressure)
        assert exec_price > base_price

        # Calculate expected slippage
        max_btc = (env.initial_balance * env.max_position_size) / base_price
        expected_slippage = env.slippage_factor * trade['position_change'] / max_btc
        expected_price = base_price * (1 + expected_slippage)

        assert abs(exec_price - expected_price) < 0.01

    def test_transaction_costs(self, env):
        """Test transaction cost calculation."""
        env.reset()

        # Open position
        env.step(np.array([0.5]))

        trade = env.trades[0]
        expected_cost = abs(trade['position_change'] * trade['price'] * env.taker_fee)

        assert abs(trade['cost'] - expected_cost) < 0.001

    def test_reward_calculation(self, env):
        """Test reward calculation through mock."""
        env.reset()

        # Mock reward calculator
        mock_reward = 0.5
        mock_metrics = {'sharpe': 1.2, 'drawdown': 0.05}
        env.reward_calculator.calculate_reward = Mock(
            return_value=(mock_reward, mock_metrics)
        )

        obs, reward, terminated, truncated, info = env.step(np.array([0.5]))

        assert reward == mock_reward
        assert info['reward_metrics'] == mock_metrics

        # Verify reward calculator was called with correct args
        env.reward_calculator.calculate_reward.assert_called_once()
        call_args = env.reward_calculator.calculate_reward.call_args[1]
        assert 'current_equity' in call_args
        assert 'position_pnl' in call_args
        assert 'trade_cost' in call_args
        assert 'holding_period' in call_args

    def test_holding_period_tracking(self, env):
        """Test holding period tracking."""
        env.reset()

        # Open position
        env.step(np.array([0.5]))
        assert env.holding_period == 0  # Just opened

        # Hold position
        env.step(np.array([0.5]))
        assert env.holding_period == 1

        env.step(np.array([0.5]))
        assert env.holding_period == 2

        # Close position
        env.step(np.array([0.0]))
        assert env.holding_period == 0

    def test_equity_calculation(self, env):
        """Test equity calculation with position."""
        env.reset()
        initial_balance = env.initial_balance

        # Open long position
        env.step(np.array([0.5]))

        # Price should affect equity
        current_price = env.prices[env.current_step]
        position_value = env.position * current_price
        expected_equity = env.balance + position_value

        info = env._get_info()
        assert abs(info['equity'] - expected_equity) < 0.01

    def test_pnl_calculation(self, env):
        """Test P&L calculation."""
        env.reset()

        # Open long position
        env.step(np.array([0.5]))
        entry_price = env.entry_price

        # Let price move
        env.current_step += 1
        current_price = env.prices[env.current_step]

        # Get observation to calculate P&L
        obs = env._get_observation()

        # Check P&L in observation
        # Position features include unrealized P&L %
        if entry_price > 0:
            expected_pnl_pct = (current_price - entry_price) / current_price
            # P&L % is at index 2 of position features (last 7 elements)
            actual_pnl_pct = obs[-5]  # 7 position features, P&L is 3rd
            assert abs(actual_pnl_pct - expected_pnl_pct) < 0.01

    def test_episode_termination_time(self, env):
        """Test episode termination at end of data."""
        env.reset()

        # Step to near end
        while env.current_step < env.n_steps - 2:
            env.step(np.array([0.0]))

        # Last step should truncate
        obs, reward, terminated, truncated, info = env.step(np.array([0.0]))

        assert not terminated
        assert truncated
        assert env.done

    def test_episode_termination_drawdown(self, env):
        """Test episode termination on large drawdown."""
        env.reset()

        # Force a large loss by setting balance very low
        env.balance = env.initial_balance * 0.15  # 85% loss

        obs, reward, terminated, truncated, info = env.step(np.array([0.0]))

        assert terminated
        assert not truncated
        assert env.done
        assert reward < -5  # Should have penalty

    def test_position_closing_on_done(self, env):
        """Test automatic position closing when episode ends."""
        env.reset()

        # Open position
        env.step(np.array([1.0]))
        assert env.position > 0

        # Force episode end
        env.current_step = env.n_steps - 2
        obs, reward, terminated, truncated, info = env.step(np.array([1.0]))

        assert env.done
        assert env.position == 0.0  # Should be closed

    def test_render_human_mode(self, sample_df):
        """Test human-readable rendering."""
        env = BTCTradingEnvironment(
            df=sample_df,
            render_mode="human"
        )
        env.reset()

        # Test render output
        output = env.render()
        assert isinstance(output, str)
        assert "Step:" in output
        assert "Balance:" in output
        assert "Position:" in output
        assert "Equity:" in output
        assert "Price:" in output
        assert "Trades:" in output

    def test_render_none_mode(self, env):
        """Test no rendering."""
        output = env.render()
        assert output is None

    def test_get_episode_summary(self, env):
        """Test episode summary generation."""
        env.reset()

        # Execute some trades
        env.step(np.array([0.5]))
        env.step(np.array([-0.5]))
        env.step(np.array([0.0]))

        summary = env.get_episode_summary()

        assert 'n_trades' in summary
        assert summary['n_trades'] == 3
        assert 'avg_trade_value' in summary
        assert 'total_fees' in summary
        assert 'final_equity' in summary
        assert 'total_return_pct' in summary

        # Check return calculation
        expected_return = (
            (summary['final_equity'] - env.initial_balance) /
            env.initial_balance * 100
        )
        assert abs(summary['total_return_pct'] - expected_return) < 0.01

    def test_minimum_trade_size(self, env):
        """Test minimum trade size enforcement."""
        env.reset()

        # Very small action should not trade
        env.step(np.array([0.00001]))

        assert len(env.trades) == 0
        assert env.position == 0.0

    def test_observation_normalization(self, env):
        """Test observation values are normalized."""
        env.reset()

        # Open large position
        env.step(np.array([1.0]))

        obs = env._get_observation()

        # Check no NaN or inf values
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))

        # Position ratio should be normalized
        position_ratio = obs[-7]  # First position feature
        assert -2.0 <= position_ratio <= 2.0

        # Balance ratio should be around 1.0 initially
        balance_ratio = obs[-6]
        assert 0.0 <= balance_ratio <= 2.0

    def test_error_on_step_after_done(self, env):
        """Test error when stepping after episode done."""
        env.reset()
        env.done = True

        with pytest.raises(ValueError, match="Episode is done"):
            env.step(np.array([0.0]))
