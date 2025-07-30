"""Tests for RL models without gymnasium dependency."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch


class TestTauSACModels:
    """Test τ-SAC model components."""

    def test_critic_network_init(self):
        """Test critic network initialization."""
        from src.rl.models import CriticNetwork

        critic = CriticNetwork(state_dim=10, action_dim=2)
        assert critic.fc1.in_features == 12  # state_dim + action_dim
        assert critic.fc1.out_features == 256
        assert critic.fc2.out_features == 256
        assert critic.out.out_features == 1

    def test_critic_forward(self):
        """Test critic forward pass."""
        from src.rl.models import CriticNetwork

        critic = CriticNetwork(state_dim=10, action_dim=2)
        state = torch.randn(32, 10)
        action = torch.randn(32, 2)

        q_value = critic(state, action)
        assert q_value.shape == (32, 1)

    def test_actor_network_init(self):
        """Test actor network initialization."""
        from src.rl.models import ActorNetwork

        actor = ActorNetwork(state_dim=10, action_dim=2)
        assert actor.fc1.in_features == 10
        assert actor.fc1.out_features == 256
        assert actor.fc2.out_features == 256
        assert actor.mean.out_features == 2
        assert actor.log_std.out_features == 2

    def test_actor_forward(self):
        """Test actor forward pass."""
        from src.rl.models import ActorNetwork

        actor = ActorNetwork(state_dim=10, action_dim=2)
        state = torch.randn(32, 10)

        action, log_prob = actor(state)
        assert action.shape == (32, 2)
        assert log_prob.shape == (32,)

    def test_tau_sac_trader_init(self):
        """Test τ-SAC trader initialization."""
        from src.rl.models import TauSACTrader

        trader = TauSACTrader(state_dim=10, action_dim=2, tau_init=1.0, tau_min=0.1, tau_max=2.0)

        assert trader.state_dim == 10
        assert trader.action_dim == 2
        assert trader.current_temp == 1.0
        assert trader.tau_min == 0.1
        assert trader.tau_max == 2.0

    def test_tau_sac_act(self):
        """Test τ-SAC act method."""
        from src.rl.models import TauSACTrader

        trader = TauSACTrader(state_dim=10, action_dim=2)
        state = np.random.randn(10)

        action = trader.act(state, eval_mode=False)
        assert action.shape == (2,)
        assert -1 <= action.min() <= action.max() <= 1

    def test_tau_sac_train_step(self):
        """Test τ-SAC train step."""
        from src.rl.models import TauSACTrader

        trader = TauSACTrader(state_dim=10, action_dim=2)

        # Create fake batch
        batch = {
            "states": torch.randn(32, 10),
            "actions": torch.randn(32, 2),
            "rewards": torch.randn(32, 1),
            "next_states": torch.randn(32, 10),
            "dones": torch.zeros(32, 1),
        }

        # Mock the optimizers
        trader.critic_optimizer = Mock()
        trader.actor_optimizer = Mock()
        trader.temp_optimizer = Mock()

        losses = trader.train_step(batch)

        assert "critic_loss" in losses
        assert "actor_loss" in losses
        assert "temp_loss" in losses
        assert "temperature" in losses


class TestRewardFunctions:
    """Test reward functions."""

    def test_sharpe_reward(self):
        """Test Sharpe ratio reward."""
        from src.rl.rewards import SharpeReward

        reward_fn = SharpeReward(window=20)

        # Test with returns
        returns = [0.01, -0.005, 0.02, -0.01, 0.015]
        reward = reward_fn.calculate(returns)
        assert isinstance(reward, float)

    def test_profit_reward(self):
        """Test profit reward."""
        from src.rl.rewards import ProfitReward

        reward_fn = ProfitReward()

        # Test simple profit
        reward = reward_fn.calculate(position=1.0, price_change=0.02, transaction_cost=0.001)
        expected = 1.0 * 0.02 - 0.001
        assert abs(reward - expected) < 1e-6

    def test_risk_adjusted_reward(self):
        """Test risk-adjusted reward."""
        from src.rl.rewards import RiskAdjustedReward

        reward_fn = RiskAdjustedReward(risk_penalty=0.5)

        reward = reward_fn.calculate(profit=0.02, position_size=0.8, volatility=0.15)

        # Should penalize based on risk
        assert reward < 0.02  # Less than raw profit due to risk penalty


class TestEnvironmentWrappers:
    """Test environment wrappers."""

    def test_trading_env_wrapper_init(self):
        """Test wrapper initialization."""
        from src.rl.wrappers import TradingEnvWrapper

        # Mock base environment
        mock_env = Mock()
        mock_env.observation_space = Mock(shape=(10,))
        mock_env.action_space = Mock()
        mock_env.reset = Mock(return_value=(np.zeros(10), {}))

        wrapper = TradingEnvWrapper(mock_env)
        assert wrapper.env is mock_env

    def test_normalize_wrapper(self):
        """Test normalization wrapper."""
        from src.rl.wrappers import NormalizeWrapper

        # Mock environment
        mock_env = Mock()
        mock_env.observation_space = Mock(shape=(10,))
        mock_env.reset = Mock(return_value=(np.random.randn(10), {}))
        mock_env.step = Mock(return_value=(np.random.randn(10), 0.1, False, False, {}))

        wrapper = NormalizeWrapper(mock_env)

        # Test reset
        obs, info = wrapper.reset()
        assert obs.shape == (10,)

        # Test step
        obs, reward, terminated, truncated, info = wrapper.step(0)
        assert obs.shape == (10,)

    def test_reward_scaling_wrapper(self):
        """Test reward scaling wrapper."""
        from src.rl.wrappers import RewardScalingWrapper

        mock_env = Mock()
        mock_env.reset = Mock(return_value=(np.zeros(10), {}))
        mock_env.step = Mock(return_value=(np.zeros(10), 10.0, False, False, {}))

        wrapper = RewardScalingWrapper(mock_env, scale=0.1)

        obs, reward, terminated, truncated, info = wrapper.step(0)
        assert reward == 1.0  # 10.0 * 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
