"""Tests for RL modules to boost coverage."""

import pytest
import numpy as np
import pandas as pd
import gymnasium as gym
from unittest.mock import Mock, patch


class TestRLEnvironments:
    """Test RL environments."""
    
    def test_trading_environment(self):
        """Test trading environment."""
        from src.rl.environments import TradingEnvironment
        
        # Create sample data
        dates = pd.date_range("2024-01-01", periods=1000, freq="h")
        data = pd.DataFrame({
            "timestamp": dates,
            "open": 50000 + np.random.randn(1000) * 100,
            "high": 50100 + np.random.randn(1000) * 100,
            "low": 49900 + np.random.randn(1000) * 100,
            "close": 50000 + np.random.randn(1000) * 100,
            "volume": 10000 + np.random.randn(1000) * 1000
        })
        
        # Add features
        data["returns"] = data["close"].pct_change()
        data["rsi"] = 50 + np.random.randn(1000) * 10
        data["macd"] = np.random.randn(1000) * 0.01
        
        env = TradingEnvironment(
            data=data,
            initial_balance=100000,
            fee=0.001,
            max_position=1.0
        )
        
        # Test reset
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert env.balance == 100000
        assert env.position == 0
        
        # Test step with different actions
        # Action 0: Hold
        obs, reward, done, truncated, info = env.step(0)
        assert env.position == 0
        
        # Action 1: Buy
        obs, reward, done, truncated, info = env.step(1)
        assert env.position > 0
        
        # Action 2: Sell
        obs, reward, done, truncated, info = env.step(2)
        assert env.position == 0
        
        # Test episode completion
        for _ in range(100):
            if done or truncated:
                break
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
        
        # Test render
        env.render()
    
    def test_tau_trading_environment(self):
        """Test tau-specific trading environment."""
        from src.rl.environments import TauTradingEnvironment
        
        # Create data
        data = pd.DataFrame({
            "close": 50000 + np.cumsum(np.random.randn(1000) * 100),
            "volume": 10000 + np.random.randn(1000) * 1000,
            "rsi": 50 + np.random.randn(1000) * 10
        })
        
        env = TauTradingEnvironment(
            data=data,
            tau_values=[3, 6, 9, 12],
            initial_balance=100000
        )
        
        # Test initialization
        assert len(env.tau_values) == 4
        assert env.current_tau is None
        
        # Test reset
        obs, info = env.reset()
        assert env.current_tau in env.tau_values
        
        # Test tau-specific actions
        action = 0  # Hold with current tau
        obs, reward, done, truncated, info = env.step(action)
        assert "tau" in info
        
        # Test tau switching
        action = 4  # Switch to different tau
        obs, reward, done, truncated, info = env.step(action)
        
    def test_multi_timeframe_environment(self):
        """Test multi-timeframe environment."""
        from src.rl.environments import MultiTimeframeEnvironment
        
        # Create multi-timeframe data
        timeframes = {
            "1h": pd.DataFrame({
                "close": np.random.randn(1000) + 50000,
                "volume": np.random.randn(1000) * 1000 + 10000
            }),
            "4h": pd.DataFrame({
                "close": np.random.randn(250) + 50000,
                "volume": np.random.randn(250) * 1000 + 10000
            }),
            "1d": pd.DataFrame({
                "close": np.random.randn(42) + 50000,
                "volume": np.random.randn(42) * 1000 + 10000
            })
        }
        
        env = MultiTimeframeEnvironment(
            data=timeframes,
            primary_timeframe="1h"
        )
        
        # Test observation includes all timeframes
        obs, info = env.reset()
        assert len(obs) > len(timeframes["1h"].columns)


class TestRLModels:
    """Test RL models."""
    
    def test_tau_sac_model(self):
        """Test Tau-SAC model."""
        from src.rl.models import TauSAC
        
        # Create model
        model = TauSAC(
            observation_dim=10,
            action_dim=3,
            tau_values=[3, 6, 9, 12],
            hidden_dim=256,
            learning_rate=3e-4
        )
        
        # Test forward pass
        obs = np.random.randn(32, 10).astype(np.float32)
        tau = np.array([3] * 32)
        
        # Get action
        action, log_prob = model.get_action(obs, tau)
        assert action.shape == (32,)
        assert log_prob.shape == (32,)
        
        # Test critic
        q1, q2 = model.critic(obs, action, tau)
        assert q1.shape == (32, 1)
        assert q2.shape == (32, 1)
        
        # Test value function
        value = model.value(obs, tau)
        assert value.shape == (32, 1)
    
    def test_position_aware_sac(self):
        """Test position-aware SAC."""
        from src.rl.models import PositionAwareSAC
        
        model = PositionAwareSAC(
            observation_dim=10,
            action_dim=3,
            max_position=1.0
        )
        
        # Test with position encoding
        obs = np.random.randn(32, 10).astype(np.float32)
        position = np.random.uniform(-1, 1, size=(32, 1)).astype(np.float32)
        
        # Augment observation with position
        obs_with_pos = np.concatenate([obs, position], axis=1)
        
        # Get action considering position
        action = model.get_action(obs_with_pos)
        assert len(action) == 32
    
    def test_ensemble_model(self):
        """Test ensemble model."""
        from src.rl.models import EnsembleModel
        
        # Create ensemble
        models = []
        for _ in range(3):
            model = Mock()
            model.predict = Mock(return_value=np.random.randn(32, 3))
            models.append(model)
        
        ensemble = EnsembleModel(models, weights=[0.5, 0.3, 0.2])
        
        # Test prediction
        obs = np.random.randn(32, 10)
        predictions = ensemble.predict(obs)
        assert predictions.shape == (32, 3)
        
        # Test voting
        votes = ensemble.vote(obs)
        assert len(votes) == 32


class TestRLRewards:
    """Test RL reward functions."""
    
    def test_sharpe_reward(self):
        """Test Sharpe ratio reward."""
        from src.rl.rewards import SharpeReward
        
        reward_fn = SharpeReward(window=20)
        
        # Test with returns
        returns = [0.01, -0.02, 0.015, -0.005, 0.02] * 5
        
        for ret in returns:
            reward = reward_fn.calculate(ret)
        
        # Should have Sharpe-based reward after window
        assert isinstance(reward, float)
    
    def test_risk_adjusted_reward(self):
        """Test risk-adjusted reward."""
        from src.rl.rewards import RiskAdjustedReward
        
        reward_fn = RiskAdjustedReward(
            risk_free_rate=0.02,
            downside_penalty=2.0
        )
        
        # Test positive return
        reward_pos = reward_fn.calculate(
            return_pct=0.05,
            position_size=0.5,
            volatility=0.2
        )
        assert reward_pos > 0
        
        # Test negative return (higher penalty)
        reward_neg = reward_fn.calculate(
            return_pct=-0.05,
            position_size=0.5,
            volatility=0.2
        )
        assert reward_neg < 0
        assert abs(reward_neg) > abs(reward_pos)
    
    def test_composite_reward(self):
        """Test composite reward function."""
        from src.rl.rewards import CompositeReward
        
        # Create composite with multiple components
        components = {
            "return": lambda r, **kw: r,
            "sharpe": lambda r, **kw: r / 0.2 if r > 0 else r * 2,
            "drawdown": lambda r, dd=0, **kw: -dd * 10
        }
        
        reward_fn = CompositeReward(
            components=components,
            weights={"return": 0.5, "sharpe": 0.3, "drawdown": 0.2}
        )
        
        # Test calculation
        reward = reward_fn.calculate(
            return_pct=0.02,
            drawdown=0.05
        )
        
        expected = 0.5 * 0.02 + 0.3 * (0.02 / 0.2) + 0.2 * (-0.05 * 10)
        assert abs(reward - expected) < 0.001
    
    def test_tau_specific_reward(self):
        """Test tau-specific reward."""
        from src.rl.rewards import TauSpecificReward
        
        reward_fn = TauSpecificReward(
            tau_penalties={3: 0.1, 6: 0.05, 9: 0.02, 12: 0.01}
        )
        
        # Test with different tau values
        base_return = 0.02
        
        reward_3 = reward_fn.calculate(base_return, tau=3)
        reward_12 = reward_fn.calculate(base_return, tau=12)
        
        # Longer tau should have less penalty
        assert reward_12 > reward_3


class TestRLWrappers:
    """Test RL environment wrappers."""
    
    def test_normalization_wrapper(self):
        """Test observation normalization wrapper."""
        from src.rl.wrappers import NormalizeObservation
        
        # Create base environment
        base_env = Mock(spec=gym.Env)
        base_env.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,)
        )
        base_env.reset = Mock(return_value=(np.random.randn(10), {}))
        base_env.step = Mock(return_value=(
            np.random.randn(10), 0.1, False, False, {}
        ))
        
        # Wrap with normalization
        env = NormalizeObservation(base_env)
        
        # Test reset
        obs, info = env.reset()
        assert obs.shape == (10,)
        
        # Test step multiple times to build statistics
        for _ in range(100):
            action = 0
            obs, reward, done, truncated, info = env.step(action)
            assert obs.shape == (10,)
        
        # Check normalization statistics
        assert hasattr(env, 'obs_rms')
        assert env.obs_rms.mean.shape == (10,)
    
    def test_reward_scaling_wrapper(self):
        """Test reward scaling wrapper."""
        from src.rl.wrappers import ScaleReward
        
        base_env = Mock(spec=gym.Env)
        base_env.reset = Mock(return_value=(np.array([0]), {}))
        base_env.step = Mock(return_value=(
            np.array([0]), 10.0, False, False, {}
        ))
        
        # Wrap with scaling
        env = ScaleReward(base_env, scale=0.1)
        
        obs, info = env.reset()
        obs, reward, done, truncated, info = env.step(0)
        
        assert reward == 1.0  # 10.0 * 0.1
    
    def test_frame_stack_wrapper(self):
        """Test frame stacking wrapper."""
        from src.rl.wrappers import FrameStack
        
        base_env = Mock(spec=gym.Env)
        base_env.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(4,)
        )
        base_env.reset = Mock(return_value=(np.array([1, 2, 3, 4]), {}))
        
        # Wrap with frame stacking
        env = FrameStack(base_env, num_stack=4)
        
        # Test reset
        obs, info = env.reset()
        assert obs.shape == (4, 4)  # 4 frames of 4 features
        
        # Test step
        base_env.step = Mock(return_value=(
            np.array([5, 6, 7, 8]), 0.1, False, False, {}
        ))
        
        obs, reward, done, truncated, info = env.step(0)
        assert obs.shape == (4, 4)
        assert obs[-1, 0] == 5  # Latest frame
    
    def test_action_repeat_wrapper(self):
        """Test action repeat wrapper."""
        from src.rl.wrappers import ActionRepeat
        
        base_env = Mock(spec=gym.Env)
        base_env.reset = Mock(return_value=(np.array([0]), {}))
        
        step_count = 0
        rewards = [0.1, 0.2, 0.3]
        
        def mock_step(action):
            nonlocal step_count
            reward = rewards[step_count % 3]
            step_count += 1
            return np.array([step_count]), reward, False, False, {}
        
        base_env.step = Mock(side_effect=mock_step)
        
        # Wrap with action repeat
        env = ActionRepeat(base_env, repeat=3)
        
        obs, info = env.reset()
        obs, total_reward, done, truncated, info = env.step(1)
        
        assert step_count == 3  # Action repeated 3 times
        assert total_reward == sum(rewards)  # Sum of all rewards