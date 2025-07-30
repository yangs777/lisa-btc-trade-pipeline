"""Test coverage for RL environment wrappers."""

import pytest
import numpy as np
import gymnasium as gym
from typing import Any, Dict, Tuple, Optional, Union
from unittest.mock import Mock, patch
from src.rl.wrappers import TradingEnvWrapper, EpisodeMonitor, ActionNoiseWrapper


class MockEnv(gym.Env):
    """Mock environment for testing."""
    
    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(5,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.step_count = 0
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.step_count = 0
        return np.random.randn(5).astype(np.float32), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        obs = np.random.randn(5).astype(np.float32)
        reward = float(np.random.randn())
        terminated = self.step_count >= 10
        truncated = False
        return obs, reward, terminated, truncated, {}


class TestTradingEnvWrapper:
    """Test TradingEnvWrapper class."""
    
    def test_initialization(self) -> None:
        """Test wrapper initialization."""
        env = MockEnv()
        wrapper = TradingEnvWrapper(
            env,
            normalize_obs=True,
            clip_obs=5.0,
            action_repeat=2,
            reward_scale=0.1
        )
        
        assert wrapper.normalize_obs == True
        assert wrapper.clip_obs == 5.0
        assert wrapper.action_repeat == 2
        assert wrapper.reward_scale == 0.1
        assert wrapper.obs_mean is None
        assert wrapper.obs_std is None
        assert wrapper.n_obs == 0
    
    def test_reset(self) -> None:
        """Test reset method."""
        env = MockEnv()
        wrapper = TradingEnvWrapper(env)
        
        obs, info = wrapper.reset()
        
        assert obs.shape == (5,)
        assert isinstance(info, dict)
        assert wrapper.n_obs == 1
    
    def test_step_without_normalization(self) -> None:
        """Test step without normalization."""
        env = MockEnv()
        wrapper = TradingEnvWrapper(env, normalize_obs=False, reward_scale=0.5)
        
        obs, _ = wrapper.reset()
        action = np.array([0.5, -0.5], dtype=np.float32)
        
        obs, reward, terminated, truncated, info = wrapper.step(action)
        
        assert obs.shape == (5,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_step_with_normalization(self) -> None:
        """Test step with normalization."""
        env = MockEnv()
        wrapper = TradingEnvWrapper(env, normalize_obs=True, clip_obs=3.0)
        
        # Set mock statistics
        wrapper.obs_mean = np.zeros(5)
        wrapper.obs_std = np.ones(5)
        wrapper.n_obs = 100
        
        obs, _ = wrapper.reset()
        action = np.array([0.5, -0.5], dtype=np.float32)
        
        obs, reward, terminated, truncated, info = wrapper.step(action)
        
        # Check observation is clipped
        assert np.all(obs >= -3.0)
        assert np.all(obs <= 3.0)
    
    def test_action_repeat(self) -> None:
        """Test action repeat functionality."""
        env = MockEnv()
        wrapper = TradingEnvWrapper(env, action_repeat=3)
        
        wrapper.reset()
        action = np.array([0.5, -0.5], dtype=np.float32)
        
        # Track original step count
        original_step_count = env.step_count
        
        # Step should repeat action 3 times
        obs, reward, terminated, truncated, info = wrapper.step(action)
        
        # Verify action was repeated
        assert env.step_count == original_step_count + 3
    
    def test_compute_statistics(self) -> None:
        """Test computing observation statistics."""
        env = MockEnv()
        wrapper = TradingEnvWrapper(env, normalize_obs=True)
        
        # Reset and step multiple times
        wrapper.reset()
        for _ in range(10):
            action = env.action_space.sample()
            wrapper.step(action)
        
        # Get normalization statistics
        stats = wrapper.get_normalization_stats()
        
        assert stats["mean"] is not None
        assert stats["std"] is not None
        assert stats["mean"].shape == (5,)
        assert stats["std"].shape == (5,)
        assert stats["n_samples"] > 0


class TestEpisodeMonitor:
    """Test EpisodeMonitor wrapper."""
    
    def test_initialization(self) -> None:
        """Test monitor initialization."""
        env = MockEnv()
        monitor = EpisodeMonitor(env)
        
        assert monitor.episode_rewards == []
        assert monitor.episode_lengths == []
        assert monitor.episode_count == 0
        assert monitor.current_reward == 0
        assert monitor.current_length == 0
    
    def test_reset(self) -> None:
        """Test monitor reset."""
        env = MockEnv()
        monitor = EpisodeMonitor(env)
        
        # Set some current values
        monitor.current_reward = 10.0
        monitor.current_length = 5
        
        obs, info = monitor.reset()
        
        assert monitor.current_reward == 0
        assert monitor.current_length == 0
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
    
    def test_step_tracking(self) -> None:
        """Test step tracking."""
        env = MockEnv()
        monitor = EpisodeMonitor(env)
        
        monitor.reset()
        
        total_reward = 0
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = monitor.step(action)
            total_reward += float(reward)
            
            assert monitor.current_length == i + 1
            assert abs(monitor.current_reward - total_reward) < 1e-6
            
            if terminated or truncated:
                break
    
    def test_episode_completion(self) -> None:
        """Test episode completion tracking."""
        env = MockEnv()
        
        # Mock the step method to control termination
        def mock_step(action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
            env.step_count += 1
            obs = np.random.randn(5).astype(np.float32)
            reward = 1.0  # Fixed reward
            terminated = env.step_count >= 5  # Terminate after 5 steps
            truncated = False
            return obs, reward, terminated, truncated, {}
        
        with patch.object(env, 'step', mock_step):
            monitor = EpisodeMonitor(env)
            
            monitor.reset()
            
            # Run until episode ends
            done = False
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = monitor.step(action)
                done = terminated or truncated
            
            # Check episode was recorded
            assert len(monitor.episode_rewards) == 1
            assert len(monitor.episode_lengths) == 1
            assert monitor.episode_count == 1
            assert monitor.episode_rewards[0] == 5.0  # 5 steps * 1.0 reward
            assert monitor.episode_lengths[0] == 5
    
    def test_get_episode_statistics(self) -> None:
        """Test getting episode statistics."""
        env = MockEnv()
        monitor = EpisodeMonitor(env)
        
        # Add some mock episodes
        monitor.episode_rewards = [10.0, 20.0, 15.0]
        monitor.episode_lengths = [100, 200, 150]
        monitor.episode_count = 3
        
        stats = monitor.get_episode_statistics()
        
        assert stats["episode_count"] == 3
        assert stats["mean_reward"] == 15.0
        assert stats["mean_length"] == 150.0
        assert stats["total_steps"] == 450
        
        # Test with no episodes
        monitor.episode_rewards = []
        monitor.episode_lengths = []
        monitor.episode_count = 0
        
        stats = monitor.get_episode_statistics()
        
        assert stats["episode_count"] == 0
        assert stats["mean_reward"] == 0.0
        assert stats["mean_length"] == 0.0
        assert stats["total_steps"] == 0


class TestActionNoiseWrapper:
    """Test ActionNoiseWrapper."""
    
    def test_initialization(self) -> None:
        """Test noise wrapper initialization."""
        env = MockEnv()
        wrapper = ActionNoiseWrapper(env, noise_scale=0.1)
        
        assert wrapper.noise_scale == 0.1
    
    def test_reset(self) -> None:
        """Test reset passes through correctly."""
        env = MockEnv()
        wrapper = ActionNoiseWrapper(env)
        
        obs1, info1 = env.reset()
        env.reset()  # Reset again
        obs2, info2 = wrapper.reset()
        
        assert obs2.shape == obs1.shape
        assert isinstance(info2, dict)
    
    def test_gaussian_noise(self) -> None:
        """Test Gaussian noise addition."""
        env = MockEnv()
        wrapper = ActionNoiseWrapper(env, noise_scale=0.1)
        
        wrapper.reset()
        
        # Test multiple actions
        for _ in range(10):
            clean_action = np.array([0.5, -0.5], dtype=np.float32)
            
            # Mock the step to capture the noisy action
            noisy_actions = []
            
            def capture_action(action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
                noisy_actions.append(action.copy())
                return np.zeros(5, dtype=np.float32), 0.0, False, False, {}
            
            with patch.object(env, 'step', capture_action):
                wrapper.step(clean_action)
            
            # Check noise was added
            assert len(noisy_actions) == 1
            noisy_action = noisy_actions[0]
            assert not np.allclose(noisy_action, clean_action)
            
            # Check action is still within bounds
            assert np.all(noisy_action >= -1.0)
            assert np.all(noisy_action <= 1.0)
    
    def test_noise_decay(self) -> None:
        """Test noise decay over episodes."""
        env = MockEnv()
        wrapper = ActionNoiseWrapper(env, noise_scale=1.0, noise_decay=0.9, min_noise=0.1)
        
        initial_noise = wrapper.noise_scale
        
        # Reset multiple times to trigger decay
        for i in range(5):
            wrapper.reset()
            expected_noise = max(initial_noise * (0.9 ** (i + 1)), 0.1)
            assert abs(wrapper.noise_scale - expected_noise) < 1e-6
        
        # Check minimum noise is respected
        for _ in range(20):
            wrapper.reset()
        
        assert wrapper.noise_scale >= 0.1
    
    def test_no_noise_with_zero_scale(self) -> None:
        """Test no noise when scale is zero."""
        env = MockEnv()
        wrapper = ActionNoiseWrapper(env, noise_scale=0.0)
        
        wrapper.reset()
        
        clean_action = np.array([0.5, -0.5], dtype=np.float32)
        
        # Mock the step to capture the action
        captured_actions = []
        
        def capture_action(action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
            captured_actions.append(action.copy())
            return np.zeros(5, dtype=np.float32), 0.0, False, False, {}
        
        with patch.object(env, 'step', capture_action):
            wrapper.step(clean_action)
        
        # Check no noise was added
        assert np.allclose(captured_actions[0], clean_action)