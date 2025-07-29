"""Test coverage for RL environment wrappers."""

import pytest
import numpy as np
import gymnasium as gym
from src.rl.wrappers import TradingEnvWrapper, EpisodeMonitor, ActionNoiseWrapper


class MockEnv(gym.Env):
    """Mock environment for testing."""
    
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(5,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.step_count = 0
    
    def reset(self, seed=None, options=None):
        self.step_count = 0
        return np.random.randn(5).astype(np.float32), {}
    
    def step(self, action):
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
        """Test environment reset."""
        env = MockEnv()
        wrapper = TradingEnvWrapper(env)
        
        obs, info = wrapper.reset()
        
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (5,)
        assert obs.dtype == np.float32
        assert len(wrapper.episode_rewards) == 0
        assert wrapper.episode_length == 0
    
    def test_step_basic(self) -> None:
        """Test basic step functionality."""
        env = MockEnv()
        wrapper = TradingEnvWrapper(env, reward_scale=0.5)
        
        wrapper.reset()
        action = np.array([0.5, -0.5])
        obs, reward, terminated, truncated, info = wrapper.step(action)
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "wrapper" in info
        assert info["wrapper"]["episode_length"] == 1
    
    def test_action_repeat(self) -> None:
        """Test action repeat functionality."""
        env = MockEnv()
        wrapper = TradingEnvWrapper(env, action_repeat=3)
        
        wrapper.reset()
        action = np.array([0.1, 0.2])
        obs, reward, terminated, truncated, info = wrapper.step(action)
        
        # Should have stepped 3 times
        assert wrapper.episode_length == 3
        assert len(wrapper.episode_rewards) == 3
    
    def test_observation_normalization(self) -> None:
        """Test observation normalization."""
        env = MockEnv()
        wrapper = TradingEnvWrapper(env, normalize_obs=True)
        
        # Reset and step multiple times to build statistics
        wrapper.reset()
        for i in range(150):  # Need > 100 samples for normalization
            wrapper.step(np.array([0.0, 0.0]))
            if wrapper.episode_length >= 10:
                wrapper.reset()
        
        # Check normalization stats exist
        assert wrapper.n_obs > 100
        assert wrapper.obs_mean is not None
        assert wrapper.obs_std is not None
        
        # Next observation should be normalized
        obs, _ = wrapper.reset()
        assert obs.dtype == np.float32
        assert np.all(np.abs(obs) <= wrapper.clip_obs)
    
    def test_observation_clipping(self) -> None:
        """Test observation clipping."""
        env = MockEnv()
        wrapper = TradingEnvWrapper(env, normalize_obs=False, clip_obs=2.0)
        
        # Override reset to return large values
        def mock_reset(seed=None, options=None):
            return np.array([15.0, -15.0, 3.0, -3.0, 0.0]), {}
        
        env.reset = mock_reset
        obs, _ = wrapper.reset()
        
        # Check clipping
        assert np.all(obs <= 2.0)
        assert np.all(obs >= -2.0)
        assert obs[0] == 2.0  # Clipped from 15.0
        assert obs[1] == -2.0  # Clipped from -15.0
    
    def test_reward_scaling(self) -> None:
        """Test reward scaling."""
        env = MockEnv()
        wrapper = TradingEnvWrapper(env, reward_scale=0.1)
        
        # Override step to return fixed reward
        original_step = env.step
        def mock_step(action):
            obs, _, terminated, truncated, info = original_step(action)
            return obs, 10.0, terminated, truncated, info
        
        env.step = mock_step
        
        wrapper.reset()
        _, reward, _, _, _ = wrapper.step(np.array([0.0, 0.0]))
        
        assert reward == 1.0  # 10.0 * 0.1
    
    def test_early_termination(self) -> None:
        """Test early termination with action repeat."""
        env = MockEnv()
        wrapper = TradingEnvWrapper(env, action_repeat=5)
        
        # Override step to terminate early
        def mock_step(action):
            obs = np.random.randn(5).astype(np.float32)
            return obs, 1.0, True, False, {}
        
        env.step = mock_step
        
        wrapper.reset()
        obs, reward, terminated, truncated, info = wrapper.step(np.array([0.0, 0.0]))
        
        # Should terminate after first step despite action_repeat=5
        assert terminated == True
        assert wrapper.episode_length == 1
    
    def test_get_normalization_stats(self) -> None:
        """Test getting normalization statistics."""
        env = MockEnv()
        wrapper = TradingEnvWrapper(env)
        
        # No stats initially
        stats = wrapper.get_normalization_stats()
        assert stats["n_samples"] == 0
        assert len(stats["mean"]) == 0
        assert len(stats["std"]) == 0
        
        # Add some observations
        wrapper.reset()
        for _ in range(10):
            wrapper.step(np.array([0.0, 0.0]))
        
        stats = wrapper.get_normalization_stats()
        assert stats["n_samples"] > 0
        assert stats["mean"].shape == (5,)
        assert stats["std"].shape == (5,)


class TestEpisodeMonitor:
    """Test EpisodeMonitor class."""
    
    def test_initialization(self) -> None:
        """Test monitor initialization."""
        env = MockEnv()
        monitor = EpisodeMonitor(env, log_dir="/tmp/test_logs")
        
        assert monitor.log_dir == "/tmp/test_logs"
        assert monitor.episode_count == 0
        assert len(monitor.episode_rewards) == 0
        assert len(monitor.episode_lengths) == 0
    
    def test_episode_tracking(self) -> None:
        """Test episode tracking."""
        env = MockEnv()
        monitor = EpisodeMonitor(env)
        
        # Run one episode
        monitor.reset()
        total_reward = 0
        for i in range(5):
            _, reward, terminated, truncated, info = monitor.step(np.array([0.0, 0.0]))
            total_reward += reward
            
            assert "monitor" in info
            assert info["monitor"]["episode"] == 0
            assert info["monitor"]["episode_length"] == i + 1
            
            if terminated or truncated:
                assert "episode" in info
                break
        
        # Start second episode
        monitor.reset()
        assert monitor.episode_count == 1
        assert len(monitor.episode_rewards) == 1
        assert len(monitor.episode_lengths) == 1
    
    def test_episode_statistics(self) -> None:
        """Test episode statistics calculation."""
        env = MockEnv()
        monitor = EpisodeMonitor(env)
        
        # No episodes yet
        stats = monitor.get_episode_statistics()
        assert stats["episodes"] == 0
        assert stats["mean_reward"] == 0.0
        
        # Run multiple episodes
        for ep in range(3):
            monitor.reset()
            for _ in range(10):
                _, _, terminated, truncated, _ = monitor.step(np.array([0.0, 0.0]))
                if terminated or truncated:
                    break
        
        stats = monitor.get_episode_statistics()
        assert stats["episodes"] >= 2  # At least 2 complete episodes
        assert "mean_reward" in stats
        assert "std_reward" in stats
        assert "max_reward" in stats
        assert "min_reward" in stats
    
    def test_log_file_creation(self) -> None:
        """Test episode log file creation."""
        import tempfile
        import json
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            env = MockEnv()
            monitor = EpisodeMonitor(env, log_dir=tmpdir)
            
            # Run one episode
            monitor.reset()
            for _ in range(5):
                _, _, terminated, truncated, _ = monitor.step(np.array([0.0, 0.0]))
                if terminated or truncated:
                    break
            
            # Reset to trigger logging
            monitor.reset()
            
            # Check log file exists
            log_files = list(Path(tmpdir).glob("episode_*.json"))
            assert len(log_files) == 1
            
            # Check log content
            with open(log_files[0]) as f:
                log_data = json.load(f)
                assert "episode" in log_data
                assert "reward" in log_data
                assert "length" in log_data


class TestActionNoiseWrapper:
    """Test ActionNoiseWrapper class."""
    
    def test_initialization(self) -> None:
        """Test noise wrapper initialization."""
        env = MockEnv()
        wrapper = ActionNoiseWrapper(
            env,
            noise_scale=0.2,
            noise_decay=0.99,
            min_noise=0.05
        )
        
        assert wrapper.initial_noise_scale == 0.2
        assert wrapper.noise_scale == 0.2
        assert wrapper.noise_decay == 0.99
        assert wrapper.min_noise == 0.05
        assert wrapper.episode_count == 0
    
    def test_noise_addition(self) -> None:
        """Test noise is added to actions."""
        env = MockEnv()
        wrapper = ActionNoiseWrapper(env, noise_scale=0.5)
        
        # Track actions sent to base env
        actions_received = []
        original_step = env.step
        def track_step(action):
            actions_received.append(action.copy())
            return original_step(action)
        env.step = track_step
        
        wrapper.reset()
        
        # Send same action multiple times
        base_action = np.array([0.5, -0.5])
        for _ in range(10):
            wrapper.step(base_action)
        
        # Actions should be different due to noise
        actions_array = np.array(actions_received)
        assert not np.all(actions_array == base_action)
        
        # But should be centered around base action
        mean_action = np.mean(actions_array, axis=0)
        assert np.allclose(mean_action, base_action, atol=0.2)
    
    def test_noise_decay(self) -> None:
        """Test noise decay over episodes."""
        env = MockEnv()
        wrapper = ActionNoiseWrapper(
            env,
            noise_scale=1.0,
            noise_decay=0.9,
            min_noise=0.1
        )
        
        initial_noise = wrapper.noise_scale
        
        # Run multiple episodes
        for i in range(5):
            obs, info = wrapper.reset()
            assert info["noise_scale"] == wrapper.noise_scale
            assert wrapper.noise_scale == max(initial_noise * (0.9 ** i), 0.1)
            
            # Run episode
            for _ in range(5):
                wrapper.step(np.array([0.0, 0.0]))
    
    def test_minimum_noise(self) -> None:
        """Test minimum noise constraint."""
        env = MockEnv()
        wrapper = ActionNoiseWrapper(
            env,
            noise_scale=0.2,
            noise_decay=0.5,  # Fast decay
            min_noise=0.15
        )
        
        # Run many episodes to ensure decay
        for _ in range(10):
            wrapper.reset()
        
        # Noise should not go below minimum
        assert wrapper.noise_scale >= 0.15
    
    def test_action_clipping(self) -> None:
        """Test actions are clipped to space bounds."""
        env = MockEnv()
        wrapper = ActionNoiseWrapper(env, noise_scale=5.0)  # Very high noise
        
        wrapper.reset()
        
        # Send action at boundary
        action = np.array([0.9, 0.9])
        
        # Track clipped actions
        clipped_actions = []
        original_step = env.step
        def track_step(action):
            clipped_actions.append(action.copy())
            return original_step(action)
        env.step = track_step
        
        # Step multiple times
        for _ in range(20):
            wrapper.step(action)
        
        # All actions should be within bounds
        for act in clipped_actions:
            assert np.all(act >= -1.0)
            assert np.all(act <= 1.0)