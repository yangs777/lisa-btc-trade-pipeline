"""Test coverage for RL training module."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import json

from src.rl.training import TrainingConfig, ReplayBuffer, TauSACTrainer


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""
    
    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = TrainingConfig()
        
        assert config.learning_rate == 3e-4
        assert config.batch_size == 256
        assert config.n_epochs == 100
        assert config.tau_values == [1, 10, 100]
        assert config.gamma == 0.99
        assert config.tau_update == 0.005
        assert config.buffer_size == 1000000
        assert config.gradient_steps == 1
        assert config.learning_starts == 1000
        assert config.save_freq == 10000
    
    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = TrainingConfig(
            learning_rate=1e-3,
            batch_size=512,
            n_epochs=50,
            tau_values=[5, 20, 50],
            gamma=0.95
        )
        
        assert config.learning_rate == 1e-3
        assert config.batch_size == 512
        assert config.n_epochs == 50
        assert config.tau_values == [5, 20, 50]
        assert config.gamma == 0.95


class TestReplayBuffer:
    """Test ReplayBuffer class."""
    
    def test_initialization(self) -> None:
        """Test buffer initialization."""
        buffer = ReplayBuffer(
            buffer_size=1000,
            observation_shape=(10,),
            action_shape=(3,),
            n_envs=1
        )
        
        assert buffer.buffer_size == 1000
        assert buffer.observation_shape == (10,)
        assert buffer.action_shape == (3,)
        assert buffer.pos == 0
        assert buffer.full == False
        assert len(buffer) == 0
    
    def test_add_single_transition(self) -> None:
        """Test adding single transition."""
        buffer = ReplayBuffer(1000, (5,), (2,))
        
        obs = np.random.randn(5)
        next_obs = np.random.randn(5)
        action = np.random.randn(2)
        reward = 1.0
        done = False
        info = {"tau": 10}
        
        buffer.add(obs, next_obs, action, reward, done, [info])
        
        assert len(buffer) == 1
        assert buffer.pos == 1
        assert not buffer.full
    
    def test_buffer_overflow(self) -> None:
        """Test buffer overflow behavior."""
        buffer = ReplayBuffer(10, (2,), (1,))
        
        # Add more than buffer size
        for i in range(15):
            obs = np.array([i, i])
            next_obs = np.array([i+1, i+1])
            action = np.array([i])
            buffer.add(obs, next_obs, action, float(i), False, [{}])
        
        assert len(buffer) == 10
        assert buffer.full == True
        assert buffer.pos == 5  # Wrapped around
    
    def test_sample_basic(self) -> None:
        """Test basic sampling."""
        buffer = ReplayBuffer(100, (3,), (2,))
        
        # Add some transitions
        for i in range(50):
            obs = np.ones(3) * i
            next_obs = np.ones(3) * (i + 1)
            action = np.ones(2) * i
            buffer.add(obs, next_obs, action, float(i), False, [{"tau": i}])
        
        # Sample batch
        batch = buffer.sample(10)
        
        assert batch.observations.shape == (10, 3)
        assert batch.next_observations.shape == (10, 3)
        assert batch.actions.shape == (10, 2)
        assert batch.rewards.shape == (10,)
        assert batch.dones.shape == (10,)
        assert len(batch.infos) == 10
    
    def test_sample_with_env_indices(self) -> None:
        """Test sampling with environment indices."""
        buffer = ReplayBuffer(100, (3,), (2,), n_envs=4)
        
        # Add transitions from different environments
        for env_idx in range(4):
            for i in range(10):
                obs = np.ones(3) * (env_idx * 10 + i)
                next_obs = obs + 1
                action = np.ones(2) * env_idx
                buffer.add(obs, next_obs, action, float(env_idx), False, [{"env": env_idx}])
        
        # Sample with specific env indices
        batch = buffer.sample(20, env_indices=[0, 2])
        
        # Check that samples come from correct environments
        for info in batch.infos:
            assert info["env"] in [0, 2]
    
    def test_sample_empty_buffer(self) -> None:
        """Test sampling from empty buffer."""
        buffer = ReplayBuffer(100, (3,), (2,))
        
        with pytest.raises(ValueError):
            buffer.sample(10)
    
    def test_sample_more_than_available(self) -> None:
        """Test sampling more than available transitions."""
        buffer = ReplayBuffer(100, (3,), (2,))
        
        # Add only 5 transitions
        for i in range(5):
            buffer.add(np.zeros(3), np.zeros(3), np.zeros(2), 0.0, False, [{}])
        
        # Try to sample 10
        with pytest.raises(ValueError):
            buffer.sample(10)


class TestTauSACTrainer:
    """Test TauSACTrainer class."""
    
    def test_initialization(self) -> None:
        """Test trainer initialization."""
        with patch('src.rl.training.TauSAC') as mock_tau_sac:
            env = Mock()
            env.observation_space = Mock(shape=(10,))
            env.action_space = Mock(shape=(3,))
            
            config = TrainingConfig(tau_values=[1, 5, 10])
            trainer = TauSACTrainer(env, config)
            
            assert trainer.env == env
            assert trainer.config == config
            assert trainer.n_timesteps == 0
            assert trainer.num_episodes == 0
            assert len(trainer.episode_rewards) == 0
            
            # Check agent initialization
            mock_tau_sac.assert_called_once()
            call_args = mock_tau_sac.call_args[1]
            assert call_args["tau_values"] == [1, 5, 10]
            assert call_args["learning_rate"] == 3e-4
    
    def test_train_basic(self) -> None:
        """Test basic training loop."""
        # Setup environment mock
        env = Mock()
        env.observation_space = Mock(shape=(5,))
        env.action_space = Mock(shape=(2,))
        env.reset.return_value = (np.zeros(5), {})
        
        # Make environment terminate after 5 steps
        step_count = 0
        def mock_step(action):
            nonlocal step_count
            step_count += 1
            obs = np.random.randn(5)
            reward = np.random.randn()
            done = step_count >= 5
            return obs, reward, done, False, {}
        
        env.step = mock_step
        
        # Setup agent mock
        with patch('src.rl.training.TauSAC') as mock_tau_sac_class:
            mock_agent = Mock()
            mock_agent.predict.return_value = (np.zeros(2), None)
            mock_agent.train.return_value = None
            mock_tau_sac_class.return_value = mock_agent
            
            config = TrainingConfig(learning_starts=2)
            trainer = TauSACTrainer(env, config)
            
            # Train for a few timesteps
            trainer.train(total_timesteps=10)
            
            # Check training occurred
            assert trainer.n_timesteps == 10
            assert trainer.num_episodes >= 1
            assert len(trainer.episode_rewards) >= 1
            
            # Check agent was trained
            assert mock_agent.train.call_count > 0
    
    def test_train_with_callbacks(self) -> None:
        """Test training with callbacks."""
        env = Mock()
        env.observation_space = Mock(shape=(5,))
        env.action_space = Mock(shape=(2,))
        env.reset.return_value = (np.zeros(5), {})
        env.step.return_value = (np.zeros(5), 0.0, True, False, {})
        
        with patch('src.rl.training.TauSAC'):
            config = TrainingConfig()
            trainer = TauSACTrainer(env, config)
            
            # Setup callbacks
            callback_called = {"step": 0, "episode": 0}
            
            def on_step(trainer_instance, timestep):
                callback_called["step"] += 1
                return True  # Continue training
            
            def on_episode_end(trainer_instance, episode, reward):
                callback_called["episode"] += 1
            
            # Train with callbacks
            trainer.train(
                total_timesteps=5,
                callback_on_step=on_step,
                callback_on_episode_end=on_episode_end
            )
            
            # Check callbacks were called
            assert callback_called["step"] == 5
            assert callback_called["episode"] >= 1
    
    def test_save_and_load(self) -> None:
        """Test model saving and loading."""
        env = Mock()
        env.observation_space = Mock(shape=(5,))
        env.action_space = Mock(shape=(2,))
        
        with patch('src.rl.training.TauSAC') as mock_tau_sac_class:
            with patch('builtins.open', create=True) as mock_open:
                with patch('src.rl.training.json.dump') as mock_json_dump:
                    mock_agent = Mock()
                    mock_agent.state_dict.return_value = {"weights": [1, 2, 3]}
                    mock_tau_sac_class.return_value = mock_agent
                    
                    config = TrainingConfig()
                    trainer = TauSACTrainer(env, config)
                    
                    # Save model
                    trainer.save("model.pkl")
                    
                    # Check file operations
                    mock_open.assert_called_with("model.pkl", "wb")
                    
                    # Test loading
                    with patch('src.rl.training.json.load') as mock_json_load:
                        mock_json_load.return_value = {
                            "agent_state": {"weights": [1, 2, 3]},
                            "config": config.__dict__,
                            "training_info": {"n_timesteps": 1000}
                        }
                        
                        loaded_trainer = TauSACTrainer.load("model.pkl", env)
                        
                        assert loaded_trainer is not None
                        mock_agent.load_state_dict.assert_called()
    
    def test_evaluate(self) -> None:
        """Test model evaluation."""
        env = Mock()
        env.observation_space = Mock(shape=(5,))
        env.action_space = Mock(shape=(2,))
        env.reset.return_value = (np.zeros(5), {})
        
        # Make deterministic environment
        rewards_sequence = [1.0, 2.0, 3.0, -1.0, 0.5]
        step_idx = 0
        
        def mock_step(action):
            nonlocal step_idx
            reward = rewards_sequence[step_idx % len(rewards_sequence)]
            step_idx += 1
            done = step_idx % 5 == 0
            return np.zeros(5), reward, done, False, {}
        
        env.step = mock_step
        
        with patch('src.rl.training.TauSAC') as mock_tau_sac_class:
            mock_agent = Mock()
            mock_agent.predict.return_value = (np.zeros(2), None)
            mock_tau_sac_class.return_value = mock_agent
            
            config = TrainingConfig()
            trainer = TauSACTrainer(env, config)
            
            # Evaluate for multiple episodes
            mean_reward, std_reward = trainer.evaluate(n_episodes=3)
            
            # Check evaluation results
            expected_mean = sum(rewards_sequence) / len(rewards_sequence)
            assert abs(mean_reward - expected_mean) < 0.1
            assert std_reward >= 0
    
    def test_get_training_stats(self) -> None:
        """Test getting training statistics."""
        env = Mock()
        env.observation_space = Mock(shape=(5,))
        env.action_space = Mock(shape=(2,))
        
        with patch('src.rl.training.TauSAC'):
            config = TrainingConfig()
            trainer = TauSACTrainer(env, config)
            
            # Add some episode rewards
            trainer.episode_rewards = [10.0, 20.0, 15.0, 25.0, 30.0]
            trainer.episode_lengths = [100, 150, 120, 180, 200]
            trainer.n_timesteps = 750
            trainer.num_episodes = 5
            
            stats = trainer.get_training_stats()
            
            assert stats["total_timesteps"] == 750
            assert stats["total_episodes"] == 5
            assert stats["mean_reward"] == 20.0
            assert stats["std_reward"] > 0
            assert stats["mean_length"] == 150.0
            assert stats["last_10_rewards"] == [10.0, 20.0, 15.0, 25.0, 30.0]