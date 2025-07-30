"""Training utilities for RL models."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from collections import deque
import random


@dataclass
class TrainingConfig:
    """Training configuration."""
    learning_rate: float = 3e-4
    batch_size: int = 256
    n_epochs: int = 100
    tau_values: List[int] = None
    buffer_size: int = 100000
    gamma: float = 0.99
    tau: float = 0.005
    update_frequency: int = 1
    gradient_steps: int = 1
    
    def __post_init__(self):
        if self.tau_values is None:
            self.tau_values = [3, 6, 9, 12]


class ReplayBuffer:
    """Experience replay buffer."""
    
    def __init__(self, capacity: int = 100000):
        """Initialize buffer."""
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        **kwargs: Any
    ) -> None:
        """Add experience to buffer."""
        self.buffer.append({
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs,
            "done": done,
            **kwargs
        })
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample batch from buffer."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        return {
            "obs": np.array([e["obs"] for e in batch]),
            "actions": np.array([e["action"] for e in batch]),
            "rewards": np.array([e["reward"] for e in batch]),
            "next_obs": np.array([e["next_obs"] for e in batch]),
            "dones": np.array([e["done"] for e in batch])
        }
    
    def __len__(self) -> int:
        """Get buffer size."""
        return len(self.buffer)


class TauSACTrainer:
    """Trainer for Tau-SAC model."""
    
    def __init__(
        self,
        env: Optional[Any] = None,
        config: Optional[TrainingConfig] = None,
        log_dir: Optional[str] = None
    ):
        """Initialize trainer."""
        self.env = env
        self.config = config or TrainingConfig()
        self.log_dir = log_dir
        self.buffer = ReplayBuffer(self.config.buffer_size)
        self.model = None
        self.total_steps = 0
        self.episode_count = 0
    
    def train(self) -> None:
        """Run training loop."""
        # Mock training loop
        for epoch in range(self.config.n_epochs):
            # Collect experience
            self._collect_experience()
            
            # Update model
            if len(self.buffer) >= self.config.batch_size:
                for _ in range(self.config.gradient_steps):
                    self._update_model()
            
            # Log progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{self.config.n_epochs}")
    
    def _collect_experience(self) -> None:
        """Collect experience from environment."""
        if self.env is None:
            return
        
        obs, _ = self.env.reset()
        done = False
        
        while not done:
            # Sample action
            action = self._sample_action(obs)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store in buffer
            self.buffer.add(obs, action, reward, next_obs, done)
            
            obs = next_obs
            self.total_steps += 1
    
    def _sample_action(self, obs: np.ndarray) -> int:
        """Sample action from policy."""
        # Mock action sampling
        return np.random.randint(0, 3)
    
    def _update_model(self) -> None:
        """Update model parameters."""
        # Mock model update
        batch = self.buffer.sample(self.config.batch_size)
        # Would perform gradient updates here
    
    def save_model(self, path: str) -> None:
        """Save trained model."""
        # Mock model saving
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({"config": self.config, "steps": self.total_steps}, f)
    
    def load_model(self, path: str) -> None:
        """Load trained model."""
        # Mock model loading
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.config = data["config"]
            self.total_steps = data["steps"]


class ModelSerializer:
    """Serialize and deserialize models."""
    
    def save(self, model: Any, path: str) -> None:
        """Save model to file."""
        import torch
        torch.save(model.state_dict(), path)
    
    def load(self, model: Any, path: str) -> None:
        """Load model from file."""
        import torch
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)