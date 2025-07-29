"""τ-SAC (Temperature-aware Soft Actor-Critic) implementation for trading."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback


class TradingFeatureExtractor(BaseFeaturesExtractor):
    """Custom feature extractor for trading observations.
    
    Uses separate pathways for market features and position features
    with attention mechanism for better feature integration.
    """
    
    def __init__(
        self,
        observation_space: Any,
        features_dim: int = 256,
        n_position_features: int = 7,
    ):
        """Initialize feature extractor.
        
        Args:
            observation_space: Observation space
            features_dim: Output feature dimension
            n_position_features: Number of position-related features
        """
        super().__init__(observation_space, features_dim)
        
        n_input = observation_space.shape[0]
        n_market = n_input - n_position_features
        
        # Market feature pathway
        self.market_net = nn.Sequential(
            nn.Linear(n_market, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        
        # Position feature pathway
        self.position_net = nn.Sequential(
            nn.Linear(n_position_features, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        
        # Attention layer for feature integration
        self.attention = nn.MultiheadAttention(
            embed_dim=192,  # 128 + 64
            num_heads=4,
            dropout=0.1,
        )
        
        # Final projection
        self.final_net = nn.Sequential(
            nn.Linear(192, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract features from observations."""
        # Split market and position features
        market_features = observations[:, :-7]
        position_features = observations[:, -7:]
        
        # Process through separate pathways
        market_encoded = self.market_net(market_features)
        position_encoded = self.position_net(position_features)
        
        # Concatenate features
        combined = torch.cat([market_encoded, position_encoded], dim=1)
        
        # Apply self-attention
        combined = combined.unsqueeze(0)  # Add sequence dimension
        attended, _ = self.attention(combined, combined, combined)
        attended = attended.squeeze(0)  # Remove sequence dimension
        
        # Final projection
        features = self.final_net(attended)
        
        return features


class TemperatureCallback(BaseCallback):
    """Callback for dynamic temperature adjustment in τ-SAC."""
    
    def __init__(
        self,
        initial_temp: float = 1.0,
        min_temp: float = 0.1,
        temp_decay: float = 0.995,
        performance_threshold: float = 0.1,
        verbose: int = 0,
    ):
        """Initialize temperature callback.
        
        Args:
            initial_temp: Starting temperature
            min_temp: Minimum temperature
            temp_decay: Temperature decay rate
            performance_threshold: Performance improvement threshold for decay
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.temp_decay = temp_decay
        self.performance_threshold = performance_threshold
        
        self.current_temp = initial_temp
        self.best_reward = -float('inf')
        self.episodes_since_improvement = 0
    
    def _on_step(self) -> bool:
        """Update temperature based on performance."""
        # Check for episode end
        dones = self.locals.get("dones", [])
        if not any(dones):
            return True
        
        # Get episode rewards
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                episode_reward = info["episode"]["r"]
                
                # Check for improvement
                if episode_reward > self.best_reward * (1 + self.performance_threshold):
                    self.best_reward = episode_reward
                    self.episodes_since_improvement = 0
                else:
                    self.episodes_since_improvement += 1
                
                # Decay temperature if no improvement
                if self.episodes_since_improvement > 10:
                    self.current_temp = max(
                        self.current_temp * self.temp_decay,
                        self.min_temp
                    )
                    self.episodes_since_improvement = 0
                
                # Update SAC temperature
                if hasattr(self.model, "ent_coef"):
                    self.model.ent_coef = self.current_temp
                
                # Log
                if self.verbose > 0:
                    print(f"Episode reward: {episode_reward:.2f}, "
                          f"Temperature: {self.current_temp:.3f}")
        
        return True


class TauSACTrader:
    """τ-SAC trader with enhanced exploration and risk management."""
    
    def __init__(
        self,
        env: Any,
        learning_rate: float = 3e-4,
        batch_size: int = 256,
        buffer_size: int = int(1e6),
        tau: float = 0.005,
        gamma: float = 0.99,
        initial_temperature: float = 1.0,
        use_sde: bool = True,
        device: str = "auto",
    ):
        """Initialize τ-SAC trader.
        
        Args:
            env: Trading environment
            learning_rate: Learning rate for all networks
            batch_size: Batch size for training
            buffer_size: Replay buffer size
            tau: Soft update coefficient
            gamma: Discount factor
            initial_temperature: Initial entropy temperature
            use_sde: Use State Dependent Exploration
            device: Device for training (auto, cpu, cuda)
        """
        self.env = env
        self.device = device
        
        # Create custom policy kwargs
        policy_kwargs = {
            "features_extractor_class": TradingFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 256},
            "net_arch": {
                "pi": [256, 256],  # Actor network
                "qf": [256, 256],  # Critic networks
            },
            "activation_fn": nn.ReLU,
            "use_sde": use_sde,
            "log_std_init": -3,  # Lower initial noise
            "use_expln": True,  # Use exponential for log_std
            "clip_mean": 2.0,  # Clip actor output
            "normalize_images": False,
        }
        
        # Initialize SAC model
        self.model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=1000,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=1,
            gradient_steps=1,
            action_noise=None,  # Using SAC's inherent exploration
            ent_coef=initial_temperature,
            target_update_interval=1,
            target_entropy="auto",
            use_sde=use_sde,
            sde_sample_freq=4,
            policy_kwargs=policy_kwargs,
            tensorboard_log=None,
            verbose=1,
            device=device,
        )
        
        # Temperature callback
        self.temp_callback = TemperatureCallback(
            initial_temp=initial_temperature,
            min_temp=0.1,
            temp_decay=0.995,
        )
    
    def train(
        self,
        total_timesteps: int,
        log_interval: int = 100,
        eval_env: Optional[Any] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        save_path: Optional[str] = None,
    ) -> None:
        """Train the τ-SAC model.
        
        Args:
            total_timesteps: Total training timesteps
            log_interval: Logging frequency
            eval_env: Environment for evaluation
            eval_freq: Evaluation frequency
            n_eval_episodes: Number of evaluation episodes
            save_path: Path to save model checkpoints
        """
        # Setup callbacks
        callbacks = [self.temp_callback]
        
        # Add evaluation callback if eval env provided
        if eval_env is not None:
            from stable_baselines3.common.callbacks import EvalCallback
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=save_path if save_path else "./logs/",
                log_path=save_path if save_path else "./logs/",
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False,
            )
            callbacks.append(eval_callback)
        
        # Train model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=log_interval,
            progress_bar=True,
        )
        
        # Save final model
        if save_path:
            self.save(f"{save_path}/final_model")
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Tuple[torch.Tensor, ...]]]:
        """Predict action for given observation.
        
        Args:
            observation: Current observation
            deterministic: Use deterministic policy
            
        Returns:
            Tuple of (action, states)
        """
        return self.model.predict(observation, deterministic=deterministic)
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        self.model.save(path)
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        self.model = SAC.load(path, env=self.env, device=self.device)
    
    def get_trading_metrics(self, test_env: Any, n_episodes: int = 10) -> Dict[str, float]:
        """Evaluate trading performance on test environment.
        
        Args:
            test_env: Test environment
            n_episodes: Number of test episodes
            
        Returns:
            Dictionary of performance metrics
        """
        all_metrics = []
        
        for _ in range(n_episodes):
            obs, _ = test_env.reset()
            done = False
            
            while not done:
                action, _ = self.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = test_env.step(action)
                done = terminated or truncated
            
            # Get episode metrics
            if hasattr(test_env, 'get_episode_summary'):
                metrics = test_env.get_episode_summary()
                all_metrics.append(metrics)
        
        # Aggregate metrics
        if all_metrics:
            aggregated = {}
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics]
                aggregated[f"mean_{key}"] = np.mean(values)
                aggregated[f"std_{key}"] = np.std(values)
            return aggregated
        
        return {}