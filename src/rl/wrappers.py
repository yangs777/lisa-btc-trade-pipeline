"""Environment wrappers for training stability and monitoring."""

from typing import Any

import gymnasium as gym
import numpy as np


class TradingEnvWrapper(gym.Wrapper):
    """Wrapper for trading environment with additional features.

    Provides:
    - Observation normalization
    - Action scaling
    - Episode monitoring
    - Safety checks
    """

    def __init__(
        self,
        env: gym.Env,
        normalize_obs: bool = True,
        clip_obs: float = 10.0,
        action_repeat: int = 1,
        reward_scale: float = 1.0,
    ):
        """Initialize wrapper.

        Args:
            env: Base trading environment
            normalize_obs: Whether to normalize observations
            clip_obs: Maximum absolute value for observation clipping
            action_repeat: Number of times to repeat each action
            reward_scale: Scaling factor for rewards
        """
        super().__init__(env)

        self.normalize_obs = normalize_obs
        self.clip_obs = clip_obs
        self.action_repeat = action_repeat
        self.reward_scale = reward_scale

        # Track statistics for normalization
        self.obs_mean = None
        self.obs_std = None
        self.n_obs = 0

        # Episode tracking
        self.episode_rewards: list[float] = []
        self.episode_length = 0

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset environment and wrapper state."""
        obs, info = self.env.reset(seed=seed, options=options)

        # Reset episode tracking
        self.episode_rewards = []
        self.episode_length = 0

        # Process observation
        obs = self._process_observation(obs)

        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Step through environment with action repeat and processing."""
        # Accumulate rewards over action repeat
        total_reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}

        for _ in range(self.action_repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)

            self.episode_rewards.append(float(reward))
            self.episode_length += 1

            if terminated or truncated:
                break

        # Scale reward
        total_reward *= self.reward_scale

        # Process observation
        obs = self._process_observation(obs)

        # Add wrapper info
        info["wrapper"] = {
            "episode_length": self.episode_length,
            "episode_reward": sum(self.episode_rewards),
            "raw_reward": total_reward / self.reward_scale,
        }

        return obs, total_reward, terminated, truncated, info

    def _process_observation(self, obs: np.ndarray) -> np.ndarray:
        """Process observation with normalization and clipping."""
        # Update running statistics
        if self.normalize_obs:
            self._update_stats(obs)

            # Normalize if we have enough samples
            if self.n_obs > 100:
                if self.obs_mean is not None and self.obs_std is not None:
                    # Avoid division by zero
                    std = self.obs_std.copy()
                    std[std < 1e-8] = 1.0
                    obs = (obs - self.obs_mean) / std

        # Clip observations
        obs = np.clip(obs, -self.clip_obs, self.clip_obs)

        return obs.astype(np.float32)

    def _update_stats(self, obs: np.ndarray) -> None:
        """Update running mean and std for normalization."""
        self.n_obs += 1

        if self.obs_mean is None:
            self.obs_mean = obs.copy()
            self.obs_std = np.zeros_like(obs)
        else:
            # Incremental update
            delta = obs - self.obs_mean
            self.obs_mean += delta / self.n_obs
            self.obs_std += delta * (obs - self.obs_mean)

    def get_normalization_stats(self) -> dict[str, np.ndarray]:
        """Get current normalization statistics."""
        if self.n_obs > 1:
            std = np.sqrt(self.obs_std / (self.n_obs - 1))
        else:
            std = np.ones_like(self.obs_mean) if self.obs_mean is not None else None

        return {
            "mean": self.obs_mean.copy() if self.obs_mean is not None else None,
            "std": std,
            "n_samples": int(self.n_obs),
        }


class EpisodeMonitor(gym.Wrapper):
    """Monitor and log episode statistics."""

    def __init__(self, env: gym.Env, log_dir: str | None = None):
        """Initialize monitor.

        Args:
            env: Environment to monitor
            log_dir: Directory to save episode logs
        """
        super().__init__(env)
        self.log_dir = log_dir
        self.episode_count = 0
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.current_reward = 0.0
        self.current_length = 0

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset and log previous episode if complete."""
        # Log previous episode
        if self.current_length > 0:
            self.episode_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_length)
            self.episode_count += 1

            # Log to file if specified
            if self.log_dir:
                self._log_episode()

        # Reset tracking
        self.current_reward = 0.0
        self.current_length = 0

        return self.env.reset(seed=seed, options=options)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Step and track rewards/length."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.current_reward += float(reward)
        self.current_length += 1

        # Add monitoring info
        info["monitor"] = {
            "episode": self.episode_count,
            "episode_reward": self.current_reward,
            "episode_length": self.current_length,
        }

        # Add episode summary when done
        if terminated or truncated:
            info["episode"] = {
                "r": self.current_reward,
                "l": self.current_length,
                "t": info.get("TimeLimit.truncated", truncated),
            }

        return obs, reward, terminated, truncated, info

    def _log_episode(self) -> None:
        """Log episode to file."""
        import json
        from pathlib import Path

        log_path = Path(self.log_dir) / f"episode_{self.episode_count}.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, "w") as f:
            json.dump(
                {
                    "episode": self.episode_count,
                    "reward": self.current_reward,
                    "length": self.current_length,
                },
                f,
            )

    def get_episode_statistics(self) -> dict[str, Any]:
        """Get statistics over all episodes."""
        if not self.episode_rewards:
            return {
                "episodes": 0,
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "mean_length": 0.0,
                "std_length": 0.0,
            }

        return {
            "episodes": self.episode_count,
            "mean_reward": np.mean(self.episode_rewards),
            "std_reward": np.std(self.episode_rewards),
            "mean_length": np.mean(self.episode_lengths),
            "std_length": np.std(self.episode_lengths),
            "max_reward": np.max(self.episode_rewards),
            "min_reward": np.min(self.episode_rewards),
        }


class ActionNoiseWrapper(gym.Wrapper):
    """Add exploration noise to actions during training."""

    def __init__(
        self,
        env: gym.Env,
        noise_scale: float = 0.1,
        noise_decay: float = 0.999,
        min_noise: float = 0.01,
    ):
        """Initialize noise wrapper.

        Args:
            env: Base environment
            noise_scale: Initial noise standard deviation
            noise_decay: Decay factor per episode
            min_noise: Minimum noise level
        """
        super().__init__(env)
        self.initial_noise_scale = noise_scale
        self.noise_scale = noise_scale
        self.noise_decay = noise_decay
        self.min_noise = min_noise
        self.episode_count = 0

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset and decay noise."""
        # Decay noise
        self.noise_scale = max(self.noise_scale * self.noise_decay, self.min_noise)
        self.episode_count += 1

        obs, info = self.env.reset(seed=seed, options=options)
        info["noise_scale"] = self.noise_scale

        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Add noise to action before stepping."""
        # Add Gaussian noise
        noise = np.random.normal(0, self.noise_scale, size=action.shape)
        noisy_action = action + noise

        # Clip to action space bounds
        if hasattr(self.action_space, 'low') and hasattr(self.action_space, 'high'):
            noisy_action = np.clip(noisy_action, self.action_space.low, self.action_space.high)

        return self.env.step(noisy_action)
