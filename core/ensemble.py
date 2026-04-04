# core/ensemble.py
"""Ensemble helpers for Ploutos inference."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from sb3_contrib import RecurrentPPO

    HAS_RECURRENT = True
except ImportError:
    RecurrentPPO = None
    HAS_RECURRENT = False


class EnsemblePredictor:
    """Predictor using majority vote across multiple SB3 models."""

    def __init__(self, models: List, vecnormalize: Optional[VecNormalize] = None):
        self.models = models
        self.vecnormalize = vecnormalize
        self.n_models = len(models)
        self.lstm_states = [None] * self.n_models

    @classmethod
    def load(
        cls,
        model_paths: List[str],
        vecnorm_path: Optional[str] = None,
        env=None,
        obs_shape: Optional[tuple[int, ...]] = None,
        use_recurrent: bool = False,
        device: str = "auto",
    ) -> "EnsemblePredictor":
        """Load an ensemble from disk, including VecNormalize stats when available."""

        model_class = RecurrentPPO if (use_recurrent and HAS_RECURRENT) else PPO
        models = [model_class.load(path, device=device) for path in model_paths]

        vecnormalize = None
        if vecnorm_path and Path(vecnorm_path).exists():
            if env is None and obs_shape is not None:
                env = DummyVecEnv([lambda: _StaticObservationEnv(obs_shape)])
            if env is None:
                raise ValueError("env or obs_shape is required to load VecNormalize")
            vecnormalize = VecNormalize.load(vecnorm_path, env)
            vecnormalize.training = False
            vecnormalize.norm_reward = False

        return cls(models=models, vecnormalize=vecnormalize)

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        action, _ = self.predict_with_asset_confidences(
            observation, deterministic=deterministic
        )
        return action

    def predict_with_confidence(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, float]:
        action, confidences = self.predict_with_asset_confidences(
            observation, deterministic=deterministic
        )
        avg_confidence = float(confidences.mean()) if len(confidences) > 0 else 0.0
        return action, avg_confidence

    def predict_with_asset_confidences(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict actions and per-asset confidence scores."""

        obs = self._normalize_observation(observation)
        all_actions = self._collect_actions(obs, deterministic=deterministic)
        return self._vote_with_confidences(all_actions)

    def predict_filtered(
        self,
        observation: np.ndarray,
        min_confidence: float = 0.5,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Predict with a HOLD fallback when model agreement is too low."""

        action, confidences = self.predict_with_asset_confidences(
            observation, deterministic=deterministic
        )
        filtered = np.zeros(len(confidences), dtype=action.dtype)
        for asset_idx, confidence in enumerate(confidences):
            if confidence >= min_confidence:
                filtered[asset_idx] = action[asset_idx]
        return filtered

    def reset_states(self):
        """Reset recurrent states between independent episodes."""

        self.lstm_states = [None] * self.n_models

    def _normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        if self.vecnormalize is not None:
            return self.vecnormalize.normalize_obs(observation.reshape(1, -1)).flatten()
        return observation

    def _collect_actions(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        all_actions = []
        for idx, model in enumerate(self.models):
            if HAS_RECURRENT and RecurrentPPO is not None and isinstance(model, RecurrentPPO):
                action, state = model.predict(
                    observation,
                    state=self.lstm_states[idx],
                    deterministic=deterministic,
                )
                self.lstm_states[idx] = state
            else:
                action, _ = model.predict(observation, deterministic=deterministic)
            all_actions.append(action)
        return np.array(all_actions)

    def _vote_with_confidences(self, all_actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if all_actions.ndim == 1:
            all_actions = all_actions.reshape(-1, 1)

        final_action = np.zeros(all_actions.shape[1], dtype=all_actions.dtype)
        confidences = np.zeros(all_actions.shape[1], dtype=np.float32)
        for asset_idx in range(all_actions.shape[1]):
            votes = all_actions[:, asset_idx]
            values, counts = np.unique(votes, return_counts=True)
            best_idx = np.argmax(counts)
            final_action[asset_idx] = values[best_idx]
            confidences[asset_idx] = counts[best_idx] / self.n_models
        return final_action, confidences


class _StaticObservationEnv(gym.Env):
    """Minimal env used only for loading VecNormalize statistics."""

    metadata = {"render_modes": []}

    def __init__(self, obs_shape: tuple[int, ...]):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=obs_shape,
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, 0.0, True, False, {}
