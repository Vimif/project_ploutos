# core/ensemble.py
"""Ensemble Learning - Vote Majoritaire de N modèles.

Entraîne 3-5 modèles identiques avec des seeds différentes.
Pour chaque décision, vote à la majorité.

Avantage: lisse les erreurs individuelles, augmente la fiabilité.

Usage:
    from core.ensemble import EnsemblePredictor

    predictor = EnsemblePredictor.load(
        model_paths=['model_seed42.zip', 'model_seed123.zip', 'model_seed456.zip'],
        vecnorm_path='vecnormalize.pkl',
    )
    action = predictor.predict(observation)
"""

import numpy as np
from typing import List, Optional
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

try:
    from sb3_contrib import RecurrentPPO

    HAS_RECURRENT = True
except ImportError:
    HAS_RECURRENT = False


class EnsemblePredictor:
    """Prédicteur par vote majoritaire de N modèles RL."""

    def __init__(self, models: List, vecnormalize: Optional[VecNormalize] = None):
        """
        Args:
            models: Liste de modèles SB3 (PPO ou RecurrentPPO).
            vecnormalize: VecNormalize pour normaliser les observations.
        """
        self.models = models
        self.vecnormalize = vecnormalize
        self.n_models = len(models)

        # Pour RecurrentPPO : states par modèle
        self.lstm_states = [None] * self.n_models

    @classmethod
    def load(
        cls,
        model_paths: List[str],
        vecnorm_path: Optional[str] = None,
        env=None,
        use_recurrent: bool = False,
        device: str = "auto",
    ) -> "EnsemblePredictor":
        """Charge N modèles depuis le disque.

        Args:
            model_paths: Chemins vers les fichiers .zip des modèles.
            vecnorm_path: Chemin vers le VecNormalize .pkl.
            env: Environnement (nécessaire pour charger VecNormalize).
            use_recurrent: Si True, charge avec RecurrentPPO.
            device: 'auto', 'cpu', ou 'cuda'.
        """
        ModelClass = RecurrentPPO if (use_recurrent and HAS_RECURRENT) else PPO

        models = []
        for path in model_paths:
            model = ModelClass.load(path, device=device)
            models.append(model)

        vecnormalize = None
        if vecnorm_path and Path(vecnorm_path).exists() and env is not None:
            vecnormalize = VecNormalize.load(vecnorm_path, env)
            vecnormalize.training = False
            vecnormalize.norm_reward = False

        return cls(models=models, vecnormalize=vecnormalize)

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Prédit l'action par vote majoritaire.

        Args:
            observation: Observation de l'environnement.
            deterministic: Si True, actions déterministes (recommandé pour prod).

        Returns:
            Action choisie par vote majoritaire.
        """
        # Normaliser si nécessaire
        if self.vecnormalize is not None:
            obs = self.vecnormalize.normalize_obs(observation.reshape(1, -1)).flatten()
        else:
            obs = observation

        # Collecter les votes de chaque modèle
        all_actions = []
        for i, model in enumerate(self.models):
            if HAS_RECURRENT and isinstance(model, RecurrentPPO):
                # RecurrentPPO avec state
                action, state = model.predict(
                    obs, state=self.lstm_states[i], deterministic=deterministic
                )
                self.lstm_states[i] = state
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
            all_actions.append(action)

        # Vote majoritaire par action (chaque asset a son propre vote)
        all_actions = np.array(all_actions)  # shape: (n_models, n_assets)

        if all_actions.ndim == 1:
            # Un seul asset
            from scipy import stats

            result = stats.mode(all_actions, keepdims=False)
            return np.array([result.mode])

        # Multi-assets : vote par colonne
        final_action = np.zeros(all_actions.shape[1], dtype=all_actions.dtype)
        for asset_idx in range(all_actions.shape[1]):
            votes = all_actions[:, asset_idx]
            values, counts = np.unique(votes, return_counts=True)
            final_action[asset_idx] = values[np.argmax(counts)]

        return final_action

    def predict_with_confidence(self, observation: np.ndarray, deterministic: bool = True) -> tuple:
        """Prédit avec un score de confiance (unanimité du vote).

        Returns:
            (action, confidence) où confidence est le % de modèles d'accord.
        """
        if self.vecnormalize is not None:
            obs = self.vecnormalize.normalize_obs(observation.reshape(1, -1)).flatten()
        else:
            obs = observation

        all_actions = []
        for i, model in enumerate(self.models):
            if HAS_RECURRENT and isinstance(model, RecurrentPPO):
                action, state = model.predict(
                    obs, state=self.lstm_states[i], deterministic=deterministic
                )
                self.lstm_states[i] = state
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
            all_actions.append(action)

        all_actions = np.array(all_actions)

        if all_actions.ndim == 1:
            values, counts = np.unique(all_actions, return_counts=True)
            best_idx = np.argmax(counts)
            confidence = counts[best_idx] / self.n_models
            return np.array([values[best_idx]]), confidence

        final_action = np.zeros(all_actions.shape[1], dtype=all_actions.dtype)
        confidences = np.zeros(all_actions.shape[1])

        for asset_idx in range(all_actions.shape[1]):
            votes = all_actions[:, asset_idx]
            values, counts = np.unique(votes, return_counts=True)
            best_idx = np.argmax(counts)
            final_action[asset_idx] = values[best_idx]
            confidences[asset_idx] = counts[best_idx] / self.n_models

        avg_confidence = float(confidences.mean())
        return final_action, avg_confidence

    def reset_states(self):
        """Reset LSTM states (appeler au début d'un nouvel épisode)."""
        self.lstm_states = [None] * self.n_models
