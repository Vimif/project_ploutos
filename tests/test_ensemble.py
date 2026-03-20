"""Tests unitaires pour EnsemblePredictor."""

import sys
from unittest.mock import MagicMock, patch

# Mock torch et stable_baselines3 pour éviter l'import GPU
for mod in [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.optim",
    "torch.utils",
    "torch.utils.data",
    "torch.distributions",
    "stable_baselines3",
    "stable_baselines3.common",
    "stable_baselines3.common.vec_env",
    "stable_baselines3.common.monitor",
    "stable_baselines3.common.callbacks",
    "sb3_contrib",
]:
    sys.modules.setdefault(mod, MagicMock())

import pytest
import numpy as np

import core.ensemble as ensemble_module
from core.ensemble import EnsemblePredictor

# sb3_contrib est mocké → RecurrentPPO est un MagicMock (pas un type).
# Désactiver HAS_RECURRENT par défaut pour éviter isinstance() crash.
ensemble_module.HAS_RECURRENT = False


# ============================================================================
# Fixtures
# ============================================================================


class MockModel:
    """Mock d'un modèle SB3 (PPO) pour les tests."""

    def __init__(self, fixed_action: int = 0):
        self.fixed_action = fixed_action

    def predict(self, obs, deterministic=True, **kwargs):
        return np.array([self.fixed_action]), None


class MockRecurrentModel:
    """Mock d'un modèle RecurrentPPO pour les tests."""

    def __init__(self, fixed_action: int = 0):
        self.fixed_action = fixed_action

    def predict(self, obs, state=None, deterministic=True, **kwargs):
        new_state = np.array([[1.0, 2.0]])  # Simulated LSTM state
        return np.array([self.fixed_action]), new_state


@pytest.fixture
def three_models():
    """3 modèles avec votes BUY(1), BUY(1), SELL(2)."""
    return [MockModel(1), MockModel(1), MockModel(2)]


@pytest.fixture
def ensemble(three_models):
    return EnsemblePredictor(models=three_models)


@pytest.fixture
def obs():
    return np.random.randn(100).astype(np.float32)


# ============================================================================
# Tests vote majoritaire
# ============================================================================


class TestMajorityVote:
    def test_majority_vote_basic(self, ensemble, obs):
        """2 BUY vs 1 SELL → BUY gagne."""
        action = ensemble.predict(obs)
        assert action[0] == 1

    def test_unanimous_vote(self, obs):
        """3 modèles unanimes → action unique."""
        models = [MockModel(0), MockModel(0), MockModel(0)]
        ens = EnsemblePredictor(models=models)
        action = ens.predict(obs)
        assert action[0] == 0

    def test_three_way_split(self, obs):
        """3 votes différents → un gagne (comportement déterministe)."""
        models = [MockModel(0), MockModel(1), MockModel(2)]
        ens = EnsemblePredictor(models=models)
        action = ens.predict(obs)
        assert action[0] in [0, 1, 2]


# ============================================================================
# Tests LSTM state propagation
# ============================================================================


class TestLSTMStatePropagation:
    def test_lstm_states_initialized_to_none(self, ensemble):
        assert all(s is None for s in ensemble.lstm_states)

    def test_reset_states(self, ensemble):
        ensemble.lstm_states = [np.array([1]), np.array([2]), np.array([3])]
        ensemble.reset_states()
        assert all(s is None for s in ensemble.lstm_states)

    def test_lstm_states_propagated_in_predict(self, obs):
        """Vérifie que les states LSTM sont mis à jour après predict."""
        models = [MockRecurrentModel(1), MockRecurrentModel(0)]
        ens = EnsemblePredictor(models=models)
        ens.lstm_states = [np.array([[0.0, 0.0]]), np.array([[0.0, 0.0]])]

        # Activer HAS_RECURRENT et définir RecurrentPPO = MockRecurrentModel
        with (
            patch.object(ensemble_module, "HAS_RECURRENT", True),
            patch.object(ensemble_module, "RecurrentPPO", MockRecurrentModel),
        ):
            ens.predict(obs)

        # States should be updated to new values
        assert ens.lstm_states[0] is not None
        np.testing.assert_array_equal(ens.lstm_states[0], np.array([[1.0, 2.0]]))

    def test_ppo_models_dont_update_states(self, ensemble, obs):
        """PPO models should not touch lstm_states."""
        ensemble.predict(obs)
        assert all(s is None for s in ensemble.lstm_states)


# ============================================================================
# Tests confidence
# ============================================================================


class TestConfidence:
    def test_confidence_unanimous(self, obs):
        models = [MockModel(1), MockModel(1), MockModel(1)]
        ens = EnsemblePredictor(models=models)
        action, confidence = ens.predict_with_confidence(obs)
        assert action[0] == 1
        assert confidence == 1.0

    def test_confidence_majority(self, obs):
        models = [MockModel(1), MockModel(1), MockModel(2)]
        ens = EnsemblePredictor(models=models)
        action, confidence = ens.predict_with_confidence(obs)
        assert action[0] == 1
        assert abs(confidence - 2 / 3) < 1e-6

    def test_confidence_split(self, obs):
        models = [MockModel(0), MockModel(1), MockModel(2)]
        ens = EnsemblePredictor(models=models)
        action, confidence = ens.predict_with_confidence(obs)
        assert abs(confidence - 1 / 3) < 1e-6

    def test_confidence_lstm_states_propagated(self, obs):
        """predict_with_confidence should also propagate LSTM states."""
        models = [MockRecurrentModel(1), MockRecurrentModel(0)]
        ens = EnsemblePredictor(models=models)
        ens.lstm_states = [np.array([[0.0, 0.0]]), np.array([[0.0, 0.0]])]

        with (
            patch.object(ensemble_module, "HAS_RECURRENT", True),
            patch.object(ensemble_module, "RecurrentPPO", MockRecurrentModel),
        ):
            ens.predict_with_confidence(obs)

        np.testing.assert_array_equal(ens.lstm_states[0], np.array([[1.0, 2.0]]))


# ============================================================================
# Tests n_models
# ============================================================================


class TestNModels:
    def test_single_model(self, obs):
        ens = EnsemblePredictor(models=[MockModel(2)])
        action = ens.predict(obs)
        assert action[0] == 2
        assert ens.n_models == 1
