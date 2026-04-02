# ruff: noqa: E402
"""Lightweight tests for the paper trading decision wrapper."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np


for mod in [
    "stable_baselines3",
    "stable_baselines3.common",
    "stable_baselines3.common.vec_env",
    "dotenv",
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()


from scripts.paper_trade import get_model_actions


class MockModel:
    def __init__(self):
        self.last_obs = None

    def predict(self, obs, deterministic=True):
        self.last_obs = obs
        return np.array([1]), None


class MockEnv:
    observation_space = SimpleNamespace(shape=(2,))

    def __init__(self, data=None, **kwargs):
        self.data = data
        self.kwargs = kwargs

    def reset(self):
        return np.array([1.0, 2.0], dtype=np.float32), {"mode": "test"}


class BadShapeEnv(MockEnv):
    observation_space = SimpleNamespace(shape=(3,))


class TestGetModelActions:
    def test_gymnasium_reset_tuple_is_unwrapped_without_vecnorm(self):
        model = MockModel()

        action = get_model_actions(
            model,
            data={"TEST": object()},
            tickers=["TEST"],
            env_class=MockEnv,
            env_params={"mode": "backtest"},
            model_obs_size=2,
            vecnorm_path=None,
        )

        np.testing.assert_array_equal(action, np.array([1]))
        np.testing.assert_array_equal(model.last_obs, np.array([1.0, 2.0], dtype=np.float32))

    def test_observation_mismatch_returns_none(self):
        model = MockModel()

        action = get_model_actions(
            model,
            data={"TEST": object()},
            tickers=["TEST"],
            env_class=BadShapeEnv,
            env_params={},
            model_obs_size=2,
            vecnorm_path=None,
        )

        assert action is None
