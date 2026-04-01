# ruff: noqa: E402
"""Tests for recurrent/non-recurrent model prediction support."""

from unittest.mock import patch

import numpy as np

import core.model_support as model_support
from core.model_support import predict_with_optional_recurrence


class MockModel:
    def __init__(self):
        self.last_obs = None

    def predict(self, obs, deterministic=True):
        self.last_obs = obs
        return np.array([1]), None


class MockRecurrentModel:
    def __init__(self):
        self.last_state = None
        self.last_episode_start = None

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        self.last_state = state
        self.last_episode_start = episode_start
        return np.array([2]), {"hidden": 1}


class TestPredictWithOptionalRecurrence:
    def test_standard_model_predicts_without_state(self):
        model = MockModel()
        obs = np.array([0.1, 0.2], dtype=np.float32)

        action, recurrent_state = predict_with_optional_recurrence(model, obs)

        np.testing.assert_array_equal(action, np.array([1]))
        assert recurrent_state is None
        np.testing.assert_array_equal(model.last_obs, obs)

    def test_recurrent_model_propagates_state_and_episode_start(self):
        model = MockRecurrentModel()
        obs = np.array([0.3, 0.4], dtype=np.float32)
        episode_start = np.array([True], dtype=bool)

        with (
            patch.object(model_support, "HAS_RECURRENT", True),
            patch.object(model_support, "RecurrentPPO", MockRecurrentModel),
        ):
            action, recurrent_state = predict_with_optional_recurrence(
                model,
                obs,
                deterministic=False,
                recurrent_state={"hidden": 0},
                episode_start=episode_start,
            )

        np.testing.assert_array_equal(action, np.array([2]))
        assert recurrent_state == {"hidden": 1}
        assert model.last_state == {"hidden": 0}
        np.testing.assert_array_equal(model.last_episode_start, episode_start)
