"""Helpers for driving SB3 policies consistently across PPO and RecurrentPPO."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from sb3_contrib import RecurrentPPO

    HAS_RECURRENT = True
except ImportError:
    RecurrentPPO = None
    HAS_RECURRENT = False


def predict_with_optional_recurrence(
    model: Any,
    observation: np.ndarray,
    *,
    deterministic: bool = True,
    recurrent_state: Any | None = None,
    episode_start: np.ndarray | None = None,
) -> tuple[np.ndarray, Any | None]:
    """Predict an action while preserving recurrent state when needed."""

    if HAS_RECURRENT and RecurrentPPO is not None and isinstance(model, RecurrentPPO):
        if episode_start is None:
            episode_start = np.array([recurrent_state is None], dtype=bool)
        action, recurrent_state = model.predict(
            observation,
            state=recurrent_state,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        return action, recurrent_state

    action, _ = model.predict(observation, deterministic=deterministic)
    return action, recurrent_state
