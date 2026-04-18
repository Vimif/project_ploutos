"""Helpers for driving SB3 policies consistently across PPO and RecurrentPPO."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

try:
    from sb3_contrib import RecurrentPPO

    HAS_RECURRENT = True
except ImportError:
    RecurrentPPO = None
    HAS_RECURRENT = False


def _is_recurrent(model: Any) -> bool:
    if not HAS_RECURRENT or RecurrentPPO is None:
        return False
    try:
        return isinstance(model, RecurrentPPO)
    except TypeError:
        # Happens when mocked
        return type(model).__name__ == "RecurrentPPO"


def predict_with_optional_recurrence(
    model: Any,
    observation: np.ndarray,
    *,
    deterministic: bool = True,
    recurrent_state: Optional[Any] = None,
    episode_start: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[Any]]:
    """Predict an action while preserving recurrent state when needed."""

    if _is_recurrent(model):
        if episode_start is None:
            episode_start = np.array([recurrent_state is None], dtype=bool)
        action, recurrent_state = model.predict(
            observation,
            state=recurrent_state,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        return action, recurrent_state

    res = model.predict(observation, deterministic=deterministic)
    action = res[0] if isinstance(res, tuple) else res
    return action, recurrent_state
