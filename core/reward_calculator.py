# core/reward_calculator.py
"""Reward calculation using Differential Sharpe Ratio (DSR) with Welford online variance."""

import numpy as np

from core.constants import DSR_VARIANCE_FLOOR


class RewardCalculator:
    """Calculates step rewards using DSR + penalty hybridation.

    Uses Welford's online algorithm for numerically stable variance estimation.
    """

    def __init__(
        self,
        reward_scaling: float = 1.5,
        use_drawdown_penalty: bool = True,
        drawdown_penalty_factor: float = 3.0,
        drawdown_threshold: float = 0.10,
        penalty_overtrading: float = 0.005,
        variance_floor: float = DSR_VARIANCE_FLOOR,
    ):
        self.reward_scaling = reward_scaling
        self.use_drawdown_penalty = use_drawdown_penalty
        self.drawdown_penalty_factor = drawdown_penalty_factor
        self.drawdown_threshold = drawdown_threshold
        self.penalty_overtrading = penalty_overtrading
        self.variance_floor = variance_floor

        self.reset()

    def reset(self):
        """Reset DSR running statistics for a new episode."""
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def calculate(
        self,
        prev_equity: float,
        current_equity: float,
        peak_value: float,
        trades_executed: int,
    ) -> float:
        """Calculate reward for one step.

        Args:
            prev_equity: Portfolio value at previous step.
            current_equity: Portfolio value at current step.
            peak_value: Historical peak portfolio value.
            trades_executed: Number of trades executed this step.

        Returns:
            Scaled reward value.
        """
        if prev_equity <= 0:
            return 0.0

        ret = (current_equity - prev_equity) / prev_equity

        # Welford online update
        old_mean = self.mean
        self.n += 1
        delta = ret - self.mean
        self.mean += delta / self.n
        delta2 = ret - self.mean
        self.m2 += delta * delta2

        # Variance with floor
        if self.n >= 2:
            variance = max(self.m2 / (self.n - 1), self.variance_floor)
        else:
            variance = self.variance_floor

        std_dev = np.sqrt(variance)
        dsr = (ret - old_mean) / std_dev
        reward = np.clip(dsr * 0.1, -1.0, 1.0)

        # Drawdown penalty
        if self.use_drawdown_penalty and peak_value > 0:
            drawdown = (peak_value - current_equity) / peak_value
            if drawdown > self.drawdown_threshold:
                reward -= drawdown * self.drawdown_penalty_factor * 0.5

        # Overtrading penalty
        if trades_executed > 0:
            reward -= self.penalty_overtrading * 0.1

        return reward * self.reward_scaling
