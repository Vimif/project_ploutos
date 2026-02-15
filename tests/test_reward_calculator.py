"""Tests for RewardCalculator (DSR + penalties)."""

import numpy as np
import pytest
from core.reward_calculator import RewardCalculator


@pytest.fixture
def calc():
    return RewardCalculator()


class TestBasicReward:
    def test_positive_return_positive_reward(self, calc):
        # Build some history first
        for _ in range(5):
            calc.calculate(100000, 100100, 100100, 0)
        reward = calc.calculate(100000, 100500, 100500, 0)
        assert reward > 0

    def test_zero_prev_equity_returns_zero(self, calc):
        reward = calc.calculate(0.0, 100000, 100000, 0)
        assert reward == 0.0

    def test_reward_not_nan(self, calc):
        for i in range(50):
            equity = 100000 + np.random.randn() * 100
            reward = calc.calculate(100000, equity, max(100000, equity), 0)
            assert not np.isnan(reward)
            assert not np.isinf(reward)


class TestWelfordStability:
    def test_flat_market_no_explosion(self, calc):
        """With constant returns, DSR shouldn't explode."""
        rewards = []
        for _ in range(100):
            r = calc.calculate(100000, 100000, 100000, 0)
            rewards.append(r)
        assert all(abs(r) < 50 for r in rewards)

    def test_variance_floor_prevents_division_by_zero(self):
        calc = RewardCalculator(variance_floor=1e-2)
        # All identical returns
        for _ in range(10):
            r = calc.calculate(100000, 100000, 100000, 0)
            assert not np.isnan(r)


class TestPenalties:
    def test_drawdown_penalty_applied(self):
        calc = RewardCalculator(
            use_drawdown_penalty=True,
            drawdown_threshold=0.05,
            drawdown_penalty_factor=3.0,
        )
        # Build history
        for _ in range(5):
            calc.calculate(100000, 100100, 100100, 0)
        # Big drawdown: equity 80k vs peak 100k = 20% drawdown
        reward = calc.calculate(100000, 80000, 100000, 0)
        assert reward < 0

    def test_no_drawdown_penalty_when_disabled(self):
        calc = RewardCalculator(use_drawdown_penalty=False)
        for _ in range(5):
            calc.calculate(100000, 100100, 100100, 0)
        r_no_dd = calc.calculate(100000, 80000, 100000, 0)

        calc2 = RewardCalculator(use_drawdown_penalty=True, drawdown_penalty_factor=10.0)
        for _ in range(5):
            calc2.calculate(100000, 100100, 100100, 0)
        r_dd = calc2.calculate(100000, 80000, 100000, 0)

        assert r_no_dd > r_dd

    def test_overtrading_penalty(self, calc):
        for _ in range(5):
            calc.calculate(100000, 100100, 100100, 0)
        r_no_trade = calc.calculate(100000, 100200, 100200, 0)

        calc2 = RewardCalculator()
        for _ in range(5):
            calc2.calculate(100000, 100100, 100100, 0)
        r_with_trade = calc2.calculate(100000, 100200, 100200, 3)

        assert r_no_trade > r_with_trade


class TestReset:
    def test_reset_clears_state(self, calc):
        for _ in range(10):
            calc.calculate(100000, 100100, 100100, 0)
        assert calc.n > 0
        calc.reset()
        assert calc.n == 0
        assert calc.mean == 0.0
        assert calc.m2 == 0.0
