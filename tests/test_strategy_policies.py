import numpy as np
import pandas as pd

from training.strategy_policies import (
    PPOEnsembleStrategyPolicy,
    RuleMomentumRegimeStrategyPolicy,
    StrategyContext,
    build_strategy_policy,
)


class _FakeVecNorm:
    def normalize_obs(self, obs):
        return obs


class _FakeModel:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.int64)

    def predict(self, obs, deterministic=True, state=None, episode_start=None):
        del obs, deterministic, state, episode_start
        return self.action, None


def test_build_strategy_policy_supports_all_declared_families():
    config = {"live": {"min_confidence": 0.67}, "strategy": {"ensemble_size": 3}}

    families = [
        "ppo_single",
        "ppo_ensemble",
        "recurrent_ppo",
        "supervised_ranker",
        "rule_momentum_regime",
    ]
    built = [build_strategy_policy(family, config) for family in families]

    assert [policy.family for policy in built] == families


def test_ppo_ensemble_filters_low_confidence_buy_signals():
    config = {"live": {"min_confidence": 0.67}, "strategy": {"ensemble_size": 3}}
    policy = PPOEnsembleStrategyPolicy(config)
    policy.models = [
        _FakeModel([1, 2]),
        _FakeModel([1, 2]),
        _FakeModel([0, 2]),
    ]
    policy.vecnorms = [_FakeVecNorm(), _FakeVecNorm(), _FakeVecNorm()]
    policy._recurrent_states = [None, None, None]

    context = StrategyContext(
        observation=np.array([0.0, 1.0], dtype=np.float32),
        current_step=0,
        tickers=["AAA", "BBB"],
        prices={"AAA": 100.0, "BBB": 100.0},
        portfolio={"AAA": 0.0, "BBB": 1.0},
        entry_prices={"AAA": 0.0, "BBB": 100.0},
        processed_data={},
        feature_columns=[],
        macro_columns=[],
        macro_row=None,
        interval="1h",
        regime_risk_on=True,
    )

    actions = policy.predict_actions(context)

    assert actions.tolist() == [0, 2]


def test_rule_policy_prefers_positive_trend_when_regime_is_risk_on():
    close = pd.Series(np.linspace(100, 120, 80))
    processed = {"AAA": pd.DataFrame({"Close": close})}
    policy = RuleMomentumRegimeStrategyPolicy(
        {"strategy": {"rule_fast_ma": 5, "rule_slow_ma": 20, "rule_momentum_lookback": 3}}
    )

    context = StrategyContext(
        observation=np.array([0.0], dtype=np.float32),
        current_step=79,
        tickers=["AAA"],
        prices={"AAA": 120.0},
        portfolio={"AAA": 0.0},
        entry_prices={"AAA": 0.0},
        processed_data=processed,
        feature_columns=[],
        macro_columns=[],
        macro_row=None,
        interval="1h",
        regime_risk_on=True,
    )

    actions = policy.predict_actions(context)

    assert actions.tolist() == [1]
