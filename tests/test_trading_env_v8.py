"""Tests unitaires pour TradingEnv."""

import sys
from unittest.mock import MagicMock

# Mock torch pour éviter l'import GPU
sys.modules.setdefault("torch", MagicMock())

import pytest
import numpy as np
import pandas as pd
from core.environment import (
    TradingEnv,
    VALID_MODES,
)

# ============================================================================
# Fixtures
# ============================================================================


def _make_market_data(n_tickers: int = 2, n_bars: int = 500) -> dict:
    """Crée des données de marché factices mais réalistes."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n_bars, freq="h")
    data = {}
    ticker_names = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"][:n_tickers]

    for ticker in ticker_names:
        base_price = np.random.uniform(100, 500)
        returns = np.random.randn(n_bars) * 0.01
        prices = base_price * np.exp(np.cumsum(returns))

        data[ticker] = pd.DataFrame(
            {
                "Open": prices * (1 + np.random.rand(n_bars) * 0.005),
                "High": prices * (1 + abs(np.random.randn(n_bars)) * 0.01),
                "Low": prices * (1 - abs(np.random.randn(n_bars)) * 0.01),
                "Close": prices,
                "Volume": np.random.randint(500_000, 20_000_000, n_bars),
            },
            index=dates,
        )
    return data


def _make_macro_data(n_bars: int = 500) -> pd.DataFrame:
    """Crée des données macro factices (VIX, TNX, DXY)."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n_bars, freq="h")
    return pd.DataFrame(
        {
            "vix_close": 15 + np.random.randn(n_bars) * 3,
            "tnx_close": 4.0 + np.random.randn(n_bars) * 0.3,
            "dxy_close": 104 + np.random.randn(n_bars) * 1.5,
            "vix_return": np.random.randn(n_bars) * 0.02,
            "tnx_return": np.random.randn(n_bars) * 0.01,
            "dxy_return": np.random.randn(n_bars) * 0.005,
            "vix_ma_20": 15 + np.random.randn(n_bars) * 1,
            "tnx_ma_20": 4.0 + np.random.randn(n_bars) * 0.1,
            "dxy_ma_20": 104 + np.random.randn(n_bars) * 0.5,
            "vix_std_20": np.abs(np.random.randn(n_bars) * 0.5) + 0.1,
            "tnx_std_20": np.abs(np.random.randn(n_bars) * 0.1) + 0.01,
            "dxy_std_20": np.abs(np.random.randn(n_bars) * 0.3) + 0.1,
            "vix_zscore": np.random.randn(n_bars),
            "tnx_zscore": np.random.randn(n_bars),
            "dxy_zscore": np.random.randn(n_bars),
            "vix_fear": (np.random.randn(n_bars) > 1).astype(float),
            "tnx_rising": (np.random.randn(n_bars) > 0).astype(float),
            "dxy_strong": (np.random.randn(n_bars) > 0.5).astype(float),
        },
        index=dates,
    )


@pytest.fixture
def market_data():
    return _make_market_data(n_tickers=2, n_bars=500)


@pytest.fixture
def macro_data():
    return _make_macro_data(n_bars=500)


@pytest.fixture
def train_env(market_data):
    return TradingEnv(market_data, mode="train", seed=42)


@pytest.fixture
def eval_env(market_data):
    return TradingEnv(market_data, mode="eval", seed=42)


@pytest.fixture
def backtest_env(market_data):
    return TradingEnv(market_data, mode="backtest", seed=42)


@pytest.fixture
def env_with_macro(market_data, macro_data):
    return TradingEnv(market_data, macro_data=macro_data, mode="train", seed=42)


# ============================================================================
# Tests modes
# ============================================================================


class TestEnvModes:
    def test_valid_modes(self, market_data):
        for mode in VALID_MODES:
            env = TradingEnv(market_data, mode=mode)
            assert env.mode == mode

    def test_invalid_mode(self, market_data):
        with pytest.raises(ValueError, match="mode doit être"):
            TradingEnv(market_data, mode="invalid")

    def test_train_mode_random_start(self, market_data):
        starts = set()
        for seed_val in range(20):
            env = TradingEnv(market_data, mode="train", seed=seed_val)
            env.reset()
            starts.add(env.current_step)
        assert len(starts) > 3, f"Trop peu de starts différents: {starts}"

    def test_eval_mode_fixed_start(self, eval_env):
        eval_env.reset()
        assert eval_env.current_step == 100
        eval_env.reset()
        assert eval_env.current_step == 100

    def test_backtest_mode_start_at_zero(self, backtest_env):
        backtest_env.reset()
        assert backtest_env.current_step == 0

    def test_mode_in_info(self, train_env, eval_env, backtest_env):
        train_env.reset()
        eval_env.reset()
        backtest_env.reset()
        assert train_env._get_info()["mode"] == "train"
        assert eval_env._get_info()["mode"] == "eval"
        assert backtest_env._get_info()["mode"] == "backtest"


# ============================================================================
# Tests observation
# ============================================================================


class TestObservation:
    def test_observation_space_shape(self, train_env):
        n_features = len(train_env.feature_columns)
        expected = train_env.n_assets * n_features + train_env.n_assets + 3
        assert train_env.observation_space.shape == (expected,)

    def test_observation_space_shape_with_macro(self, env_with_macro):
        n_features = len(env_with_macro.feature_columns)
        n_macro = env_with_macro.n_macro_features
        expected = env_with_macro.n_assets * n_features + n_macro + env_with_macro.n_assets + 3
        assert env_with_macro.observation_space.shape == (expected,)

    def test_action_space_shape(self, train_env):
        assert train_env.action_space.shape == (train_env.n_assets,)
        assert all(train_env.action_space.nvec[i] == 3 for i in range(train_env.n_assets))

    def test_observation_in_bounds(self, train_env):
        obs, _ = train_env.reset()
        assert obs.min() >= -10.0
        assert obs.max() <= 10.0
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))

    def test_observation_dtype(self, train_env):
        obs, _ = train_env.reset()
        assert obs.dtype == np.float32


# ============================================================================
# Tests macro data
# ============================================================================


class TestMacroData:
    def test_env_without_macro(self, train_env):
        assert train_env.n_macro_features == 0
        assert train_env.macro_array is None

    def test_env_with_macro_has_features(self, env_with_macro):
        assert env_with_macro.n_macro_features > 0
        assert env_with_macro.macro_array is not None

    def test_obs_size_differs_with_macro(self, market_data, macro_data):
        env_no_macro = TradingEnv(market_data, mode="train", seed=42)
        env_macro = TradingEnv(
            market_data, macro_data=macro_data, mode="train", seed=42
        )
        assert env_macro.observation_space.shape[0] > env_no_macro.observation_space.shape[0]


# ============================================================================
# Tests features precomputed
# ============================================================================


class TestFeaturesPrecomputed:
    def test_precomputed_flag_stored(self, market_data):
        env = TradingEnv(market_data, mode="train", features_precomputed=True)
        assert env.features_precomputed is True

    def test_precomputed_vs_computed_same_columns(self, market_data):
        from core.features import FeatureEngineer

        fe = FeatureEngineer()
        precomputed_data = {
            ticker: fe.calculate_all_features(df.copy()) for ticker, df in market_data.items()
        }

        env_raw = TradingEnv(
            market_data, mode="train", seed=42, features_precomputed=False
        )
        env_pre = TradingEnv(
            precomputed_data, mode="train", seed=42, features_precomputed=True
        )

        assert len(env_raw.feature_columns) == len(env_pre.feature_columns)
        assert set(env_raw.feature_columns) == set(env_pre.feature_columns)


# ============================================================================
# Tests DSR reward
# ============================================================================


class TestDSR:
    def test_dsr_no_nan(self, train_env):
        train_env.reset()
        for _ in range(50):
            action = train_env.action_space.sample()
            obs, reward, done, trunc, info = train_env.step(action)
            assert not np.isnan(reward), "DSR reward should not be NaN"
            if done:
                break

    def test_dsr_bounded(self, train_env):
        train_env.reset()
        for _ in range(50):
            action = train_env.action_space.sample()
            obs, reward, done, trunc, info = train_env.step(action)
            assert -10 <= reward <= 10, f"Reward {reward} out of bounds"
            if done:
                break


# ============================================================================
# Tests slippage consistency
# ============================================================================


class TestSlippage:
    def test_slippage_consistent_across_modes(self, market_data):
        """After bug fix, train and backtest both use AdvancedTransactionModel."""
        env_train = TradingEnv(market_data, mode="train", seed=42)
        env_back = TradingEnv(market_data, mode="backtest", seed=42)
        env_train.reset()
        env_back.reset()

        ticker = env_train.tickers[0]
        price = 150.0

        # Both modes should use the transaction model (not simple random)
        assert env_train.transaction_model is not None
        assert env_back.transaction_model is not None

    def test_slippage_none_returns_price(self, market_data):
        env = TradingEnv(market_data, mode="train", seed=42, slippage_model="none")
        env.reset()
        ticker = env.tickers[0]
        price = 150.0
        assert env._apply_slippage_buy(ticker, price) == price
        assert env._apply_slippage_sell(ticker, price) == price


# ============================================================================
# Tests trade execution
# ============================================================================


class TestTradeExecution:
    def test_reset(self, train_env):
        obs, info = train_env.reset()
        assert train_env.balance == train_env.initial_balance
        assert train_env.equity == train_env.initial_balance
        assert train_env.total_trades == 0
        assert not train_env.done
        assert obs is not None
        assert isinstance(info, dict)

    def test_step_hold(self, train_env):
        train_env.reset()
        action = np.array([0] * train_env.n_assets)
        obs, reward, done, trunc, info = train_env.step(action)
        assert train_env.total_trades == 0

    def test_step_buy(self, train_env):
        train_env.reset()
        action = np.array([1] + [0] * (train_env.n_assets - 1))
        obs, reward, done, trunc, info = train_env.step(action)
        assert train_env.total_trades == 1
        first_ticker = train_env.tickers[0]
        assert train_env.portfolio[first_ticker] > 0

    def test_step_buy_then_sell(self, train_env):
        train_env.reset()
        first_ticker = train_env.tickers[0]

        # BUY
        action_buy = np.array([1] + [0] * (train_env.n_assets - 1))
        train_env.step(action_buy)
        assert train_env.portfolio[first_ticker] > 0

        # Wait min_holding_period
        for _ in range(train_env.min_holding_period):
            train_env.step(np.array([0] * train_env.n_assets))

        # SELL
        action_sell = np.array([2] + [0] * (train_env.n_assets - 1))
        train_env.step(action_sell)
        assert train_env.portfolio[first_ticker] == 0.0
        assert train_env.total_trades == 2

    def test_min_holding_period_enforced(self, train_env):
        train_env.reset()
        first_ticker = train_env.tickers[0]

        # BUY
        train_env.step(np.array([1] + [0] * (train_env.n_assets - 1)))
        trades_after_buy = train_env.total_trades

        # Immediate SELL (violates min_holding_period)
        train_env.step(np.array([2] + [0] * (train_env.n_assets - 1)))
        # Should not execute sell, position still held
        assert train_env.portfolio[first_ticker] > 0
        assert train_env.total_trades == trades_after_buy


# ============================================================================
# Tests episode complete
# ============================================================================


class TestEpisodeComplete:
    def test_episode_runs_without_crash(self, train_env):
        train_env.reset()
        steps = 0
        done = False
        while not done and steps < 500:
            action = train_env.action_space.sample()
            obs, reward, done, trunc, info = train_env.step(action)
            steps += 1
            assert not np.any(np.isnan(obs)), f"NaN in obs at step {steps}"
            assert not np.any(np.isinf(obs)), f"Inf in obs at step {steps}"

    def test_episode_with_macro(self, env_with_macro):
        env_with_macro.reset()
        steps = 0
        done = False
        while not done and steps < 200:
            action = env_with_macro.action_space.sample()
            obs, reward, done, trunc, info = env_with_macro.step(action)
            steps += 1
            assert not np.any(np.isnan(obs)), f"NaN in obs at step {steps}"


# ============================================================================
# Tests reproductibilité
# ============================================================================


class TestReproducibility:
    def test_backtest_deterministic(self, market_data):
        env1 = TradingEnv(market_data, mode="backtest", seed=42)
        env2 = TradingEnv(market_data, mode="backtest", seed=42)

        obs1, _ = env1.reset()
        obs2, _ = env2.reset()
        assert np.allclose(obs1, obs2), "Initial observations differ"

        for _ in range(10):
            action = np.array([1, 0])
            obs1, r1, d1, _, info1 = env1.step(action)
            obs2, r2, d2, _, info2 = env2.step(action)
            assert np.allclose(obs1, obs2), "Observations diverge"
            assert abs(r1 - r2) < 1e-6, "Rewards diverge"
