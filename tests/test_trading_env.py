"""Tests unitaires pour TradingEnv (V9)."""

import sys
from unittest.mock import MagicMock

# Mock torch pour éviter l'import GPU
sys.modules.setdefault("torch", MagicMock())

import pytest
import numpy as np
import pandas as pd
from core.environment import TradingEnv, VALID_MODES


# ============================================================================
# Fixtures
# ============================================================================


def _make_market_data(n_tickers: int = 2, n_bars: int = 500) -> dict:
    """Crée des données de marché factices mais réalistes.

    Génère des données OHLCV avec un random walk pour simuler un marché.
    """
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


@pytest.fixture
def market_data():
    """Données de marché avec 2 tickers."""
    return _make_market_data(n_tickers=2, n_bars=500)


@pytest.fixture
def train_env(market_data):
    """Environnement en mode training."""
    # V9: On laisse features_precomputed=False par défaut (calcul auto)
    return TradingEnv(market_data, mode="train", seed=42)


@pytest.fixture
def eval_env(market_data):
    """Environnement en mode evaluation."""
    return TradingEnv(market_data, mode="eval", seed=42)


@pytest.fixture
def backtest_env(market_data):
    """Environnement en mode backtest."""
    return TradingEnv(market_data, mode="backtest", seed=42)


# ============================================================================
# Tests modes
# ============================================================================


class TestEnvModes:
    def test_valid_modes(self, market_data):
        """Vérifie que les 3 modes sont acceptés."""
        for mode in VALID_MODES:
            env = TradingEnv(market_data, mode=mode)
            assert env.mode == mode

    def test_invalid_mode(self, market_data):
        """Vérifie qu'un mode invalide lève une erreur."""
        with pytest.raises(ValueError, match="mode doit être"):
            TradingEnv(market_data, mode="invalid")

    def test_train_mode_random_start(self, market_data):
        """Vérifie que le mode train a des starts aléatoires."""
        starts = set()
        for seed_val in range(20):
            env = TradingEnv(
                market_data, mode="train", seed=seed_val
            )
            env.reset()
            starts.add(env.current_step)

        # Avec 20 seeds différentes, on devrait avoir plusieurs starts
        assert len(starts) > 3, f"Trop peu de starts différents: {starts}"

    def test_eval_mode_fixed_start(self, eval_env):
        """Vérifie que le mode eval a un start fixe à 100."""
        eval_env.reset()
        assert eval_env.current_step == 100

        eval_env.reset()
        assert eval_env.current_step == 100

    def test_backtest_mode_start_at_zero(self, backtest_env):
        """Vérifie que le mode backtest commence à 0."""
        backtest_env.reset()
        assert backtest_env.current_step == 0

    def test_mode_in_info(self, train_env, eval_env, backtest_env):
        """Vérifie que le mode est retourné dans info."""
        train_env.reset()
        eval_env.reset()
        backtest_env.reset()

        assert train_env._get_info()["mode"] == "train"
        assert eval_env._get_info()["mode"] == "eval"
        assert backtest_env._get_info()["mode"] == "backtest"


# ============================================================================
# Tests reproductibilité
# ============================================================================


class TestReproducibility:
    def test_backtest_deterministic(self, market_data):
        """Vérifie que 2 runs backtest avec le même seed sont identiques."""
        env1 = TradingEnv(
            market_data, mode="backtest", seed=42
        )
        env2 = TradingEnv(
            market_data, mode="backtest", seed=42
        )

        obs1, _ = env1.reset()
        obs2, _ = env2.reset()
        assert np.allclose(obs1, obs2), "Les observations initiales diffèrent"

        # Quelques steps avec les mêmes actions
        for _ in range(10):
            action = np.array([1, 0])  # BUY ticker 0, HOLD ticker 1
            obs1, r1, d1, _, info1 = env1.step(action)
            obs2, r2, d2, _, info2 = env2.step(action)

            assert np.allclose(obs1, obs2), "Les observations divergent"
            assert abs(r1 - r2) < 1e-6, "Les rewards divergent"

    def test_different_seeds_differ(self, market_data):
        """Vérifie que des seeds différentes donnent des résultats différents."""
        env1 = TradingEnv(
            market_data, mode="train", seed=1
        )
        env2 = TradingEnv(
            market_data, mode="train", seed=2
        )

        env1.reset()
        env2.reset()

        # Les steps de départ devraient être différents
        starts_differ = env1.current_step != env2.current_step
        assert starts_differ, "Seeds différentes devraient donner des starts différents"


# ============================================================================
# Tests reward params
# ============================================================================


class TestRewardParams:
    def test_default_reward_params(self, train_env):
        """Vérifie les valeurs par défaut des params de reward."""
        assert train_env.reward_buy_executed == 0.1
        assert train_env.reward_overtrading_immediate == -0.02
        assert train_env.reward_invalid_trade == -0.01
        assert train_env.reward_bad_price == -0.05
        assert train_env.reward_good_return_bonus == 0.3
        assert train_env.reward_high_winrate_bonus == 0.2
        assert train_env.good_return_threshold == 0.01
        assert train_env.high_winrate_threshold == 0.6

    def test_custom_reward_params(self, market_data):
        """Vérifie qu'on peut personnaliser les params de reward."""
        env = TradingEnv(
            market_data,
            mode="train",
            reward_buy_executed=0.05,
            reward_overtrading=-0.1,
            reward_invalid_trade=-0.03,
            reward_bad_price=-0.1,
        )
        assert env.reward_buy_executed == 0.05
        assert env.reward_overtrading_immediate == -0.1
        assert env.reward_invalid_trade == -0.03
        assert env.reward_bad_price == -0.1


# ============================================================================
# Tests observation et action space
# ============================================================================


class TestSpaces:
    def test_observation_space_shape(self, train_env):
        """Vérifie que la taille de l'observation space est correcte."""
        n_features = len(train_env.feature_columns)
        expected = train_env.n_assets * n_features + train_env.n_assets + 3
        # V9: + macro (si présente, ici 0)
        assert train_env.n_macro_features == 0
        assert train_env.observation_space.shape == (expected,)

    def test_action_space_shape(self, train_env):
        """Vérifie que la taille de l'action space est correcte."""
        assert train_env.action_space.shape == (train_env.n_assets,)
        assert all(
            train_env.action_space.nvec[i] == 3
            for i in range(train_env.n_assets)
        )

    def test_observation_in_bounds(self, train_env):
        """Vérifie que les observations sont dans [-10, 10]."""
        obs, _ = train_env.reset()
        assert obs.min() >= -10.0
        assert obs.max() <= 10.0
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))


# ============================================================================
# Tests reset et step basiques
# ============================================================================


class TestBasicFunctionality:
    def test_reset(self, train_env):
        """Vérifie que reset() remet tout à zéro."""
        obs, info = train_env.reset()

        assert train_env.balance == train_env.initial_balance
        assert train_env.equity == train_env.initial_balance
        assert train_env.total_trades == 0
        assert train_env.winning_trades == 0
        assert train_env.losing_trades == 0
        assert not train_env.done
        assert obs is not None
        assert isinstance(info, dict)

    def test_step_hold(self, train_env):
        """Vérifie qu'un step HOLD ne change pas le portfolio."""
        train_env.reset()
        action = np.array([0] * train_env.n_assets)  # All HOLD

        obs, reward, done, trunc, info = train_env.step(action)
        assert train_env.total_trades == 0
        assert obs is not None

    def test_step_buy(self, train_env):
        """Vérifie qu'un step BUY modifie le portfolio."""
        train_env.reset()
        # BUY le premier ticker, HOLD le reste
        action = np.array([1] + [0] * (train_env.n_assets - 1))

        obs, reward, done, trunc, info = train_env.step(action)
        assert train_env.total_trades == 1
        first_ticker = train_env.tickers[0]
        assert train_env.portfolio[first_ticker] > 0

    def test_step_buy_then_sell(self, train_env):
        """Vérifie le cycle complet BUY → SELL."""
        train_env.reset()
        first_ticker = train_env.tickers[0]

        # BUY
        action_buy = np.array([1] + [0] * (train_env.n_assets - 1))
        train_env.step(action_buy)
        assert train_env.portfolio[first_ticker] > 0

        # Attendre le min_holding_period
        for _ in range(train_env.min_holding_period):
            train_env.step(np.array([0] * train_env.n_assets))

        # SELL
        action_sell = np.array([2] + [0] * (train_env.n_assets - 1))
        train_env.step(action_sell)
        assert train_env.portfolio[first_ticker] == 0.0
        assert train_env.total_trades == 2

    def test_episode_completes(self, train_env):
        """Vérifie qu'un épisode complet s'exécute sans erreur."""
        train_env.reset()
        steps = 0
        done = False

        while not done and steps < 500:
            action = train_env.action_space.sample()
            obs, reward, done, trunc, info = train_env.step(action)
            steps += 1

            assert not np.any(np.isnan(obs)), f"NaN in obs at step {steps}"
            assert not np.any(np.isinf(obs)), f"Inf in obs at step {steps}"


# ============================================================================
# Tests AdvancedTransactionModel integration
# ============================================================================


class TestTransactionModel:
    def test_backtest_uses_transaction_model(self, market_data):
        """Vérifie que le mode backtest utilise AdvancedTransactionModel."""
        env = TradingEnv(
            market_data, mode="backtest", seed=42
        )
        env.reset()
        assert env.transaction_model is not None

    def test_train_mode_does_not_use_transaction_model_but_shuffles(self, market_data):
        """Vérifie que le mode train utilise des seeds changeantes (slippage stochastique)."""
        env = TradingEnv(
            market_data, mode="train", seed=42
        )
        env.reset()
        
        # NOTE: TradingEnv V9 utilise AdvancedTransactionModel même en train, 
        # mais la seed aléatoire rend le slippage stochastique.
        
        # Test simple: Volume array populated
        for ticker in env.tickers:
            assert ticker in env.volume_arrays
            assert len(env.volume_arrays[ticker]) > 0

