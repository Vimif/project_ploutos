import sys
from pathlib import Path

# Ajouter la racine du projet au PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pytest


# ============================================================================
# Shared data generators
# ============================================================================


def make_market_data(n_tickers: int = 2, n_bars: int = 500) -> dict:
    """Create synthetic OHLCV market data for testing."""
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


def make_macro_data(n_bars: int = 500) -> pd.DataFrame:
    """Create synthetic macro data (VIX, TNX, DXY) for testing."""
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


def make_ohlcv(ticker: str = "TEST", n_bars: int = 300, seed: int = 42) -> pd.DataFrame:
    """Create a single ticker OHLCV DataFrame for testing."""
    np.random.seed(seed)
    dates = pd.date_range("2023-01-01", periods=n_bars, freq="h")
    base_price = 150.0
    returns = np.random.randn(n_bars) * 0.01
    prices = base_price * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "Open": prices * (1 + np.random.rand(n_bars) * 0.005),
            "High": prices * (1 + abs(np.random.randn(n_bars)) * 0.01),
            "Low": prices * (1 - abs(np.random.randn(n_bars)) * 0.01),
            "Close": prices,
            "Volume": np.random.randint(500_000, 20_000_000, n_bars),
        },
        index=dates,
    )


# ============================================================================
# Setup / Teardown Hooks
# ============================================================================


def pytest_runtest_setup(item):
    """
    When mocking heavy dependencies like torch or stable_baselines3 via sys.modules
    to avoid GPU imports in unit tests, teardown_module hooks may fail to prevent
    cross-file test pollution during E2E runs. This resolves 'TypeError: isinstance()
    arg 2 must be a type' during E2E tests by identifying and popping MagicMock
    instances of these modules from sys.modules.
    """
    if "e2e" in str(item.fspath):
        from unittest.mock import MagicMock

        to_pop = []
        for mod_name, mod in sys.modules.items():
            if isinstance(mod, MagicMock) and any(
                pkg in mod_name for pkg in ["torch", "stable_baselines3"]
            ):
                to_pop.append(mod_name)
        for mod_name in to_pop:
            sys.modules.pop(mod_name, None)


def pytest_runtest_teardown(item):
    if "e2e" in str(item.fspath):
        from unittest.mock import MagicMock

        to_pop = []
        for mod_name, mod in sys.modules.items():
            if isinstance(mod, MagicMock) and any(
                pkg in mod_name for pkg in ["torch", "stable_baselines3"]
            ):
                to_pop.append(mod_name)
        for mod_name in to_pop:
            sys.modules.pop(mod_name, None)


# ============================================================================
# Shared pytest fixtures
# ============================================================================


@pytest.fixture
def market_data():
    """Two-ticker market data for environment tests."""
    return make_market_data(n_tickers=2, n_bars=500)


@pytest.fixture
def macro_data():
    """Synthetic macro data (VIX/TNX/DXY) for environment tests."""
    return make_macro_data(n_bars=500)


@pytest.fixture
def single_ohlcv():
    """Single ticker OHLCV for unit tests."""
    return make_ohlcv()
