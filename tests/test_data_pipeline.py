"""Tests du pipeline de données avec split temporel."""

import sys
from unittest.mock import MagicMock

# Mock torch pour éviter l'import GPU
sys.modules.setdefault("torch", MagicMock())

import numpy as np
import pandas as pd
import pytest

from core.data_pipeline import DataSplitter

# ============================================================================
# Fixtures
# ============================================================================


def _make_fake_data(n_tickers: int = 3, n_bars: int = 1000) -> dict:
    """Crée des données factices multi-ticker."""
    dates = pd.date_range("2023-01-01", periods=n_bars, freq="h")
    data = {}
    for i in range(n_tickers):
        ticker = f"TICK{i}"
        prices = 100 + np.random.randn(n_bars).cumsum()
        data[ticker] = pd.DataFrame(
            {
                "Open": prices + np.random.rand(n_bars),
                "High": prices + abs(np.random.randn(n_bars)),
                "Low": prices - abs(np.random.randn(n_bars)),
                "Close": prices,
                "Volume": np.random.randint(100_000, 10_000_000, n_bars),
            },
            index=dates,
        )
    return data


@pytest.fixture
def sample_data():
    return _make_fake_data(n_tickers=3, n_bars=1000)


@pytest.fixture
def small_data():
    return _make_fake_data(n_tickers=2, n_bars=100)


# ============================================================================
# Tests
# ============================================================================


class TestDataSplitter:
    def test_split_ratios(self, sample_data):
        """Vérifie que les splits respectent les ratios demandés."""
        splits = DataSplitter.split(sample_data, 0.6, 0.2, 0.2)

        assert len(splits.train["TICK0"]) == 600
        assert len(splits.val["TICK0"]) == 200
        assert len(splits.test["TICK0"]) == 200

    def test_default_ratios(self, sample_data):
        """Vérifie les ratios par défaut 60/20/20."""
        splits = DataSplitter.split(sample_data)
        total = len(splits.train["TICK0"]) + len(splits.val["TICK0"]) + len(splits.test["TICK0"])
        assert total == 1000

    def test_no_temporal_overlap(self, sample_data):
        """Vérifie qu'il n'y a aucun chevauchement temporel entre splits."""
        splits = DataSplitter.split(sample_data)
        assert DataSplitter.validate_no_overlap(splits) is True

        # Vérification manuelle
        for ticker in sample_data:
            train_end = splits.train[ticker].index[-1]
            val_start = splits.val[ticker].index[0]
            val_end = splits.val[ticker].index[-1]
            test_start = splits.test[ticker].index[0]

            assert train_end < val_start, "Train/Val overlap!"
            assert val_end < test_start, "Val/Test overlap!"

    def test_all_tickers_present(self, sample_data):
        """Vérifie que chaque split contient tous les tickers."""
        splits = DataSplitter.split(sample_data)

        for ticker in sample_data:
            assert ticker in splits.train
            assert ticker in splits.val
            assert ticker in splits.test

    def test_chronological_order(self, sample_data):
        """Vérifie que train < val < test temporellement."""
        splits = DataSplitter.split(sample_data)
        ref = "TICK0"

        train_max = splits.train[ref].index.max()
        val_min = splits.val[ref].index.min()
        val_max = splits.val[ref].index.max()
        test_min = splits.test[ref].index.min()

        assert train_max < val_min
        assert val_max < test_min

    def test_info_structure(self, sample_data):
        """Vérifie la structure du dict info."""
        splits = DataSplitter.split(sample_data)
        info = splits.info

        assert "train" in info
        assert "val" in info
        assert "test" in info
        assert "total_bars" in info
        assert "n_tickers" in info
        assert "tickers" in info

        assert info["total_bars"] == 1000
        assert info["n_tickers"] == 3
        assert info["train"]["n_bars"] == 600
        assert info["val"]["n_bars"] == 200
        assert info["test"]["n_bars"] == 200

    def test_invalid_ratios_sum(self, sample_data):
        """Vérifie qu'on lève une erreur si les ratios ne somment pas à 1."""
        with pytest.raises(ValueError, match="ratios doivent sommer"):
            DataSplitter.split(sample_data, 0.5, 0.2, 0.2)

    def test_empty_data(self):
        """Vérifie qu'on lève une erreur avec des données vides."""
        with pytest.raises(ValueError, match="data est vide"):
            DataSplitter.split({})

    def test_too_few_bars(self):
        """Vérifie qu'on lève une erreur avec trop peu de données."""
        data = _make_fake_data(n_tickers=1, n_bars=5)
        with pytest.raises(ValueError, match="Pas assez de données"):
            DataSplitter.split(data)

    def test_unequal_ticker_lengths(self):
        """Vérifie que les tickers de longueurs différentes sont gérés."""
        dates_long = pd.date_range("2023-01-01", periods=1000, freq="h")
        dates_short = pd.date_range("2023-01-01", periods=800, freq="h")

        data = {
            "LONG": pd.DataFrame({"Close": np.random.randn(1000)}, index=dates_long),
            "SHORT": pd.DataFrame({"Close": np.random.randn(800)}, index=dates_short),
        }

        splits = DataSplitter.split(data)
        # Doit utiliser le plus court (800)
        assert splits.info["total_bars"] == 800
        assert len(splits.train["LONG"]) == len(splits.train["SHORT"])

    def test_custom_ratios(self, sample_data):
        """Vérifie que des ratios personnalisés fonctionnent."""
        splits = DataSplitter.split(sample_data, 0.7, 0.15, 0.15)

        assert len(splits.train["TICK0"]) == 700
        assert len(splits.val["TICK0"]) == 150
        assert len(splits.test["TICK0"]) == 150
