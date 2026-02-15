"""Tests unitaires pour FeatureEngineer."""

import numpy as np
import pandas as pd
import pytest

from core.features import FeatureEngineer

# ============================================================================
# Fixtures
# ============================================================================


def _make_ohlcv(n_bars: int = 500) -> pd.DataFrame:
    """Crée des données OHLCV factices avec random walk."""
    np.random.seed(42)
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


@pytest.fixture
def df():
    return _make_ohlcv(500)


@pytest.fixture
def fe():
    return FeatureEngineer()


@pytest.fixture
def df_with_features(df, fe):
    return fe.calculate_all_features(df)


# ============================================================================
# Tests feature count
# ============================================================================


class TestFeatureCount:
    def test_adds_features(self, df, df_with_features):
        """calculate_all_features ajoute des colonnes."""
        assert len(df_with_features.columns) > len(df.columns)

    def test_original_columns_preserved(self, df, df_with_features):
        """Les colonnes OHLCV originales sont préservées."""
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in df_with_features.columns

    def test_expected_feature_groups(self, df_with_features):
        """Vérifie que les groupes de features principaux sont présents."""
        columns = df_with_features.columns
        # Support/Resistance
        assert any("support" in c for c in columns)
        assert any("resistance" in c for c in columns)
        # Mean reversion
        assert any("zscore" in c for c in columns)
        assert any("oversold" in c for c in columns)
        # Volume
        assert "vol_ratio" in columns
        assert "vol_spike" in columns
        # Price action
        assert "hammer" in columns
        assert "bullish_engulfing" in columns
        # Divergences
        assert "rsi" in columns
        assert "bullish_divergence" in columns
        # Bollinger
        assert "bb_width" in columns
        assert "bb_squeeze" in columns
        # Entry score
        assert "buy_score" in columns
        assert "sell_score" in columns
        assert "entry_signal" in columns
        # Momentum
        assert any("roc_" in c for c in columns)
        # Trend
        assert "adx" in columns
        # Volatility
        assert "atr" in columns
        assert "atr_pct" in columns


# ============================================================================
# Tests no NaN
# ============================================================================


class TestNoNaN:
    def test_no_nan_in_output(self, df_with_features):
        """Aucun NaN dans le DataFrame final."""
        assert not df_with_features.isnull().any().any(), (
            f"NaN found in columns: "
            f"{df_with_features.columns[df_with_features.isnull().any()].tolist()}"
        )

    def test_no_inf_in_output(self, df_with_features):
        """Aucun Inf dans le DataFrame final."""
        numeric_cols = df_with_features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert not np.any(np.isinf(df_with_features[col].values)), f"Inf found in column: {col}"


# ============================================================================
# Tests no look-ahead
# ============================================================================


class TestNoLookAhead:
    def test_features_dont_change_with_future_data(self, fe):
        """Les features à l'instant T ne changent pas si on ajoute des données futures."""
        df_long = _make_ohlcv(500)
        df_short = df_long.iloc[:200].copy()

        result_short = fe.calculate_all_features(df_short)
        result_long = fe.calculate_all_features(df_long)

        # Comparer les features au step 150 (assez loin du début pour le warmup)
        check_idx = 150
        exclude_cols = ["Open", "High", "Low", "Close", "Volume"]
        feature_cols = [c for c in result_short.columns if c not in exclude_cols]

        for col in feature_cols:
            val_short = result_short[col].iloc[check_idx]
            val_long = result_long[col].iloc[check_idx]
            assert abs(val_short - val_long) < 1e-6, (
                f"Look-ahead detected in '{col}': short={val_short}, long={val_long}"
            )


# ============================================================================
# Tests edge cases
# ============================================================================


class TestEdgeCases:
    def test_short_data(self, fe):
        """Ne crash pas avec peu de données (< 20 bars)."""
        df_short = _make_ohlcv(15)
        result = fe.calculate_all_features(df_short)
        assert len(result) == 15
        assert not result.isnull().any().any()

    def test_zero_volume(self, fe):
        """Ne crash pas avec volume = 0."""
        df = _make_ohlcv(100)
        df["Volume"] = 0
        result = fe.calculate_all_features(df)
        assert not result.isnull().any().any()

    def test_constant_price(self, fe):
        """Ne crash pas avec prix constant (volatilité = 0)."""
        dates = pd.date_range("2023-01-01", periods=100, freq="h")
        df = pd.DataFrame(
            {
                "Open": 100.0,
                "High": 100.0,
                "Low": 100.0,
                "Close": 100.0,
                "Volume": 1_000_000,
            },
            index=dates,
        )
        result = fe.calculate_all_features(df)
        assert not result.isnull().any().any()

    def test_idempotent(self, fe, df):
        """Appeler 2 fois donne le même résultat."""
        result1 = fe.calculate_all_features(df.copy())
        result2 = fe.calculate_all_features(df.copy())
        pd.testing.assert_frame_equal(result1, result2)
