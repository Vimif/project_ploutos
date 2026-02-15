import time

import numpy as np
import pandas as pd
import polars as pl

from core.environment import TradingEnv
from core.features import FeatureEngineer
from core.shared_memory_manager import SharedDataManager


def _get_mock_data(n=2000):
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "Open": np.random.randn(n) + 100,
            "High": np.random.randn(n) + 110,
            "Low": np.random.randn(n) + 90,
            "Close": np.random.randn(n) + 100,
            "Volume": np.abs(np.random.randn(n)) * 1000,
        },
        index=idx,
    )


class TestV9Integration:
    def test_polars_features_speed(self):
        """Vérifie que Polars est rapide (<0.5s pour 2000 rows)."""
        fe = FeatureEngineer()
        df = _get_mock_data(2000)

        t0 = time.time()
        res = fe.calculate_all_features(df)
        dt = time.time() - t0

        assert "rsi" in res.columns
        assert dt < 0.5, f"Trop lent: {dt:.4f}s"

    def test_shared_memory_workflow(self):
        """Vérifie le cycle complet SHM -> Env."""
        # 1. Calc
        fe = FeatureEngineer()
        df = _get_mock_data(500)
        data = {"MOCK": fe.calculate_all_features(df)}

        # 2. SHM
        mgr = SharedDataManager()
        try:
            meta = mgr.put_data(data)

            # 3. Env
            env = TradingEnv(meta, mode="train", features_precomputed=True)
            obs, _ = env.reset()
            assert not np.isnan(obs).any()

            obs, _, _, _, _ = env.step(env.action_space.sample())
            assert not np.isnan(obs).any()

        finally:
            mgr.cleanup()

    def test_calculate_features_without_datetime_index(self):
        # Create a DataFrame with integer index, no datetime
        df = pd.DataFrame({
            "Open": [10, 11, 12, 13, 14],
            "High": [12, 13, 14, 15, 16],
            "Low": [9, 10, 11, 12, 13],
            "Close": [11, 12, 13, 14, 15],
            "Volume": [100, 200, 300, 400, 500]
        })

        fe = FeatureEngineer()

        # This should NOT crash
        res = fe.calculate_all_features(df)

        assert isinstance(res, pd.DataFrame)
        assert len(res) == 5
        assert "__date_idx" not in res.columns
        # Check if some features are calculated
        assert "rsi" in res.columns

    def test_calculate_features_polars_input_no_date(self):
        # Polars input without date column
        df = pl.DataFrame({
            "Open": [10, 11, 12, 13, 14],
            "High": [12, 13, 14, 15, 16],
            "Low": [9, 10, 11, 12, 13],
            "Close": [11, 12, 13, 14, 15],
            "Volume": [100, 200, 300, 400, 500]
        })

        fe = FeatureEngineer()

        # This should NOT crash
        res = fe.calculate_all_features(df, return_pandas=False)

        assert isinstance(res, pl.DataFrame)
        assert len(res) == 5
        assert "__date_idx" not in res.columns
        assert "rsi" in res.columns
