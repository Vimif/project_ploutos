import pytest
import time
import pandas as pd
import numpy as np
from core.environment import TradingEnv
from core.features import FeatureEngineer
from core.shared_memory_manager import SharedDataManager, load_shared_data

def _get_mock_data(n=2000):
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "Open": np.random.randn(n) + 100,
        "High": np.random.randn(n) + 110,
        "Low": np.random.randn(n) + 90,
        "Close": np.random.randn(n) + 100,
        "Volume": np.abs(np.random.randn(n)) * 1000
    }, index=idx)

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
