import sys
import unittest
import numpy as np
import pandas as pd
import gymnasium as gym

# Assurer que le path est correct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.environment import TradingEnv
    from core.features import FeatureEngineer
    from scripts.robustness_tests import calculate_psr, calculate_dsr
except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")
    sys.exit(1)


class TestV9Preflight(unittest.TestCase):
    """Sanity checks avant migration V9."""

    def test_imports(self):
        """Vérifie que les modules clés sont importables."""
        self.assertTrue(True)
        print("✅ Imports OK")

    def test_psr_dsr(self):
        """Vérifie que les nouvelles métriques robustness fonctionnent."""
        returns = np.random.normal(0.001, 0.01, 1000)
        psr = calculate_psr(returns, benchmark_sr=0.0)
        dsr = calculate_dsr(returns, n_trials=10)
        self.assertGreaterEqual(psr, 0.0)
        self.assertLessEqual(psr, 1.0)
        self.assertGreaterEqual(dsr, 0.0)
        print(f"✅ PSR/DSR OK (PSR={psr:.4f}, DSR={dsr:.4f})")

    def test_environment_instantiation(self):
        """Vérifie qu'on peut créer un env V8 sans crash."""
        # Mock data
        dates = pd.date_range("2020-01-01", periods=200, freq="1h")
        df = pd.DataFrame(
            {
                "Open": np.random.rand(200) * 100,
                "High": np.random.rand(200) * 105,
                "Low": np.random.rand(200) * 95,
                "Close": np.random.rand(200) * 100,
                "Volume": np.random.rand(200) * 1000,
            },
            index=dates,
        )

        data = {"MOCK": df}

        try:
            env = TradingEnv(
                data=data,
                macro_data=None,
                mode="train",
                features_precomputed=False,  # On force le calcul live pour tester Pandera/Pipeline
            )
            obs, _ = env.reset()
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)
            self.assertEqual(len(obs), env.observation_space.shape[0])
            print("✅ Environment V8 Instantiation & Step OK")
        except Exception as e:
            self.fail(f"Environment crashed: {e}")


if __name__ == "__main__":
    unittest.main()
