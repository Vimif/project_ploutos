
import unittest
import numpy as np
import pandas as pd
import multiprocessing as mp
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.shared_memory_manager import SharedDataManager, load_shared_data

class TestSharedMemory(unittest.TestCase):
    def test_shared_data_integrity(self):
        """Vérifie que les données stockées sont identiques aux données rechargées."""
        
        # 1. Créer données dummy
        df = pd.DataFrame({
            "A": np.random.rand(100).astype(np.float32),
            "B": np.random.rand(100).astype(np.float32)
        }, index=pd.date_range("2020-01-01", periods=100))
        
        data_src = {"TEST": df}
        
        # 2. Stocker en Shared Memory (Processus Parent)
        manager = SharedDataManager()
        try:
            metadata = manager.put_data(data_src)
            
            # 3. Recharger (Simuler Processus Enfant)
            data_dst = load_shared_data(metadata)
            
            # 4. Vérifier égalité (Manuelle pour debug)
            original_sum = data_src["TEST"].values.sum()
            reloaded_sum = data_dst["TEST"].values.sum()
            print(f"Original Sum: {original_sum}, Reloaded Sum: {reloaded_sum}")
            self.assertAlmostEqual(original_sum, reloaded_sum, places=5)
            
            val0_orig = data_src["TEST"].iloc[0, 0]
            val0_dest = data_dst["TEST"].iloc[0, 0]
            print(f"Val[0]: {val0_orig} vs {val0_dest}")
            self.assertAlmostEqual(val0_orig, val0_dest, places=5)
            
            print("✅ Data Integrity Check passed (Manual)")
            
        finally:
            manager.cleanup()

if __name__ == "__main__":
    unittest.main()
