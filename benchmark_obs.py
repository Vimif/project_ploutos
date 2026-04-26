import time
import numpy as np
from collections import deque
from core.observation_builder import ObservationBuilder
from core.constants import OBSERVATION_CLIP_RANGE

tickers = [f"TICK{i}" for i in range(10)]
n_features = 30
feature_arrays = {t: np.random.randn(1000, n_features).astype(np.float32) for t in tickers}
macro_array = np.random.randn(1000, 5).astype(np.float32)

builder = ObservationBuilder(
    tickers=tickers,
    feature_columns=[f"F{i}" for i in range(n_features)],
    feature_arrays=feature_arrays,
    macro_array=macro_array,
    n_macro_features=5
)

portfolio = {t: 10.0 for t in tickers}
prices = {t: 100.0 for t in tickers}
entry_prices = {t: 90.0 for t in tickers}
portfolio_value_history = deque(np.linspace(9000, 10000, 30))

def run_bench():
    start = time.time()
    for _ in range(10000):
        builder.build(
            current_step=500,
            portfolio=portfolio,
            prices=prices,
            equity=10000.0,
            balance=1000.0,
            initial_balance=10000.0,
            peak_value=11000.0,
            entry_prices=entry_prices,
            portfolio_value_history=portfolio_value_history
        )
    return time.time() - start

# Warmup
run_bench()

# Measure
t = run_bench()
print(f"Time for 10k calls: {t:.4f}s")
