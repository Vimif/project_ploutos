import time
import numpy as np
from collections import deque
from core.observation_builder import ObservationBuilder

tickers = ['AAPL', 'MSFT', 'GOOG']
feature_columns = ['close', 'volume', 'rsi']
n_steps = 1000

# Create dummy features
feature_arrays = {
    t: np.random.randn(n_steps, len(feature_columns)).astype(np.float32)
    for t in tickers
}

macro_array = np.random.randn(n_steps, 2).astype(np.float32)

builder = ObservationBuilder(
    tickers=tickers,
    feature_columns=feature_columns,
    feature_arrays=feature_arrays,
    macro_array=macro_array,
    n_macro_features=2
)

portfolio = {'AAPL': 10, 'MSFT': 5}
prices = {'AAPL': 150.0, 'MSFT': 250.0, 'GOOG': 100.0}
equity = 3000.0
balance = 250.0
initial_balance = 1000.0
peak_value = 3500.0
entry_prices = {'AAPL': 100.0, 'MSFT': 200.0}
portfolio_value_history = deque([1000.0] * 30)

start = time.time()
for i in range(10000):
    obs = builder.build(
        current_step=10,
        portfolio=portfolio,
        prices=prices,
        equity=equity,
        balance=balance,
        initial_balance=initial_balance,
        peak_value=peak_value,
        entry_prices=entry_prices,
        portfolio_value_history=portfolio_value_history
    )
print(f"Original time: {time.time() - start:.4f}s")
