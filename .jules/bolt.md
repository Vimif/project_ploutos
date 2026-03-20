## 2026-03-20 - Optimize Slippage Volatility Calculation with NumPy

**Learning:** In high-frequency operations like RL simulation steps, wrapping NumPy arrays into `pd.Series` and using Pandas methods like `pct_change().dropna().std()` introduces severe overhead. The environment's `_get_recent_prices` allocating a Pandas Series on every step was a major bottleneck.

**Action:** Pass raw NumPy array slices directly from the environment to downstream models (like `AdvancedTransactionModel`), and use equivalent fast NumPy primitives: `np.diff(arr) / arr[:-1]` and `np.nanstd(..., ddof=1)` to achieve identical numeric results but ~5-6x faster execution time.
