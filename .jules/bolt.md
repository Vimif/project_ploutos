
## $(date +%Y-%m-%d) - [High-Frequency Loop Pandas Allocation Bottleneck]
**Learning:** In the high-frequency trading simulation loop (`core/environment.py`), allocating new `pd.Series` objects via `pd.Series(prices[start:end])` on every trade evaluation creates a severe performance bottleneck. Pandas object creation and its built-in statistical methods (`pct_change().dropna().std()`) have significant overhead compared to raw NumPy arrays.
**Action:** When extracting data windows for repetitive calculations in RL environments, always return native `np.ndarray` slices and use vectorized NumPy functions like `np.diff` and `np.std(ddof=1)`. Never instantiate Pandas objects within the core `step()` or trade execution loops.
