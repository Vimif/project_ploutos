
## $(date +%Y-%m-%d) - [High-Frequency Loop Pandas Allocation Bottleneck]
**Learning:** In the high-frequency trading simulation loop (`core/environment.py`), allocating new `pd.Series` objects via `pd.Series(prices[start:end])` on every trade evaluation creates a severe performance bottleneck. Pandas object creation and its built-in statistical methods (`pct_change().dropna().std()`) have significant overhead compared to raw NumPy arrays.
**Action:** When extracting data windows for repetitive calculations in RL environments, always return native `np.ndarray` slices and use vectorized NumPy functions like `np.diff` and `np.std(ddof=1)`. Never instantiate Pandas objects within the core `step()` or trade execution loops.

## $(date +%Y-%m-%d) - [Mock Bleed Breaking Global PyTorch Types]
**Learning:** Overwriting dependencies unconditionally using `sys.modules['torch'] = MagicMock()` in test files completely replaces the real module globally during `pytest` discovery. This destroys PyTorch type structures, causing downstream E2E training tests to crash deeply within layers like `torch.nn.init.kaiming_uniform_` with `TypeError: isinstance() arg 2 must be a type...` because `float` isn't compatible with a mock.
**Action:** When mocking large dependencies like Torch or NumPy in isolated tests, ALWAYS use a conditional block (`if 'torch' not in sys.modules: sys.modules['torch'] = MagicMock()`) so that if the real library is installed (e.g., in CI), it is used instead of a mock.
