import sys

with open("core/transaction_costs.py", "rb") as f:
    content = f.read()

# Update imports
content = content.replace(b"from typing import Dict, Tuple\r\n", b"from typing import Dict, Tuple, Union\r\n")
content = content.replace(b"from typing import Dict, Tuple\n", b"from typing import Dict, Tuple, Union\n")

# Update calculate_execution_price signature
content = content.replace(b"recent_prices: pd.Series = None) -> Tuple[float, Dict]:", b"recent_prices: Union[pd.Series, np.ndarray] = None) -> Tuple[float, Dict]:")

# Update _calculate_slippage signature
content = content.replace(b"def _calculate_slippage(self, ticker: str, recent_prices: pd.Series = None) -> float:", b"def _calculate_slippage(self, ticker: str, recent_prices: Union[pd.Series, np.ndarray] = None) -> float:")

# Update estimate_total_cost signature
content = content.replace(b"recent_prices: pd.Series = None) -> Dict:", b"recent_prices: Union[pd.Series, np.ndarray] = None) -> Dict:")

# Update volatility calculation
old_calc_unix = b"""        # Calculer volatilit\xc3\xa9 r\xc3\xa9cente (20 p\xc3\xa9riodes)
        returns = recent_prices.pct_change().dropna()
        volatility = returns.std()"""

old_calc_dos = b"""        # Calculer volatilit\xc3\xa9 r\xc3\xa9cente (20 p\xc3\xa9riodes)\r
        returns = recent_prices.pct_change().dropna()\r
        volatility = returns.std()"""

new_calc = b"""        # Calculer volatilit\xc3\xa9 r\xc3\xa9cente (20 p\xc3\xa9riodes) avec NumPy pour la performance
        if isinstance(recent_prices, pd.Series):
            vals = recent_prices.values
        else:
            vals = recent_prices

        returns = np.diff(vals) / vals[:-1]
        volatility = np.nanstd(returns, ddof=1)"""

content = content.replace(old_calc_unix, new_calc)
content = content.replace(old_calc_dos, new_calc.replace(b'\n', b'\r\n'))

with open("core/transaction_costs.py", "wb") as f:
    f.write(content)

print("Patched.")
