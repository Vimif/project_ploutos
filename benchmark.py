import numpy as np
import pandas as pd
import timeit

prices = pd.Series(100 + np.cumsum(np.random.randn(30) * 2))

def pandas_calc():
    returns = prices.pct_change().dropna()
    return returns.std()

def numpy_calc():
    prices_arr = prices.to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        returns = np.diff(prices_arr) / prices_arr[:-1]
        returns = np.where(np.isinf(returns), np.nan, returns)
    return np.nanstd(returns, ddof=1)

print("Pandas calc:", pandas_calc())
print("Numpy calc:", numpy_calc())

n = 10000
print("Pandas time:", timeit.timeit(pandas_calc, number=n))
print("Numpy time:", timeit.timeit(numpy_calc, number=n))
