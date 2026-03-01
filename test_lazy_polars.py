import time
import numpy as np
import pandas as pd
import polars as pl
from core.features import FeatureEngineer

def test_lazy():
    N = 100_000
    dates = pd.date_range("2020-01-01", periods=N, freq="h")
    df_pd = pd.DataFrame(
        {
            "Open": 100 + np.random.randn(N).cumsum(),
            "High": 0,
            "Low": 0,
            "Close": 0,
            "Volume": np.random.randint(1000, 10000, N),
        },
        index=dates,
    )
    df_pd["Close"] = df_pd["Open"] + np.random.randn(N)
    df_pd["High"] = df_pd[["Open", "Close"]].max(axis=1) + abs(np.random.randn(N))
    df_pd["Low"] = df_pd[["Open", "Close"]].min(axis=1) - abs(np.random.randn(N))

    # Profile eager performance
    fe = FeatureEngineer()

    t0 = time.time()
    for _ in range(5):
        fe.calculate_all_features(df_pd.copy(), return_pandas=True)
    t_eager = (time.time() - t0) / 5

    print(f"Eager average time: {t_eager:.4f}s")

if __name__ == "__main__":
    test_lazy()
