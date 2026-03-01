import time
import numpy as np
import pandas as pd
from core.features import FeatureEngineer

def run_benchmark():
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

    fe = FeatureEngineer()

    # warmup
    _ = fe.calculate_all_features(df_pd.iloc[:1000].copy(), return_pandas=True)

    times = []
    for _ in range(5):
        t0 = time.time()
        _ = fe.calculate_all_features(df_pd.copy(), return_pandas=True)
        times.append(time.time() - t0)

    print(f"Average time: {np.mean(times):.4f}s")

if __name__ == "__main__":
    run_benchmark()
