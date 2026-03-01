import time
import numpy as np
import pandas as pd
import polars as pl

# Generate data
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

from core.features import FeatureEngineer

class LazyFeatureEngineer(FeatureEngineer):
    def calculate_all_features(self, df, return_pandas=True):
        _original_index_name = None

        # Conversion entrée -> Polars LazyFrame
        if isinstance(df, pd.DataFrame):
            _original_index_name = df.index.name
            if isinstance(df.index, pd.DatetimeIndex):
                pdf = pl.from_pandas(df.reset_index()).lazy()
                idx_col = _original_index_name if _original_index_name else "index"
                if idx_col in pdf.collect_schema().names():
                    pdf = pdf.rename({idx_col: "__date_idx"})
            else:
                pdf = pl.from_pandas(df).lazy()
        elif isinstance(df, pl.DataFrame):
            pdf = df.lazy()
        else:
            # Assume it's already a LazyFrame or compatible
            if hasattr(df, "lazy"):
                pdf = df.lazy()
            else:
                pdf = df

        # Calculs chaînés (Lazy execution)
        pdf = self._calculate_support_resistance(pdf)
        pdf = self._calculate_mean_reversion(pdf)
        pdf = self._calculate_volume_patterns(pdf)
        pdf = self._calculate_price_action(pdf)
        pdf = self._calculate_divergences(pdf)
        pdf = self._calculate_bollinger_patterns(pdf)
        pdf = self._calculate_enhanced_momentum(pdf)
        pdf = self._calculate_trend_strength(pdf)
        pdf = self._calculate_volatility_regime(pdf)

        # Collect pour valider les colonnes et executer le lazy graph
        # FeatureEngineer expects eager DataFrame for _calculate_entry_score because
        # it checks if columns exist (`if s in cols`)
        pdf = pdf.collect()
        pdf = self._calculate_entry_score(pdf)

        # Protect datetime column from fill_null(0) which would corrupt it
        # More efficient than splitting columns: use pl.exclude
        if "__date_idx" in pdf.columns:
            pdf = pdf.with_columns(
                pl.all().exclude("__date_idx").fill_nan(0).fill_null(strategy="forward").fill_null(0)
            )
        else:
            pdf = pdf.with_columns(
                pl.all().fill_nan(0).fill_null(strategy="forward").fill_null(0)
            )

        # Conversion sortie
        if return_pandas:
            res_df = pdf.to_pandas()
            # Restaurer l'index date si présent (via normalized name)
            if "__date_idx" in res_df.columns:
                res_df = res_df.set_index("__date_idx")
                if 'df' in locals() and isinstance(df, pd.DataFrame):
                    res_df.index.name = _original_index_name
            elif "date" in res_df.columns:
                res_df = res_df.set_index("date")
            elif "time" in res_df.columns:
                res_df = res_df.set_index("time")
            elif "index" in res_df.columns:
                res_df = res_df.set_index("index")
            return res_df

        return pdf

if __name__ == "__main__":
    fe = FeatureEngineer()
    lfe = LazyFeatureEngineer()

    # Warmup
    fe.calculate_all_features(df_pd.copy(), return_pandas=True)
    lfe.calculate_all_features(df_pd.copy(), return_pandas=True)

    # Check output match
    res_fe = fe.calculate_all_features(df_pd.copy(), return_pandas=True)
    res_lfe = lfe.calculate_all_features(df_pd.copy(), return_pandas=True)

    # Order of columns might differ, align them for comparison
    res_lfe = res_lfe[res_fe.columns]

    np.testing.assert_allclose(res_fe.values, res_lfe.values, atol=1e-5, rtol=1e-5)
    print("Results match exactly!")

    t0 = time.time()
    for _ in range(10):
        fe.calculate_all_features(df_pd.copy(), return_pandas=True)
    t_eager = (time.time() - t0)/10
    print(f"Eager Time: {t_eager:.4f}s")

    t0 = time.time()
    for _ in range(10):
        lfe.calculate_all_features(df_pd.copy(), return_pandas=True)
    t_lazy = (time.time() - t0)/10
    print(f"Lazy Time:  {t_lazy:.4f}s")
    print(f"Speedup:    {((t_eager - t_lazy)/t_eager)*100:.1f}%")
