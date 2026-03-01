import warnings
import numpy as np
import pandas as pd
import polars as pl
from typing import Optional, Union

from core.features import FeatureEngineer

class LazyFeatureEngineer(FeatureEngineer):
    def calculate_all_features(
        self, df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame], return_pandas: bool = True
    ) -> Union[pd.DataFrame, pl.DataFrame]:

        # Conversion entrée -> Polars LazyFrame
        if isinstance(df, pd.DataFrame):
            _original_index_name = df.index.name
            if isinstance(df.index, pd.DatetimeIndex):
                pdf = pl.from_pandas(df.reset_index()).lazy()
                idx_col = _original_index_name if _original_index_name else "index"
                if idx_col in pdf.columns:
                    pdf = pdf.rename({idx_col: "__date_idx"})
            else:
                pdf = pl.from_pandas(df).lazy()
        elif isinstance(df, pl.DataFrame):
            pdf = df.lazy()
        else:
            pdf = df

        pdf = self._calculate_support_resistance(pdf)
        pdf = self._calculate_mean_reversion(pdf)
        pdf = self._calculate_volume_patterns(pdf)
        pdf = self._calculate_price_action(pdf)
        pdf = self._calculate_divergences(pdf)
        pdf = self._calculate_bollinger_patterns(pdf)
        pdf = self._calculate_enhanced_momentum(pdf)
        pdf = self._calculate_trend_strength(pdf)
        pdf = self._calculate_volatility_regime(pdf)

        # entry score needs eager to check columns, let's just collect first
        pdf_eager = pdf.collect()
        pdf_eager = self._calculate_entry_score(pdf_eager)

        _date_col = None
        if "__date_idx" in pdf_eager.columns:
            _date_col = pdf_eager.select("__date_idx")
            pdf_eager = pdf_eager.drop("__date_idx")

        pdf_eager = pdf_eager.fill_nan(0).fill_null(strategy="forward").fill_null(0)

        if _date_col is not None:
            pdf_eager = pl.concat([_date_col, pdf_eager], how="horizontal")

        if return_pandas:
            res_df = pdf_eager.to_pandas()
            if "__date_idx" in res_df.columns:
                res_df = res_df.set_index("__date_idx")
                res_df.index.name = _original_index_name if 'df' in locals() and isinstance(df, pd.DataFrame) else None
            return res_df

        return pdf_eager

import time
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

    fe = LazyFeatureEngineer()

    t0 = time.time()
    for _ in range(5):
        fe.calculate_all_features(df_pd.copy(), return_pandas=True)
    t_lazy = (time.time() - t0) / 5

    print(f"Lazy average time: {t_lazy:.4f}s")

if __name__ == "__main__":
    test_lazy()
