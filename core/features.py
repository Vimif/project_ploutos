# core/features_polars.py
"""🚀 FEATURES V9 (POLARS) - High Performance Feature Engineering

Optimisation de `core/features.py` utilisant Polars pour une vitesse x10-x100.
Mêmes calculs, même logique, mais exécutés en Rust/Arrow.

Auteur: Ploutos AI Team
Date: Feb 2026
"""

import warnings

import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")


class FeatureEngineer:
    """
    Features avancées V9 utilisant Polars pour une performance maximale.
    Drop-in replacement pour FeatureEngineer (V2).
    """

    def __init__(self):
        self.features_calculated = []

    def calculate_all_features(
        self, df: pd.DataFrame | pl.DataFrame, return_pandas: bool = True
    ) -> pd.DataFrame | pl.DataFrame:
        """
        Calcule TOUTES les features optimisées.
        Accepte Pandas ou Polars en entrée.
        """
        # Conversion entrée -> Polars
        _original_index_name = None
        if isinstance(df, pd.DataFrame):
            # Polars n'a pas d'index. On sauvegarde le nom d'index pour le restaurer.
            _original_index_name = df.index.name
            if isinstance(df.index, pd.DatetimeIndex):
                pdf = pl.from_pandas(df.reset_index())
                # Normalize index column name to '__date_idx' for reliable round-trip
                idx_col = _original_index_name if _original_index_name else "index"
                if idx_col in pdf.columns:
                    pdf = pdf.rename({idx_col: "__date_idx"})
            else:
                pdf = pl.from_pandas(df)
        else:
            pdf = df

        # --- OPTIMIZATION START: Use LazyFrame ---
        pdf = pdf.lazy()

        # 1. Support/Resistance
        pdf = self._calculate_support_resistance(pdf)

        # 2. Mean Reversion
        pdf = self._calculate_mean_reversion(pdf)

        # 3. Volume Patterns
        pdf = self._calculate_volume_patterns(pdf)

        # 4. Price Action
        pdf = self._calculate_price_action(pdf)

        # 5. Divergences
        pdf = self._calculate_divergences(pdf)

        # 6. Bollinger Patterns
        pdf = self._calculate_bollinger_patterns(pdf)

        # 7. Entry Score Composite (Optimized with sum_horizontal)
        pdf = self._calculate_entry_score(pdf)

        # 8. Momentum (amélioré)
        pdf = self._calculate_enhanced_momentum(pdf)

        # 9. Trend Strength
        pdf = self._calculate_trend_strength(pdf)

        # 10. Volatility Regime
        pdf = self._calculate_volatility_regime(pdf)

        # Protect datetime column from fill_null(0) which would corrupt it
        # Optimization: use with_columns with exclude instead of drop/concat
        # Check if __date_idx exists in schema before excluding to avoid Error
        schema_cols = pdf.collect_schema().names()
        exclude_cols = [c for c in ["__date_idx"] if c in schema_cols]

        pdf = pdf.with_columns(
            pl.all().exclude(exclude_cols).fill_nan(0).fill_null(strategy="forward").fill_null(0)
        )

        # Collect at the end
        pdf = pdf.collect()
        # --- OPTIMIZATION END ---

        # Conversion sortie
        if return_pandas:
            res_df = pdf.to_pandas()
            # Restaurer l'index date si présent (via normalized name)
            if "__date_idx" in res_df.columns:
                res_df = res_df.set_index("__date_idx")
                res_df.index.name = _original_index_name
            elif "date" in res_df.columns:
                res_df = res_df.set_index("date")
            elif "time" in res_df.columns:
                res_df = res_df.set_index("time")
            elif "index" in res_df.columns:
                res_df = res_df.set_index("index")
            return res_df

        return pdf

    def _calculate_support_resistance(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """✅ Support/Resistance DYNAMIQUES"""
        windows = [20, 50, 100]

        ops = []
        for w in windows:
            # Support = min local
            ops.append(pl.col("Low").rolling_min(window_size=w).alias(f"support_{w}"))
            # Resistance = max local
            ops.append(pl.col("High").rolling_max(window_size=w).alias(f"resistance_{w}"))

        df = df.with_columns(ops)

        # Second pass required because we reference newly created columns
        ops2 = []
        for w in windows:
            # Distance
            ops2.append(
                ((pl.col("Close") - pl.col(f"support_{w}")) / pl.col("Close")).alias(
                    f"dist_support_{w}"
                )
            )
            ops2.append(
                ((pl.col(f"resistance_{w}") - pl.col("Close")) / pl.col("Close")).alias(
                    f"dist_resistance_{w}"
                )
            )

        df = df.with_columns(ops2)

        # Third pass for binary signals
        ops3 = []
        for w in windows:
            ops3.append(
                (pl.col(f"dist_support_{w}") < 0.02).cast(pl.Int32).alias(f"near_support_{w}")
            )
            ops3.append(
                (pl.col(f"dist_resistance_{w}") < 0.02).cast(pl.Int32).alias(f"near_resistance_{w}")
            )

        return df.with_columns(ops3)

    def _calculate_mean_reversion(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """✅ MEAN REVERSION signals"""
        windows = [20, 50]
        ops = []

        for w in windows:
            ops.append(pl.col("Close").rolling_mean(window_size=w).alias(f"ma_{w}"))
            ops.append(pl.col("Close").rolling_std(window_size=w).alias(f"std_{w}"))

        df = df.with_columns(ops)

        ops2 = []
        for w in windows:
            zscore_expr = (pl.col("Close") - pl.col(f"ma_{w}")) / (pl.col(f"std_{w}") + 1e-8)
            ops2.append(zscore_expr.alias(f"zscore_{w}"))

        df = df.with_columns(ops2)

        ops3 = []
        for w in windows:
            ops3.append((pl.col(f"zscore_{w}") < -1.5).cast(pl.Int32).alias(f"oversold_{w}"))
            ops3.append((pl.col(f"zscore_{w}") > 1.5).cast(pl.Int32).alias(f"overbought_{w}"))

            reverting = (
                (pl.col(f"zscore_{w}").shift(1) < pl.col(f"zscore_{w}").shift(2))
                & (pl.col(f"zscore_{w}") < -1.0)
            ).cast(pl.Int32)
            ops3.append(reverting.alias(f"reverting_{w}"))

        return df.with_columns(ops3)

    def _calculate_volume_patterns(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """✅ VOLUME confirmation"""
        ops = [
            pl.col("Volume").rolling_mean(window_size=20).alias("vol_ma_20"),
            pl.col("Volume").rolling_mean(window_size=50).alias("vol_ma_50"),
        ]
        df = df.with_columns(ops)

        vol_ratio = pl.col("Volume") / (pl.col("vol_ma_20") + 1e-8)

        ops2 = [
            vol_ratio.alias("vol_ratio"),
            (vol_ratio > 1.5).cast(pl.Int32).alias("vol_spike"),
            (vol_ratio < 0.7).cast(pl.Int32).alias("vol_low"),
        ]
        df = df.with_columns(ops2)

        vol_bullish = (
            ((pl.col("vol_ratio") > 1.2) & (pl.col("Close") > pl.col("Close").shift(1)))
            .cast(pl.Int32)
            .alias("vol_bullish")
        )

        vol_bearish = (
            ((pl.col("vol_ratio") > 1.2) & (pl.col("Close") < pl.col("Close").shift(1)))
            .cast(pl.Int32)
            .alias("vol_bearish")
        )

        return df.with_columns([vol_bullish, vol_bearish])

    def _calculate_price_action(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """✅ PRICE ACTION patterns"""
        body = (pl.col("Close") - pl.col("Open")).abs()
        body_pct = body / pl.col("Open")

        # Upper wick: High - max(Open, Close)
        upper_wick = pl.col("High") - pl.max_horizontal(["Open", "Close"])
        # Lower wick: min(Open, Close) - Low
        lower_wick = pl.min_horizontal(["Open", "Close"]) - pl.col("Low")

        ops = [
            body.alias("body"),
            body_pct.alias("body_pct"),
            upper_wick.alias("upper_wick"),
            lower_wick.alias("lower_wick"),
        ]
        df = df.with_columns(ops)

        # Hammer
        hammer = (
            (
                (pl.col("lower_wick") > pl.col("body") * 2)
                & (pl.col("upper_wick") < pl.col("body") * 0.5)
                & (pl.col("Close") > pl.col("Open"))
            )
            .cast(pl.Int32)
            .alias("hammer")
        )

        # Shooting star
        shooting_star = (
            (
                (pl.col("upper_wick") > pl.col("body") * 2)
                & (pl.col("lower_wick") < pl.col("body") * 0.5)
                & (pl.col("Close") < pl.col("Open"))
            )
            .cast(pl.Int32)
            .alias("shooting_star")
        )

        # Doji
        doji = (pl.col("body_pct") < 0.001).cast(pl.Int32).alias("doji")

        # Bullish Engulfing
        bullish_eng = (
            (
                (pl.col("Close") > pl.col("Open"))
                & (pl.col("Close").shift(1) < pl.col("Open").shift(1))
                & (pl.col("Close") > pl.col("Open").shift(1))
                & (pl.col("Open") < pl.col("Close").shift(1))
            )
            .cast(pl.Int32)
            .alias("bullish_engulfing")
        )

        # Bearish Engulfing
        bearish_eng = (
            (
                (pl.col("Close") < pl.col("Open"))
                & (pl.col("Close").shift(1) > pl.col("Open").shift(1))
                & (pl.col("Close") < pl.col("Open").shift(1))
                & (pl.col("Open") > pl.col("Close").shift(1))
            )
            .cast(pl.Int32)
            .alias("bearish_engulfing")
        )

        return df.with_columns([hammer, shooting_star, doji, bullish_eng, bearish_eng])

    def _calculate_divergences(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """✅ DIVERGENCES RSI/Prix"""
        # RSI Implementation
        period = 14
        delta = pl.col("Close").diff()

        gain = pl.when(delta > 0).then(delta).otherwise(0)
        loss = pl.when(delta < 0).then(-delta).otherwise(0)

        avg_gain = gain.rolling_mean(window_size=period)
        avg_loss = loss.rolling_mean(window_size=period)

        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))

        df = df.with_columns(rsi.alias("rsi"))

        lookback = 10

        # Signals
        price_lower_low = pl.col("Close") < pl.col("Close").shift(lookback)
        rsi_higher_low = pl.col("rsi") > pl.col("rsi").shift(lookback)

        bull_div = (price_lower_low & rsi_higher_low).cast(pl.Int32).alias("bullish_divergence")

        price_higher_high = pl.col("Close") > pl.col("Close").shift(lookback)
        rsi_lower_high = pl.col("rsi") < pl.col("rsi").shift(lookback)

        bear_div = (price_higher_high & rsi_lower_high).cast(pl.Int32).alias("bearish_divergence")

        return df.with_columns([bull_div, bear_div])

    def _calculate_bollinger_patterns(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """✅ BOLLINGER BANDS patterns"""
        period = 20
        std_mult = 2

        mid = pl.col("Close").rolling_mean(window_size=period).alias("bb_mid")
        std = pl.col("Close").rolling_std(window_size=period).alias("bb_std")

        df = df.with_columns([mid, std])

        upper = (pl.col("bb_mid") + (pl.col("bb_std") * std_mult)).alias("bb_upper")
        lower = (pl.col("bb_mid") - (pl.col("bb_std") * std_mult)).alias("bb_lower")

        df = df.with_columns([upper, lower])

        width = ((pl.col("bb_upper") - pl.col("bb_lower")) / pl.col("bb_mid")).alias("bb_width")
        position = (
            (pl.col("Close") - pl.col("bb_lower"))
            / (pl.col("bb_upper") - pl.col("bb_lower") + 1e-8)
        ).alias("bb_position")

        touch_lower = (
            (pl.col("Close") <= pl.col("bb_lower") * 1.01).cast(pl.Int32).alias("touch_lower_bb")
        )
        touch_upper = (
            (pl.col("Close") >= pl.col("bb_upper") * 0.99).cast(pl.Int32).alias("touch_upper_bb")
        )

        df = df.with_columns([width, position, touch_lower, touch_upper])

        # Squeeze
        squeeze_thresh = pl.col("bb_width").rolling_quantile(quantile=0.2, window_size=50)
        squeeze = (pl.col("bb_width") < squeeze_thresh).cast(pl.Int32).alias("bb_squeeze")

        return df.with_columns(squeeze)

    def _calculate_entry_score(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """✅ ENTRY SCORE composite - OPTIMIZED with sum_horizontal"""
        buy_signals = [
            "near_support_20",
            "near_support_50",
            "oversold_20",
            "oversold_50",
            "reverting_20",
            "reverting_50",
            "vol_bullish",
            "hammer",
            "bullish_engulfing",
            "bullish_divergence",
            "touch_lower_bb",
        ]

        sell_signals = [
            "near_resistance_20",
            "near_resistance_50",
            "overbought_20",
            "overbought_50",
            "shooting_star",
            "bearish_engulfing",
            "bearish_divergence",
            "touch_upper_bb",
        ]

        # Use schema to check columns in LazyFrame
        schema = df.collect_schema()
        cols = schema.names()

        valid_buy_signals = [s for s in buy_signals if s in cols]
        valid_sell_signals = [s for s in sell_signals if s in cols]

        # Optimization: Use sum_horizontal for performance
        if valid_buy_signals:
            buy_score = pl.sum_horizontal([pl.col(s).fill_null(0) for s in valid_buy_signals])
        else:
            buy_score = pl.lit(0)

        if valid_sell_signals:
            sell_score = pl.sum_horizontal([pl.col(s).fill_null(0) for s in valid_sell_signals])
        else:
            sell_score = pl.lit(0)

        ops = [buy_score.alias("buy_score"), sell_score.alias("sell_score")]
        df = df.with_columns(ops)

        max_buy = len(valid_buy_signals)
        max_sell = len(valid_sell_signals)

        buy_norm = (pl.col("buy_score") / max_buy) if max_buy > 0 else pl.lit(0)
        sell_norm = (pl.col("sell_score") / max_sell) if max_sell > 0 else pl.lit(0)

        ops2 = [buy_norm.alias("buy_score_norm"), sell_norm.alias("sell_score_norm")]
        df = df.with_columns(ops2)

        entry_signal = (pl.col("buy_score_norm") - pl.col("sell_score_norm")).alias("entry_signal")

        return df.with_columns(entry_signal)

    def _calculate_enhanced_momentum(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """✅ Momentum AMÉLIORÉ"""
        periods = [5, 10, 20]
        ops = []

        for p in periods:
            # ROC: (Close / Close_shifted) - 1
            roc = ((pl.col("Close") / pl.col("Close").shift(p)) - 1).alias(f"roc_{p}")
            ops.append(roc)

        df = df.with_columns(ops)

        ops2 = []
        for p in periods:
            accel = pl.col(f"roc_{p}").diff().alias(f"momentum_accel_{p}")
            ops2.append(accel)

        df = df.with_columns(ops2)

        ops3 = []
        for p in periods:
            start = (
                ((pl.col(f"momentum_accel_{p}") > 0) & (pl.col(f"roc_{p}").abs() < 0.05))
                .cast(pl.Int32)
                .alias(f"momentum_start_{p}")
            )
            ops3.append(start)

        return df.with_columns(ops3)

    def _calculate_trend_strength(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """✅ Force du TREND (ADX)"""
        period = 14

        high = pl.col("High")
        low = pl.col("Low")
        close = pl.col("Close")
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        # Max of 3 values
        tr = pl.max_horizontal([tr1, tr2, tr3])
        atr = tr.rolling_mean(window_size=period).alias(
            "atr_temp"
        )  # Keep logical name local for now

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = pl.when((up_move > down_move) & (up_move > 0)).then(up_move).otherwise(0)
        minus_dm = pl.when((down_move > up_move) & (down_move > 0)).then(down_move).otherwise(0)

        df = df.with_columns([atr, plus_dm.alias("plus_dm_raw"), minus_dm.alias("minus_dm_raw")])

        # Smooth DM ?? Original uses rolling mean on DM directly
        plus_di = (
            100
            * pl.col("plus_dm_raw").rolling_mean(window_size=period)
            / (pl.col("atr_temp") + 1e-8)
        )
        minus_di = (
            100
            * pl.col("minus_dm_raw").rolling_mean(window_size=period)
            / (pl.col("atr_temp") + 1e-8)
        )

        df = df.with_columns([plus_di.alias("plus_di"), minus_di.alias("minus_di")])

        dx = (
            100
            * (pl.col("plus_di") - pl.col("minus_di")).abs()
            / (pl.col("plus_di") + pl.col("minus_di") + 1e-8)
        )
        adx = dx.rolling_mean(window_size=period).alias("adx")

        df = df.with_columns(adx)

        strong = (pl.col("adx") > 25).cast(pl.Int32).alias("strong_trend")
        weak = (pl.col("adx") < 20).cast(pl.Int32).alias("weak_trend")

        return df.with_columns([strong, weak]).drop(["atr_temp", "plus_dm_raw", "minus_dm_raw"])

    def _calculate_volatility_regime(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """✅ Régime de VOLATILITÉ"""
        # Re-calc ATR properly if needed, but we can reuse logic
        period = 14
        high = pl.col("High")
        low = pl.col("Low")
        close = pl.col("Close")
        prev_close = close.shift(1)

        tr = pl.max_horizontal([high - low, (high - prev_close).abs(), (low - prev_close).abs()])
        atr = tr.rolling_mean(window_size=period).alias("atr")

        df = df.with_columns(atr)

        atr_pct = (pl.col("atr") / pl.col("Close")).alias("atr_pct")
        df = df.with_columns(atr_pct)

        roll_atr = pl.col("atr_pct").rolling_quantile(quantile=0.7, window_size=50)
        roll_atr_low = pl.col("atr_pct").rolling_quantile(quantile=0.3, window_size=50)

        high_vol = (pl.col("atr_pct") > roll_atr).cast(pl.Int32).alias("high_vol")
        low_vol = (pl.col("atr_pct") < roll_atr_low).cast(pl.Int32).alias("low_vol")

        return df.with_columns([high_vol, low_vol])


if __name__ == "__main__":
    # Test Performance
    print("🧪 Test Features V9 (Polars)...")
    import time

    # Créer données test massives pour voir la diff
    N = 100_000
    dates = pd.date_range("2020-01-01", periods=N, freq="1h")
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
    # Fix High/Low/Close structure
    df_pd["Close"] = df_pd["Open"] + np.random.randn(N)
    df_pd["High"] = df_pd[["Open", "Close"]].max(axis=1) + abs(np.random.randn(N))
    df_pd["Low"] = df_pd[["Open", "Close"]].min(axis=1) - abs(np.random.randn(N))

    fe = FeatureEngineer()

    # Run Polars
    t0 = time.time()
    res_pl = fe.calculate_all_features(df_pd, return_pandas=True)
    dt_pl = time.time() - t0

    print(f"✅ Polars calc time ({N} rows): {dt_pl:.4f}s")
    print(f"Features: {len(res_pl.columns)}")
    print(res_pl[["Close", "rsi", "adx", "entry_signal"]].tail())
