#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""📊 Chart Tools - Fibonacci, Volume Profile, Support/Resistance"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return None
        return x
    except Exception:
        return None


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les colonnes OHLCV de yfinance"""
    if df is None or df.empty:
        return df

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)

    required = {"High", "Low", "Close", "Volume"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Colonnes OHLCV manquantes: {sorted(missing)}")

    return out


@dataclass
class SRLevel:
    price: float
    strength: int


class ChartTools:
    """Outils chartistes avancés"""

    def calculate_fibonacci(self, df: pd.DataFrame, lookback: int = 90) -> Dict[str, Any]:
        """Calcule niveaux Fibonacci (retracements + extensions)"""
        df = _ensure_ohlcv(df)
        lookback = max(20, int(lookback))
        data = df.tail(lookback).copy()
        if data.empty:
            return {"error": "no_data"}

        swing_high = float(data["High"].max())
        swing_low = float(data["Low"].min())
        last_close = float(data["Close"].iloc[-1])

        if swing_high <= swing_low:
            return {"error": "invalid_range", "swing_high": swing_high, "swing_low": swing_low}

        # Trend: close dans moitié haute = up, sinon down
        mid = swing_low + 0.5 * (swing_high - swing_low)
        trend = "up" if last_close >= mid else "down"

        retracements = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        extensions = [1.618, 2.618]
        levels: Dict[str, float] = {}
        rng = swing_high - swing_low

        if trend == "up":
            for r in retracements:
                price = swing_high - r * rng
                levels[f"{r*100:.1f}"] = float(price)
            for e in extensions:
                price = swing_high + (e - 1.0) * rng
                levels[f"{e*100:.1f}"] = float(price)
        else:
            for r in retracements:
                price = swing_low + r * rng
                levels[f"{r*100:.1f}"] = float(price)
            for e in extensions:
                price = swing_low - (e - 1.0) * rng
                levels[f"{e*100:.1f}"] = float(price)

        return {
            "trend": trend,
            "lookback": lookback,
            "swing_high": swing_high,
            "swing_low": swing_low,
            "current_price": last_close,
            "levels": levels,
        }

    def calculate_volume_profile(self, df: pd.DataFrame, bins: int = 24) -> Dict[str, Any]:
        """Volume Profile avec POC et Value Area"""
        df = _ensure_ohlcv(df)
        bins = max(10, min(int(bins), 200))
        data = df.dropna(subset=["High", "Low", "Close", "Volume"]).copy()
        if data.empty:
            return {"error": "no_data"}

        typical = (data["High"] + data["Low"] + data["Close"]) / 3.0
        volumes = data["Volume"].astype(float).values

        p_min = float(data["Low"].min())
        p_max = float(data["High"].max())
        if p_max <= p_min:
            return {"error": "invalid_range", "min": p_min, "max": p_max}

        edges = np.linspace(p_min, p_max, bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2.0
        idx = np.digitize(typical.values, edges) - 1
        idx = np.clip(idx, 0, bins - 1)

        vol_by_bin = np.zeros(bins, dtype=float)
        for i, v in zip(idx, volumes):
            vol_by_bin[int(i)] += float(v)

        total_vol = float(vol_by_bin.sum())
        if total_vol <= 0:
            return {"error": "no_volume"}

        poc_i = int(np.argmax(vol_by_bin))
        poc_price = float(centers[poc_i])

        # Value Area 70%
        target = 0.70 * total_vol
        selected = {poc_i}
        current = float(vol_by_bin[poc_i])
        left = poc_i - 1
        right = poc_i + 1

        while current < target and (left >= 0 or right < bins):
            left_vol = vol_by_bin[left] if left >= 0 else -1
            right_vol = vol_by_bin[right] if right < bins else -1

            if right_vol > left_vol:
                selected.add(right)
                current += float(right_vol)
                right += 1
            else:
                selected.add(left)
                current += float(left_vol)
                left -= 1

        sel = sorted(selected)
        va_low = float(edges[sel[0]])
        va_high = float(edges[sel[-1] + 1])

        profile = []
        for c, v in zip(centers, vol_by_bin):
            profile.append({"price": float(c), "volume": float(v), "pct": float(v / total_vol)})

        return {
            "bins": bins,
            "min_price": p_min,
            "max_price": p_max,
            "total_volume": total_vol,
            "poc_price": poc_price,
            "value_area_low": va_low,
            "value_area_high": va_high,
            "profile": profile,
        }

    def detect_support_resistance(
        self,
        df: pd.DataFrame,
        window: int = 20,
        max_levels: int = 8,
        tolerance_pct: float = 0.006,
    ) -> Dict[str, Any]:
        """Détection Support/Resistance avec clustering"""
        df = _ensure_ohlcv(df)
        window = max(5, int(window))
        max_levels = max(1, int(max_levels))
        tolerance_pct = float(tolerance_pct)

        data = df.dropna(subset=["High", "Low", "Close"]).copy()
        if len(data) < window * 2:
            return {"supports": [], "resistances": [], "note": "not_enough_data"}

        lows = data["Low"]
        highs = data["High"]

        roll_min = lows.rolling(window, center=True).min()
        roll_max = highs.rolling(window, center=True).max()

        pivot_lows = data[(lows == roll_min)].copy()
        pivot_highs = data[(highs == roll_max)].copy()

        def cluster_levels(prices: List[float]) -> List[SRLevel]:
            prices = [float(p) for p in prices if _safe_float(p) is not None]
            prices.sort()
            clusters: List[List[float]] = []
            for p in prices:
                if not clusters:
                    clusters.append([p])
                    continue
                ref = float(np.mean(clusters[-1]))
                if abs(p - ref) / max(ref, 1e-9) <= tolerance_pct:
                    clusters[-1].append(p)
                else:
                    clusters.append([p])

            levels = [SRLevel(price=float(np.mean(c)), strength=len(c)) for c in clusters]
            levels.sort(key=lambda x: x.strength, reverse=True)
            return levels[:max_levels]

        supports = cluster_levels(pivot_lows["Low"].tolist())
        resistances = cluster_levels(pivot_highs["High"].tolist())

        return {
            "window": window,
            "tolerance_pct": tolerance_pct,
            "supports": [{"price": s.price, "strength": s.strength} for s in supports],
            "resistances": [{"price": r.price, "strength": r.strength} for r in resistances],
        }
