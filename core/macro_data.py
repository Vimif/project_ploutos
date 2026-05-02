# core/macro_data.py
"""Free-friendly macro data fetching for VIX, TNX, and DXY.

Primary source:
- FRED daily series (free API key)

Fallback source:
- Yahoo Finance daily closes

The macro series are daily by nature, so we align them onto intraday ticker bars
with forward-fill in ``align_to_ticker``.
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from core.utils import setup_logging

warnings.filterwarnings("ignore")

load_dotenv()

logger = setup_logging(__name__)

YAHOO_MACRO_TICKERS = {
    "vix": "^VIX",
    "tnx": "^TNX",
    "dxy": "DX-Y.NYB",
}

FRED_MACRO_SERIES = {
    "vix": "VIXCLS",
    "tnx": "DGS10",
    "dxy": "DTWEXBGS",
}

FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
YAHOO_INTRADAY_LIMIT_DAYS = 729
VALID_MACRO_PROVIDERS = {"auto", "fred", "yahoo"}


class MacroDataFetcher:
    """Fetch and transform macro-economic data."""

    def __init__(
        self,
        source: Optional[str] = None,
        fred_api_key: Optional[str] = None,
    ):
        env_source = os.getenv("MACRO_DATA_PROVIDER", "auto").strip().lower()
        resolved_source = (source or env_source or "auto").strip().lower()
        self.source = resolved_source if resolved_source in VALID_MACRO_PROVIDERS else "auto"
        self.fred_api_key = fred_api_key or os.getenv("FRED_API_KEY", "").strip()

    def fetch_all(
        self,
        start_date: str = None,
        end_date: str = None,
        interval: str = "1h",
    ) -> pd.DataFrame:
        """Fetch VIX, TNX, and DXY then compute derived macro features."""

        end_dt = self._coerce_date(end_date) or datetime.now()
        start_dt = self._coerce_date(start_date) or (end_dt - timedelta(days=729))
        provider_order = self._provider_order()

        logger.info(
            "Fetch macro data (%s -> %s, %s) via %s",
            start_dt.date(),
            end_dt.date(),
            interval,
            " -> ".join(provider_order),
        )

        for provider in provider_order:
            if provider == "fred":
                raw = self._fetch_all_fred(start_dt, end_dt)
            else:
                raw = self._fetch_all_yahoo(start_dt, end_dt, interval)

            if raw:
                macro_df = pd.DataFrame(raw).sort_index()
                macro_df = macro_df.ffill().bfill()

                if hasattr(macro_df.index, "tz") and macro_df.index.tz is not None:
                    macro_df.index = macro_df.index.tz_localize(None)

                macro_df = self._compute_features(macro_df)
                macro_df = macro_df.replace([np.inf, -np.inf], np.nan)
                macro_df = macro_df.ffill().fillna(0.0)
                logger.info(
                    "Macro data ready from %s: %s bars, %s features",
                    provider,
                    len(macro_df),
                    len(macro_df.columns),
                )
                return macro_df

        logger.error("No macro data source succeeded")
        return pd.DataFrame()

    def _provider_order(self) -> list[str]:
        if self.source == "fred":
            return ["fred", "yahoo"]
        if self.source == "yahoo":
            return ["yahoo"]
        if self.fred_api_key:
            return ["fred", "yahoo"]
        return ["yahoo"]

    def _coerce_date(self, value) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        if isinstance(value, str):
            return datetime.strptime(value, "%Y-%m-%d")
        raise TypeError(f"Unsupported date type for macro fetch: {type(value).__name__}")

    def _resolve_yahoo_interval(self, interval: str, start_dt: datetime, end_dt: datetime) -> str:
        requested = str(interval).lower()
        requested_days = max((end_dt - start_dt).days, 0)
        daily_like_intervals = {"1d", "1wk", "1mo", "3mo"}
        if requested not in daily_like_intervals and requested_days > YAHOO_INTRADAY_LIMIT_DAYS:
            logger.warning(
                "Yahoo macro intraday limit hit for %s over %sd; falling back to daily bars",
                requested,
                requested_days,
            )
            return "1d"
        return requested

    def _fetch_all_fred(self, start_dt: datetime, end_dt: datetime) -> dict[str, pd.Series]:
        if not self.fred_api_key:
            logger.info("FRED API key missing; skipping FRED macro source")
            return {}

        raw: dict[str, pd.Series] = {}
        for name, series_id in FRED_MACRO_SERIES.items():
            series = self._fetch_fred_series(series_id, start_dt, end_dt)
            if series is not None and not series.empty:
                series.name = name
                raw[name] = series
                logger.info("  %s (%s): %s daily bars from FRED", name, series_id, len(series))
            else:
                logger.warning("  %s (%s): no FRED data", name, series_id)
        return raw

    def _fetch_fred_series(
        self,
        series_id: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> Optional[pd.Series]:
        try:
            response = requests.get(
                FRED_API_URL,
                params={
                    "series_id": series_id,
                    "api_key": self.fred_api_key,
                    "file_type": "json",
                    "observation_start": start_dt.strftime("%Y-%m-%d"),
                    "observation_end": end_dt.strftime("%Y-%m-%d"),
                    "sort_order": "asc",
                },
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            observations = payload.get("observations", [])
            rows = []
            for item in observations:
                value = item.get("value")
                if value in {None, ".", ""}:
                    continue
                try:
                    rows.append((pd.Timestamp(item["date"]), float(value)))
                except (KeyError, ValueError):
                    continue
            if not rows:
                return pd.Series(dtype=float)
            index = pd.DatetimeIndex([row[0] for row in rows])
            values = [row[1] for row in rows]
            return pd.Series(values, index=index, dtype=float)
        except requests.RequestException as exc:
            logger.warning("FRED fetch failed for %s: %s", series_id, exc)
            return pd.Series(dtype=float)

    def _fetch_all_yahoo(
        self,
        start_dt: datetime,
        end_dt: datetime,
        interval: str,
    ) -> dict[str, pd.Series]:
        import yfinance as yf

        macro_interval = self._resolve_yahoo_interval(interval, start_dt, end_dt)
        raw: dict[str, pd.Series] = {}
        for name, ticker in YAHOO_MACRO_TICKERS.items():
            try:
                df = yf.download(
                    ticker,
                    start=start_dt.strftime("%Y-%m-%d"),
                    end=end_dt.strftime("%Y-%m-%d"),
                    interval=macro_interval,
                    progress=False,
                    auto_adjust=True,
                )
                if df is None or df.empty:
                    logger.warning("  %s (%s): no Yahoo data", name, ticker)
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    try:
                        series = df.xs("Close", axis=1, level=0)
                        if isinstance(series, pd.DataFrame):
                            series = series.iloc[:, 0]
                    except KeyError:
                        series = df["Close"]
                else:
                    series = df["Close"]
                if len(series) > 0:
                    series.name = name
                    raw[name] = series
                    logger.info(
                        "  %s (%s): %s bars from Yahoo at %s",
                        name,
                        ticker,
                        len(series),
                        macro_interval,
                    )
            except Exception as exc:
                logger.warning("  %s (%s): Yahoo fetch failed: %s", name, ticker, exc)
        return raw

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute derived macro features."""

        for col in ["vix", "tnx", "dxy"]:
            if col not in df.columns:
                continue

            series = df[col]
            df[f"{col}_ma20"] = series.rolling(20, min_periods=1).mean()
            df[f"{col}_ma50"] = series.rolling(50, min_periods=1).mean()
            df[f"{col}_pct_1"] = series.pct_change(1)
            df[f"{col}_pct_5"] = series.pct_change(5)

            ma = df[f"{col}_ma20"]
            std = series.rolling(20, min_periods=1).std()
            df[f"{col}_zscore"] = (series - ma) / (std + 1e-8)

        if "vix" in df.columns:
            df["vix_fear"] = (df["vix"] > 25).astype(np.float32)
            df["vix_extreme_fear"] = (df["vix"] > 35).astype(np.float32)
            df["vix_complacent"] = (df["vix"] < 15).astype(np.float32)

        if "tnx" in df.columns:
            df["tnx_rising"] = (df["tnx_pct_5"] > 0.02).astype(np.float32)
            df["tnx_falling"] = (df["tnx_pct_5"] < -0.02).astype(np.float32)

        if "dxy" in df.columns:
            df["dxy_strong"] = (df["dxy_zscore"] > 1.0).astype(np.float32)
            df["dxy_weak"] = (df["dxy_zscore"] < -1.0).astype(np.float32)

        return df

    def align_to_ticker(self, macro_df: pd.DataFrame, ticker_df: pd.DataFrame) -> pd.DataFrame:
        """Align macro data to the ticker index using forward-fill."""

        if macro_df.empty:
            return pd.DataFrame(index=ticker_df.index)

        ticker_tz = getattr(ticker_df.index, "tz", None)
        macro_tz = getattr(macro_df.index, "tz", None)
        macro_df = macro_df.copy()

        if ticker_tz is not None and macro_tz is None:
            macro_df.index = macro_df.index.tz_localize("UTC")
        elif ticker_tz is None and macro_tz is not None:
            macro_df.index = macro_df.index.tz_localize(None)

        aligned = macro_df.reindex(ticker_df.index, method="ffill")
        aligned = aligned.bfill().fillna(0.0)
        return aligned
