from datetime import datetime

import pandas as pd

from core.macro_data import MacroDataFetcher


def _sample_macro_raw():
    index = pd.date_range("2025-01-01", periods=5, freq="D")
    return {
        "vix": pd.Series([18.0, 19.0, 17.5, 20.0, 21.0], index=index),
        "tnx": pd.Series([4.1, 4.0, 3.9, 4.05, 4.2], index=index),
        "dxy": pd.Series([104.0, 103.8, 104.2, 104.5, 104.7], index=index),
    }


def test_resolve_yahoo_interval_falls_back_to_daily_for_long_intraday():
    fetcher = MacroDataFetcher(source="yahoo")

    interval = fetcher._resolve_yahoo_interval(
        "1h",
        datetime(2020, 1, 1),
        datetime(2026, 1, 1),
    )

    assert interval == "1d"


def test_fetch_all_prefers_fred_when_key_available(monkeypatch):
    calls = {"fred": 0, "yahoo": 0}

    def fake_fred(self, start_dt, end_dt):
        del self, start_dt, end_dt
        calls["fred"] += 1
        return _sample_macro_raw()

    def fake_yahoo(self, start_dt, end_dt, interval):
        del self, start_dt, end_dt, interval
        calls["yahoo"] += 1
        return {}

    monkeypatch.setattr(MacroDataFetcher, "_fetch_all_fred", fake_fred)
    monkeypatch.setattr(MacroDataFetcher, "_fetch_all_yahoo", fake_yahoo)

    fetcher = MacroDataFetcher(source="auto", fred_api_key="free-key")
    macro = fetcher.fetch_all("2025-01-01", "2025-01-10", interval="1h")

    assert calls["fred"] == 1
    assert calls["yahoo"] == 0
    assert not macro.empty
    assert {"vix", "tnx", "dxy", "vix_ma20", "tnx_zscore", "dxy_strong"}.issubset(macro.columns)


def test_fetch_all_falls_back_to_yahoo_when_fred_unavailable(monkeypatch):
    calls = {"fred": 0, "yahoo": 0}

    def fake_yahoo(self, start_dt, end_dt, interval):
        del self, start_dt, end_dt, interval
        calls["yahoo"] += 1
        return _sample_macro_raw()

    monkeypatch.setattr(MacroDataFetcher, "_fetch_all_yahoo", fake_yahoo)

    fetcher = MacroDataFetcher(source="auto", fred_api_key="")
    macro = fetcher.fetch_all("2025-01-01", "2025-01-10", interval="1h")

    assert calls["fred"] == 0
    assert calls["yahoo"] == 1
    assert not macro.empty
    assert "vix_fear" in macro.columns


def test_align_to_ticker_forward_fills_daily_macro_onto_hourly_bars():
    fetcher = MacroDataFetcher(source="fred", fred_api_key="free-key")
    macro = pd.DataFrame({"vix": [18.0, 19.0]}, index=pd.to_datetime(["2025-01-01", "2025-01-02"]))
    ticker = pd.DataFrame(
        {"Close": [100, 101, 102, 103]},
        index=pd.date_range("2025-01-01 10:00:00", periods=4, freq="12h"),
    )

    aligned = fetcher.align_to_ticker(macro, ticker)

    assert list(aligned.index) == list(ticker.index)
    assert aligned.loc[ticker.index[0], "vix"] == 18.0
    assert aligned.loc[ticker.index[-1], "vix"] == 19.0
