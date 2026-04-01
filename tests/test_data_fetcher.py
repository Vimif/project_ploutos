# ruff: noqa: E402
from unittest.mock import patch

import pandas as pd

from core.data_fetcher import UniversalDataFetcher


def test_fetch_yfinance():
    with patch("core.data_fetcher.UniversalDataFetcher._init_alpaca"), \
         patch("core.data_fetcher.UniversalDataFetcher._init_polygon"), \
         patch("core.data_fetcher.UniversalDataFetcher._fetch_yfinance") as mock_yf, \
         patch("core.data_fetcher.UniversalDataFetcher._normalize_dataframe") as mock_norm:

        mock_df = pd.DataFrame({"Close": [100.0] * 150})
        mock_yf.return_value = mock_df
        mock_norm.return_value = mock_df

        fetcher = UniversalDataFetcher()
        fetcher.sources = ['yfinance']
        df = fetcher.fetch("AAPL", "2023-01-01", "2023-01-02")
        assert df is not None
        assert "Close" in df.columns
        assert len(df) == 150

def test_fetch_bulk():
    with patch("core.data_fetcher.UniversalDataFetcher._init_alpaca"), \
         patch("core.data_fetcher.UniversalDataFetcher._init_polygon"), \
         patch("core.data_fetcher.UniversalDataFetcher.fetch") as mock_fetch:

        mock_df = pd.DataFrame({"Close": [100.0, 101.0]})
        mock_fetch.return_value = mock_df

        fetcher = UniversalDataFetcher()
        results = fetcher.bulk_fetch(["AAPL", "MSFT"], "2023-01-01", "2023-01-02")
        assert len(results) == 2
        assert "AAPL" in results

def test_normalize_dataframe():
    with patch("core.data_fetcher.UniversalDataFetcher._init_alpaca"), \
         patch("core.data_fetcher.UniversalDataFetcher._init_polygon"):
        fetcher = UniversalDataFetcher()
        df = pd.DataFrame({
            "open": [100.0, None, 101.0],
            "high": [100.0, None, 101.0],
            "low": [100.0, None, 101.0],
            "close": [100.0, None, 101.0],
            "volume": [100.0, None, 101.0],
        })
        df.index = pd.date_range("2023-01-01", periods=3)
        df.index.name = "date"
        norm = fetcher._normalize_dataframe(df)
        assert "Close" in norm.columns
