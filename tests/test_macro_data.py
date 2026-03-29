import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from core.macro_data import MacroDataFetcher


@pytest.fixture
def fetcher():
    return MacroDataFetcher()


@patch("yfinance.download")
def test_fetch_all_basic(mock_download, fetcher):
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({"Close": np.random.rand(10)}, index=dates)
    mock_download.return_value = df

    result = fetcher.fetch_all(start_date="2024-01-01", end_date="2024-01-10", interval="1d")
    assert "vix" in result.columns
    assert "tnx" in result.columns
    assert "dxy" in result.columns
    assert "vix_ma20" in result.columns
    assert "tnx_ma50" in result.columns
    assert len(result) == 10


@patch("yfinance.download")
def test_fetch_all_multiindex(mock_download, fetcher):
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
    columns = pd.MultiIndex.from_tuples([("Close", "^VIX")])
    df = pd.DataFrame(np.random.rand(10, 1), index=dates, columns=columns)
    mock_download.return_value = df

    result = fetcher.fetch_all(start_date="2024-01-01", end_date="2024-01-10", interval="1d")
    assert not result.empty


@patch("yfinance.download")
def test_fetch_all_limit_730_days(mock_download, fetcher):
    mock_download.return_value = pd.DataFrame()
    start = (datetime.now() - timedelta(days=1000)).strftime("%Y-%m-%d")
    result = fetcher.fetch_all(start_date=start, interval="1h")

    call_args = mock_download.call_args[1]
    assert call_args["interval"] == "1h"
    assert call_args["start"] != start


@patch("yfinance.download")
def test_fetch_all_failure(mock_download, fetcher):
    mock_download.side_effect = Exception("API Error")
    result = fetcher.fetch_all()
    assert result.empty


@patch("yfinance.download")
def test_fetch_all_multiindex_fallback(mock_download, fetcher):
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
    columns = pd.MultiIndex.from_tuples([("NotClose", "^VIX"), ("AlsoNotClose", "^TNX")])
    df = pd.DataFrame(np.random.rand(10, 2), index=dates, columns=columns)
    mock_download.return_value = df

    result = fetcher.fetch_all(start_date="2024-01-01", end_date="2024-01-10", interval="1d")
    assert result.empty


@patch("yfinance.download")
def test_fetch_all_empty(mock_download, fetcher):
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({"NotClose": np.random.rand(10)}, index=dates)
    mock_download.return_value = df
    result = fetcher.fetch_all()
    assert result.empty
