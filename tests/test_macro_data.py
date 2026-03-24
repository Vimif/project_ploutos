import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from core.macro_data import MacroDataFetcher

@pytest.fixture
def mock_yf_download():
    # Helper to create realistic-looking yfinance mock data
    dates = pd.date_range(start="2023-01-01", periods=10, freq="1h")
    df = pd.DataFrame({
        "Close": np.random.uniform(10, 30, 10)
    }, index=dates)
    return df

@patch('yfinance.download')
def test_fetch_all_with_defaults(mock_download, mock_yf_download):
    # Setup mock to return the same dataframe for all tickers
    mock_download.side_effect = [mock_yf_download.copy(), mock_yf_download.copy(), mock_yf_download.copy()]

    fetcher = MacroDataFetcher()
    with patch('core.macro_data.datetime') as mock_dt:
        mock_dt.now.return_value = datetime(2023, 10, 1)
        mock_dt.strptime = datetime.strptime
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

        result = fetcher.fetch_all(start_date="2023-01-01", end_date="2023-01-10", interval="1h")

        assert isinstance(result, pd.DataFrame)
        assert not result.empty

        expected_cols = ['vix', 'tnx', 'dxy', 'vix_ma20']
        for col in expected_cols:
            assert col in result.columns

@patch('yfinance.download')
def test_fetch_all_no_data(mock_download):
    mock_download.return_value = pd.DataFrame()
    fetcher = MacroDataFetcher()
    result = fetcher.fetch_all(start_date="2023-01-01", end_date="2023-01-10")
    assert result.empty

@patch('yfinance.download')
def test_fetch_all_partial_data(mock_download, mock_yf_download):
    mock_download.side_effect = [mock_yf_download.copy(), pd.DataFrame(), pd.DataFrame()]
    fetcher = MacroDataFetcher()
    result = fetcher.fetch_all(start_date="2023-01-01", end_date="2023-01-10")

    assert not result.empty
    assert 'vix' in result.columns
    assert 'vix_ma20' in result.columns

def test_align_to_ticker():
    fetcher = MacroDataFetcher()

    dates = pd.date_range(start="2023-01-01", periods=5, freq="1h")
    df = pd.DataFrame({"close": [100, 101, 102, 103, 104]}, index=dates)

    macro_df = pd.DataFrame({
        "vix": [15.0, 16.0, 15.0, 14.0, 15.0],
        "vix_ma20": [15.5, 15.5, 15.5, 15.5, 15.5]
    }, index=dates)

    result = fetcher.align_to_ticker(macro_df, df)

    assert 'vix' in result.columns
    assert len(result) == 5

def test_align_to_ticker_empty():
    fetcher = MacroDataFetcher()
    dates = pd.date_range(start="2023-01-01", periods=5, freq="1h")
    df = pd.DataFrame({"close": [100, 101, 102, 103, 104]}, index=dates)

    result = fetcher.align_to_ticker(pd.DataFrame(), df)
    assert result.empty or len(result) == 5
    assert 'vix' not in result.columns

def test_align_to_ticker_tz_aware():
    fetcher = MacroDataFetcher()

    dates_naive = pd.date_range(start="2023-01-01", periods=5, freq="1h")
    dates_aware = dates_naive.tz_localize("UTC")

    ticker_df = pd.DataFrame({"close": [100]*5}, index=dates_aware)
    macro_df = pd.DataFrame({"vix": [15.0]*5}, index=dates_naive)

    # Ticker aware, Macro naive
    res1 = fetcher.align_to_ticker(macro_df, ticker_df)
    assert res1.index.tz is not None

    # Ticker naive, Macro aware
    ticker_df2 = pd.DataFrame({"close": [100]*5}, index=dates_naive)
    macro_df2 = pd.DataFrame({"vix": [15.0]*5}, index=dates_aware)

    res2 = fetcher.align_to_ticker(macro_df2, ticker_df2)
    assert res2.index.tz is None
