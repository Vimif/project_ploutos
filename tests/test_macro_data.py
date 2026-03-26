from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from core.macro_data import MacroDataFetcher


@pytest.fixture
def macro_fetcher():
    return MacroDataFetcher()

def create_mock_yf_download(columns):
    dates = pd.date_range("2023-01-01", periods=100, freq="1h")
    df = pd.DataFrame(np.random.randn(100, len(columns)), index=dates, columns=columns)
    return df

@patch("yfinance.download")
def test_fetch_all_with_valid_data(mock_download, macro_fetcher):
    dates = pd.date_range("2023-01-01", periods=100, freq="1h")
    df_vix = pd.DataFrame({"Close": np.random.randn(100) * 5 + 20}, index=dates)
    df_tnx = pd.DataFrame({"Close": np.random.randn(100) * 0.5 + 4}, index=dates)
    df_dxy = pd.DataFrame({"Close": np.random.randn(100) * 2 + 100}, index=dates)

    mock_download.side_effect = [df_vix, df_tnx, df_dxy]

    result_df = macro_fetcher.fetch_all("2023-01-01", "2023-01-05", interval="1h")

    assert mock_download.call_count == 3
    assert not result_df.empty
    assert "vix" in result_df.columns
    assert "tnx" in result_df.columns
    assert "dxy" in result_df.columns
    assert "vix_ma20" in result_df.columns
    assert "tnx_pct_5" in result_df.columns
    assert "dxy_zscore" in result_df.columns

@patch("yfinance.download")
def test_fetch_all_empty_data(mock_download, macro_fetcher):
    mock_download.return_value = pd.DataFrame()

    result_df = macro_fetcher.fetch_all()
    assert result_df.empty

def test_compute_features(macro_fetcher):
    dates = pd.date_range("2023-01-01", periods=100, freq="1d")
    df = pd.DataFrame({
        "vix": np.random.randn(100) * 5 + 20,
        "tnx": np.random.randn(100) * 0.5 + 4,
        "dxy": np.random.randn(100) * 2 + 100
    }, index=dates)

    result = macro_fetcher._compute_features(df)

    assert "vix_ma20" in result.columns
    assert "vix_ma50" in result.columns
    assert "vix_fear" in result.columns
    assert "tnx_pct_5" in result.columns
    assert "dxy_strong" in result.columns
    assert "vix_complacent" in result.columns

def test_align_to_ticker(macro_fetcher):
    ticker_dates = pd.date_range("2023-01-01", periods=100, freq="1d", tz="UTC")
    ticker_df = pd.DataFrame({"Close": np.random.randn(100)}, index=ticker_dates)

    macro_dates = pd.date_range("2023-01-01", periods=100, freq="1d", tz="UTC")
    macro_df = pd.DataFrame({"vix": np.random.randn(100)}, index=macro_dates)

    aligned_df = macro_fetcher.align_to_ticker(macro_df, ticker_df)

    assert aligned_df.index.equals(ticker_df.index)
    assert "vix" in aligned_df.columns

def test_align_to_ticker_empty(macro_fetcher):
    ticker_dates = pd.date_range("2023-01-01", periods=100, freq="1d", tz="UTC")
    ticker_df = pd.DataFrame({"Close": np.random.randn(100)}, index=ticker_dates)

    macro_df = pd.DataFrame()

    aligned_df = macro_fetcher.align_to_ticker(macro_df, ticker_df)
    assert aligned_df.empty
    assert aligned_df.index.equals(ticker_df.index)
