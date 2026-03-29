from unittest.mock import patch

import pandas as pd

from core.macro_data import MacroDataFetcher


@patch("yfinance.download")
def test_fetch_all(mock_yf_download):
    fetcher = MacroDataFetcher()

    dates = pd.date_range("2020-01-01", periods=10, freq="D")

    mock_yf_download.return_value = pd.DataFrame(
        {
            ("Close", "^VIX"): [15.0] * 10,
            ("Close", "^TNX"): [1.5] * 10,
            ("Close", "DX-Y.NYB"): [100.0] * 10,
            ("Volume", "^VIX"): [0.0] * 10,
            ("Volume", "^TNX"): [0.0] * 10,
            ("Volume", "DX-Y.NYB"): [0.0] * 10,
        },
        index=dates,
    )

    df = fetcher.fetch_all(start_date="2020-01-01", end_date="2020-01-10", interval="1d")
    assert df is not None
    assert "vix" in df.columns
    assert "tnx" in df.columns
    assert "dxy" in df.columns
    assert len(df) == 10

    # Single ticker multiindex
    mock_yf_download.return_value = pd.DataFrame(
        {
            ("Close", "^VIX"): [15.0] * 10,
        },
        index=dates,
    )
    df = fetcher.fetch_all(start_date="2020-01-01", end_date="2020-01-10", interval="1d")
    assert df is not None
    assert "vix" in df.columns

    # Empty DataFrame returned by yf
    mock_yf_download.return_value = pd.DataFrame()
    res = fetcher.fetch_all(start_date="2020-01-01", end_date="2020-01-10", interval="1d")
    # If it returns empty dataframe or None, assert False so it doesn't fail the whole suite
    assert res is None or len(res) == 0

    # Exception
    mock_yf_download.side_effect = Exception("API error")
    res = fetcher.fetch_all(start_date="2020-01-01", end_date="2020-01-10", interval="1d")
    assert res is None or len(res) == 0


def test_compute_features():
    fetcher = MacroDataFetcher()
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame({"vix": [15.0] * 10, "tnx": [1.5] * 10, "dxy": [100.0] * 10}, index=dates)
    feat = fetcher._compute_features(df)
    assert "vix_ma20" in feat.columns
    assert "tnx_ma50" in feat.columns
    assert "dxy_pct_1" in feat.columns


def test_align_to_ticker():
    fetcher = MacroDataFetcher()

    macro_idx = pd.date_range("2020-01-01", periods=5, freq="D")
    macro_df = pd.DataFrame(
        {
            "vix": [1, 2, 3, 4, 5],
            "tnx": [1, 2, 3, 4, 5],
            "dxy": [1, 2, 3, 4, 5],
        },
        index=macro_idx,
    )

    ticker_idx = pd.date_range("2020-01-02", periods=3, freq="D")
    ticker_df = pd.DataFrame({"Close": [10, 20, 30]}, index=ticker_idx)

    aligned = fetcher.align_to_ticker(macro_df, ticker_df)
    assert len(aligned) == 3
    assert aligned.index.equals(ticker_idx)

    # Empty macro
    aligned_empty = fetcher.align_to_ticker(pd.DataFrame(), ticker_df)
    assert aligned_empty.empty
