from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from core.macro_data import MacroDataFetcher


# Mock setup_logging to avoid side effects
@pytest.fixture(autouse=True)
def mock_logger():
    with patch("core.macro_data.setup_logging"):
        yield


@pytest.fixture
def mock_yfinance():
    with patch("yfinance.download") as mock_download:
        yield mock_download


def test_fetch_macro_data_success(mock_yfinance):
    """Test fetching macro data successfully."""
    # Setup mock data structure resembling yfinance output
    dates = pd.date_range("2023-01-01", periods=10, freq="D")

    # Simple DataFrame for single ticker download (auto_adjust=True returns 'Close' usually)
    # But code handles MultiIndex. Let's provide a simple DF with 'Close' column
    mock_df = pd.DataFrame({"Close": np.random.uniform(10, 20, 10)}, index=dates)

    mock_yfinance.return_value = mock_df

    fetcher = MacroDataFetcher()
    # Mocking fetch_all internals slightly by relying on yfinance mock
    # The code imports yfinance inside fetch_all, so we need to patch it where it is imported
    # or rely on sys.modules patching if we can't control the import scope easily.
    # Actually, patch('yfinance.download') should work if yfinance is imported at module level or
    # if we patch 'core.macro_data.yf' but the file does `import yfinance as yf` inside the method.
    # The file does `import yfinance as yf` INSIDE `fetch_all`.
    # So we must patch `yfinance` in `sys.modules` before calling.

    with patch.dict("sys.modules", {"yfinance": MagicMock()}):
        import sys

        mock_yf_module = sys.modules["yfinance"]
        mock_yf_module.download.return_value = mock_df

        df = fetcher.fetch_all(start_date="2023-01-01", end_date="2023-01-10")

        assert isinstance(df, pd.DataFrame)
        # Should have columns for vix, tnx, dxy (if successful)
        # Since we return same mock_df for all, all should be populated
        assert "vix" in df.columns
        assert "tnx" in df.columns
        assert "dxy" in df.columns
        # And derived features
        assert "vix_ma20" in df.columns


def test_align_to_ticker():
    """Test aligning macro data to ticker data."""
    fetcher = MacroDataFetcher()

    # Ticker index (hourly)
    ticker_dates = pd.date_range("2023-01-01 09:30", "2023-01-05 16:00", freq="h")
    ticker_df = pd.DataFrame({"Close": 100}, index=ticker_dates)

    # Macro data (daily or irregular)
    macro_dates = pd.date_range("2023-01-01", "2023-01-06", freq="D")
    macro_df = pd.DataFrame({"vix": [20, 21, 20, 19, 18, 17]}, index=macro_dates)

    aligned = fetcher.align_to_ticker(macro_df, ticker_df)

    assert len(aligned) == len(ticker_df)
    assert aligned.index.equals(ticker_df.index)
    assert not aligned["vix"].isnull().any()
    # Check values propagated
    assert aligned.iloc[0]["vix"] == 20.0


def test_compute_features():
    """Test feature calculation."""
    fetcher = MacroDataFetcher()

    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "vix": np.random.uniform(10, 30, 100),
            "tnx": np.random.uniform(3, 5, 100),
            "dxy": np.random.uniform(90, 110, 100),
        },
        index=dates,
    )

    processed = fetcher._compute_features(df)

    assert "vix_ma20" in processed.columns
    assert "vix_fear" in processed.columns
    assert "tnx_rising" in processed.columns
    assert "dxy_strong" in processed.columns
