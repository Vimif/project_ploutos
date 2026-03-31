import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from core.macro_data import MacroDataFetcher

@pytest.fixture
def mock_yfinance_download():
    # Helper to mock yfinance download
    with patch("yfinance.download") as mock_dl:
        yield mock_dl

@pytest.fixture
def fetcher():
    return MacroDataFetcher()

def test_fetch_all_with_multiindex(fetcher, mock_yfinance_download):
    # Create a MultiIndex dataframe simulating new yfinance output
    idx = pd.date_range("2023-01-01", periods=3)
    cols = pd.MultiIndex.from_tuples([("Close", "^VIX"), ("Open", "^VIX")])
    data = np.array([[20.0, 20.5], [21.0, 21.5], [22.0, 22.5]])
    mock_df = pd.DataFrame(data, index=idx, columns=cols)
    mock_yfinance_download.return_value = mock_df

    macro = fetcher.fetch_all(start_date="2023-01-01", end_date="2023-01-03", interval="1d")

    assert not macro.empty
    assert "vix" in macro.columns
    # Check derived feature existence
    assert "vix_ma20" in macro.columns
    assert "vix_zscore" in macro.columns

def test_fetch_all_empty(fetcher, mock_yfinance_download):
    # Simulate fetch failure or empty data
    mock_yfinance_download.return_value = pd.DataFrame()
    macro = fetcher.fetch_all(start_date="2023-01-01", end_date="2023-01-03")
    assert macro.empty

def test_compute_features_derived_logic(fetcher):
    # Ensure compute features creates specific expected fear/rate/dollar indicators
    idx = pd.date_range("2023-01-01", periods=5)
    df = pd.DataFrame({
        "vix": [10, 14, 26, 36, 12],
        "tnx": [4.0, 4.0, 4.0, 4.1, 3.9],
        "dxy": [100.0, 100.0, 100.0, 105.0, 95.0]
    }, index=idx)

    features = fetcher._compute_features(df.copy())

    assert "vix_fear" in features.columns
    assert "vix_extreme_fear" in features.columns
    assert "vix_complacent" in features.columns
    assert "tnx_rising" in features.columns
    assert "dxy_strong" in features.columns

def test_align_to_ticker(fetcher):
    # Test datetime alignment
    macro_idx = pd.date_range("2023-01-01 10:00", periods=3, freq="1h")
    macro_df = pd.DataFrame({"vix": [20, 21, 22]}, index=macro_idx)

    # Ticker has more frequent data
    ticker_idx = pd.date_range("2023-01-01 10:00", periods=5, freq="30min")
    ticker_df = pd.DataFrame({"Close": [10, 11, 12, 13, 14]}, index=ticker_idx)

    aligned = fetcher.align_to_ticker(macro_df, ticker_df)

    assert len(aligned) == len(ticker_df)
    assert aligned.index.equals(ticker_df.index)
    # Forward fill checks
    assert aligned.iloc[0]["vix"] == 20
    assert aligned.iloc[1]["vix"] == 20  # 10:30 gets 10:00
    assert aligned.iloc[2]["vix"] == 21  # 11:00 gets 11:00