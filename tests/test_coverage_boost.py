import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from core.macro_data import MacroDataFetcher
from core.data_fetcher import UniversalDataFetcher, download_data

# ============================================================================
# MacroDataFetcher Tests
# ============================================================================


def test_macro_data_fetcher_init():
    fetcher = MacroDataFetcher()
    assert isinstance(fetcher, MacroDataFetcher)


@patch("yfinance.download")
def test_fetch_macro_data(mock_download):
    # Mock yfinance response
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    mock_df = pd.DataFrame({"Close": [10.0] * 10}, index=dates)
    mock_download.return_value = mock_df

    fetcher = MacroDataFetcher()
    data = fetcher.fetch_all(start_date="2023-01-01", end_date="2023-01-10")

    assert isinstance(data, pd.DataFrame)
    assert any("vix" in col.lower() for col in data.columns)


def test_align_to_ticker():
    fetcher = MacroDataFetcher()

    # Create macro data
    macro_dates = pd.date_range("2023-01-01", periods=10, freq="D")
    macro_df = pd.DataFrame({"vix": range(10)}, index=macro_dates)

    # Create ticker data (subset of dates)
    ticker_dates = macro_dates[2:8]
    ticker_df = pd.DataFrame({"Close": range(6)}, index=ticker_dates)

    aligned = fetcher.align_to_ticker(macro_df, ticker_df)

    assert len(aligned) == len(ticker_df)
    assert aligned.index.equals(ticker_df.index)


# ============================================================================
# DataFetcher Tests
# ============================================================================


@patch("core.data_fetcher.UniversalDataFetcher.fetch")
def test_data_fetcher_download(mock_fetch):
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    mock_df = pd.DataFrame(
        {
            "Open": [100.0] * 5,
            "High": [110.0] * 5,
            "Low": [90.0] * 5,
            "Close": [105.0] * 5,
            "Volume": [1000] * 5,
        },
        index=dates,
    )
    mock_fetch.return_value = mock_df

    fetcher = UniversalDataFetcher()
    data = fetcher.fetch("AAPL", start_date="2023-01-01", end_date="2023-01-05")

    assert isinstance(data, pd.DataFrame)
    assert len(data) == 5
    assert "Close" in data.columns


def test_normalize_dataframe():
    fetcher = UniversalDataFetcher()
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    raw_df = pd.DataFrame(
        {
            "open": [100.0] * 5,
            "HIGH": [110.0] * 5,
            "low": [90.0] * 5,
            "Close": [105.0] * 5,
            "v": [1000] * 5,
        },
        index=dates,
    )

    processed = fetcher._normalize_dataframe(raw_df)
    assert isinstance(processed, pd.DataFrame)
    assert "Volume" in processed.columns  # 'v' mapped to 'Volume'
    assert "High" in processed.columns  # 'HIGH' mapped to 'High'


@patch("core.data_fetcher.UniversalDataFetcher.fetch")
def test_bulk_fetch(mock_fetch):
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    mock_df = pd.DataFrame(
        {
            "Open": [100.0] * 5,
            "High": [110.0] * 5,
            "Low": [90.0] * 5,
            "Close": [105.0] * 5,
            "Volume": [1000] * 5,
        },
        index=dates,
    )
    mock_fetch.return_value = mock_df

    fetcher = UniversalDataFetcher()
    results = fetcher.bulk_fetch(["AAPL", "MSFT"], save_to_cache=False)

    assert isinstance(results, dict)
    assert "AAPL" in results
    assert "MSFT" in results
    assert isinstance(results["AAPL"], pd.DataFrame)


@patch("core.data_fetcher.UniversalDataFetcher.fetch")
def test_download_data_wrapper(mock_fetch):
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    mock_df = pd.DataFrame(
        {
            "Open": [100.0] * 5,
            "High": [110.0] * 5,
            "Low": [90.0] * 5,
            "Close": [105.0] * 5,
            "Volume": [1000] * 5,
        },
        index=dates,
    )
    mock_fetch.return_value = mock_df

    # Test single ticker
    data_single = download_data("AAPL", period="1mo")
    assert isinstance(data_single, pd.DataFrame)

    # Test multiple tickers
    data_multi = download_data(["AAPL", "MSFT"], period="1mo")
    assert isinstance(data_multi, dict)
    assert "AAPL" in data_multi
