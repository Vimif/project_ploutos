import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from core.sp500_scanner import SP500Scanner


@pytest.fixture
def scanner():
    with patch("pathlib.Path.mkdir"):
        return SP500Scanner(cache_dir="dummy_dir", lookback_days=100)


@patch("requests.get")
@patch("pandas.read_html")
def test_fetch_sp500_list(mock_read_html, mock_get, scanner):
    mock_response = MagicMock()
    mock_response.text = "<html><table></table></html>"
    mock_get.return_value = mock_response

    dummy_df = pd.DataFrame(
        {
            "Symbol": ["AAPL", "BRK.B"],
            "Security": ["Apple Inc.", "Berkshire"],
            "GICS Sector": ["Information Technology", "Financials"],
            "GICS Sub-Industry": ["Tech", "Finance"],
        }
    )
    mock_read_html.return_value = [dummy_df]

    df = scanner.fetch_sp500_list()
    assert len(df) == 2
    assert df.iloc[1]["Symbol"] == "BRK-B"
    mock_get.assert_called_once()
    mock_read_html.assert_called_once()


@patch("core.sp500_scanner.UniversalDataFetcher")
def test_calculate_sharpe_valid(mock_fetcher_class, scanner):
    mock_fetcher = MagicMock()
    mock_fetcher_class.return_value = mock_fetcher
    scanner.fetcher = mock_fetcher

    df = pd.DataFrame({"Close": [1.0] * 101})
    df["Close"] = df["Close"] * pd.Series([1.01] * 101).cumprod()
    mock_fetcher.fetch.return_value = df

    sharpe = scanner._calculate_sharpe("AAPL")
    assert not np.isnan(sharpe)


@patch("core.sp500_scanner.UniversalDataFetcher")
def test_calculate_sharpe_invalid_short(mock_fetcher_class, scanner):
    mock_fetcher = MagicMock()
    mock_fetcher_class.return_value = mock_fetcher
    scanner.fetcher = mock_fetcher

    df = pd.DataFrame({"Close": [1.0, 1.1]})
    mock_fetcher.fetch.return_value = df

    sharpe = scanner._calculate_sharpe("AAPL")
    assert np.isnan(sharpe)


@patch("core.sp500_scanner.SP500Scanner.fetch_sp500_list")
@patch("core.sp500_scanner.SP500Scanner._calculate_sharpe")
def test_scan_sectors(mock_sharpe, mock_fetch, scanner):
    df = pd.DataFrame(
        {
            "Symbol": ["AAPL", "MSFT", "JNJ", "PFE"],
            "GICS Sector": [
                "Information Technology",
                "Information Technology",
                "Health Care",
                "Health Care",
            ],
        }
    )
    mock_fetch.return_value = df

    def mock_calc(ticker):
        sharpes = {"AAPL": 1.5, "MSFT": 2.0, "JNJ": 0.5, "PFE": 1.0}
        return sharpes.get(ticker, np.nan)

    mock_sharpe.side_effect = mock_calc

    results = scanner.scan_sectors(stocks_per_sector=1, max_workers=2)
    assert results["total_stocks"] == 2
    assert results["sectors"]["Information Technology"] == ["MSFT"]
    assert results["sectors"]["Health Care"] == ["PFE"]


@patch("pathlib.Path.mkdir")
@patch("builtins.open", new_callable=MagicMock)
def test_save_results(mock_open, mock_mkdir, scanner):
    scanner.save_results({"test": "data"})
    mock_open.assert_called_once()


@patch("pathlib.Path.exists")
def test_load_cached_results_no_file(mock_exists, scanner):
    mock_exists.return_value = False
    assert scanner.load_cached_results() is None


@patch("pathlib.Path.exists")
@patch("pathlib.Path.stat")
@patch("builtins.open", new_callable=MagicMock)
@patch("json.load")
def test_load_cached_results_valid(mock_json, mock_open, mock_stat, mock_exists, scanner):
    from datetime import datetime, timedelta

    mock_exists.return_value = True

    mock_stat_result = MagicMock()
    mock_stat_result.st_mtime = (datetime.now() - timedelta(days=1)).timestamp()
    mock_stat.return_value = mock_stat_result

    mock_json.return_value = {"total_stocks": 10, "sectors": {}}

    data = scanner.load_cached_results(max_age_days=30)
    assert data is not None
    assert data["total_stocks"] == 10


@patch("requests.get")
@patch("pandas.read_html")
def test_fetch_sp500_list_failure(mock_read_html, mock_get):
    scanner = SP500Scanner(cache_dir="dummy_dir", lookback_days=100)
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("API Error")
    mock_get.return_value = mock_response

    with pytest.raises(Exception):
        scanner.fetch_sp500_list()
