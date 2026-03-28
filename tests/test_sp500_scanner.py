import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from core.sp500_scanner import SP500Scanner
import pathlib


@pytest.fixture
def scanner():
    return SP500Scanner(cache_dir="/tmp/test_cache")


def test_fetch_sp500_list(scanner):
    dummy_df = pd.DataFrame(
        {
            "Symbol": ["AAPL", "MSFT"],
            "Security": ["Apple", "Microsoft"],
            "GICS Sector": ["Information Technology", "Information Technology"],
            "GICS Sub-Industry": ["Tech", "Tech"],
        }
    )
    mock_response = MagicMock()
    mock_response.text = "<table></table>"
    mock_response.raise_for_status = MagicMock()

    with (
        patch("requests.get", return_value=mock_response),
        patch("pathlib.Path.exists", return_value=False),
        patch("pathlib.Path.mkdir"),
        patch("pandas.DataFrame.to_csv"),
        patch("pandas.read_html", return_value=[dummy_df]),
    ):
        df = scanner.fetch_sp500_list()
        assert not df.empty
        assert "Symbol" in df.columns
        assert "AAPL" in df["Symbol"].values


def test_fetch_sp500_list_cached(scanner):
    dummy_df = pd.DataFrame(
        {
            "Symbol": ["GOOG", "AMZN"],
            "Security": ["Google", "Amazon"],
            "GICS Sector": ["Communication Services", "Consumer Discretionary"],
            "GICS Sub-Industry": ["Comm", "Retail"],
        }
    )
    mock_response = MagicMock()
    mock_response.text = "<table></table>"
    mock_response.raise_for_status = MagicMock()

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.stat") as mock_stat,
        patch("time.time", return_value=1000000),
        patch("pandas.read_csv", return_value=dummy_df),
        patch("pandas.read_html", return_value=[dummy_df]),
        patch("requests.get", return_value=mock_response),
    ):

        # Mock stat so the cache is fresh (less than 24h old)
        mock_stat.return_value.st_mtime = 1000000 - 3600

        df = scanner.fetch_sp500_list()
        assert not df.empty
        assert "GOOG" in df["Symbol"].values


def test_calculate_sharpe(scanner):
    dummy_data = pd.DataFrame({"close": [100, 101, 102, 101, 105]})
    mock_fetcher = MagicMock()
    mock_fetcher.fetch.return_value = dummy_data
    scanner.fetcher = mock_fetcher

    sharpe = scanner._calculate_sharpe("AAPL")
    assert isinstance(sharpe, float)


def test_scan_sectors(scanner):
    dummy_df = pd.DataFrame(
        {"Symbol": ["AAPL", "JNJ"], "GICS Sector": ["Information Technology", "Health Care"]}
    )
    with (
        patch.object(scanner, "fetch_sp500_list", return_value=dummy_df),
        patch.object(scanner, "_calculate_sharpe", return_value=1.5),
    ):

        results = scanner.scan_sectors(stocks_per_sector=1)
        assert "sectors" in results
        assert "Information Technology" in results["sectors"]
        assert "Health Care" in results["sectors"]
        assert len(results["sectors"]["Information Technology"]) == 1


def test_get_top_stocks(scanner):
    scan_results = {
        "sectors": {
            "Information Technology": [{"ticker": "AAPL", "sharpe": 1.5}],
            "Health Care": [{"ticker": "JNJ", "sharpe": 1.2}],
        }
    }
    top_stocks = scanner.get_top_stocks(scan_results)
    assert isinstance(top_stocks, list)
    assert set(top_stocks) == {"AAPL", "JNJ"}
