import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from core.sp500_scanner import SP500Scanner


@pytest.fixture
def mock_fetcher():
    with patch("core.sp500_scanner.UniversalDataFetcher") as mock:
        yield mock.return_value


@pytest.fixture
def scanner(mock_fetcher):
    # Use a temp cache dir, lookback_days=60 to satisfy min requirements
    return SP500Scanner(cache_dir="/tmp/sp500_cache_test", lookback_days=60)


def test_fetch_sp500_list(scanner):
    # Mock requests.get
    html_content = """
    <table>
        <tr>
            <th>Symbol</th>
            <th>Security</th>
            <th>GICS Sector</th>
            <th>GICS Sub-Industry</th>
        </tr>
        <tr>
            <td>AAPL</td>
            <td>Apple Inc.</td>
            <td>Information Technology</td>
            <td>Technology Hardware</td>
        </tr>
        <tr>
            <td>MSFT</td>
            <td>Microsoft Corp.</td>
            <td>Information Technology</td>
            <td>Systems Software</td>
        </tr>
        <tr>
            <td>JNJ</td>
            <td>Johnson & Johnson</td>
            <td>Health Care</td>
            <td>Pharmaceuticals</td>
        </tr>
    </table>
    """

    with patch("requests.get") as mock_get:
        mock_get.return_value.text = html_content
        mock_get.return_value.status_code = 200

        df = scanner.fetch_sp500_list()

        assert len(df) == 3
        assert "AAPL" in df["Symbol"].values
        assert "MSFT" in df["Symbol"].values
        assert "JNJ" in df["Symbol"].values
        assert "Information Technology" in df["GICS Sector"].values


def test_calculate_sharpe(scanner, mock_fetcher):
    # Create fake daily data (200 days)
    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    # Positive trend with some volatility
    prices = np.linspace(100, 150, 200) + np.random.normal(0, 2, 200)
    df = pd.DataFrame({"Close": prices}, index=dates)

    mock_fetcher.fetch.return_value = df

    sharpe = scanner._calculate_sharpe("TEST")

    # Should be positive given the trend
    assert not np.isnan(sharpe)
    assert isinstance(sharpe, float)


def test_calculate_sharpe_no_data(scanner, mock_fetcher):
    mock_fetcher.fetch.return_value = None
    sharpe = scanner._calculate_sharpe("EMPTY")
    assert np.isnan(sharpe)


def test_scan_sectors(scanner):
    # Mock list
    mock_constituents = pd.DataFrame(
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

    # Mock sharpe calculation
    def mock_sharpe(ticker):
        scores = {"AAPL": 2.5, "MSFT": 1.5, "JNJ": 1.2, "PFE": 0.8}
        return scores.get(ticker, 0.0)

    with patch.object(scanner, "fetch_sp500_list", return_value=mock_constituents):
        with patch.object(scanner, "_calculate_sharpe", side_effect=mock_sharpe):
            # 1 stock per sector
            results = scanner.scan_sectors(stocks_per_sector=1, max_workers=1)

            assert "Information Technology" in results["sectors"]
            assert "Health Care" in results["sectors"]
            assert results["sectors"]["Information Technology"] == ["AAPL"]  # Best sharpe
            assert results["sectors"]["Health Care"] == ["JNJ"]  # Best sharpe
            assert results["total_stocks"] == 2


def test_helpers(scanner):
    results = {"sectors": {"Tech": ["AAPL", "MSFT"], "Health": ["JNJ"]}}

    # get_top_stocks
    stocks = scanner.get_top_stocks(results)
    assert len(stocks) == 3
    assert "AAPL" in stocks

    # save_results / load_cached_results
    with patch("builtins.open", create=True) as mock_open:
        with patch("json.dump") as mock_json_dump:
            scanner.save_results(results, filepath="test.json")
            mock_json_dump.assert_called()

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.stat") as mock_stat:
                # Recent enough
                mock_stat.return_value.st_mtime = pd.Timestamp.now().timestamp()
                with patch("json.load", return_value=results):
                    loaded = scanner.load_cached_results()
                    assert loaded == results
