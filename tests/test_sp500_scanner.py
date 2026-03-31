import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, mock_open

from core.sp500_scanner import SP500Scanner

@pytest.fixture
def scanner():
    # Patch mkdir on Path inside init
    with patch("pathlib.Path.mkdir"):
        return SP500Scanner(cache_dir="test_cache", lookback_days=252)

def test_fetch_sp500_list(scanner):
    mock_html = """
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
            <td>Technology Hardware, Storage & Peripherals</td>
        </tr>
        <tr>
            <td>BRK.B</td>
            <td>Berkshire Hathaway</td>
            <td>Financials</td>
            <td>Multi-Sector Holdings</td>
        </tr>
    </table>
    """

    mock_response = MagicMock()
    mock_response.text = mock_html
    mock_response.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_response):
        # Also patch pd.read_html just in case lxml is missing in CI
        df = pd.DataFrame({
            "Symbol": ["AAPL", "BRK.B"],
            "Security": ["Apple Inc.", "Berkshire Hathaway"],
            "GICS Sector": ["Information Technology", "Financials"],
            "GICS Sub-Industry": ["Technology Hardware", "Multi-Sector"]
        })
        with patch("pandas.read_html", return_value=[df]):
            result = scanner.fetch_sp500_list()

            assert len(result) == 2
            assert result.iloc[0]["Symbol"] == "AAPL"
            # Ensure dot was replaced with dash
            assert result.iloc[1]["Symbol"] == "BRK-B"

def test_calculate_sharpe_valid_data(scanner):
    # Simulate valid dataframe response
    idx = pd.date_range("2022-01-01", periods=300, freq="B")

    # Generate some realistic returns (mean positive, std normal)
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 300)
    prices = 100 * np.cumprod(1 + returns)

    df = pd.DataFrame({"Close": prices}, index=idx)

    scanner.fetcher.fetch = MagicMock(return_value=df)

    sharpe = scanner._calculate_sharpe("AAPL")

    # Check that sharpe is a valid float
    assert isinstance(sharpe, float)
    assert not np.isnan(sharpe)

def test_calculate_sharpe_no_data(scanner):
    # Fetch returns None
    scanner.fetcher.fetch = MagicMock(return_value=None)
    sharpe = scanner._calculate_sharpe("INVALID")
    assert np.isnan(sharpe)

    # Fetch returns short dataframe
    scanner.fetcher.fetch = MagicMock(return_value=pd.DataFrame({"Close": [100, 101]}))
    sharpe = scanner._calculate_sharpe("SHORT")
    assert np.isnan(sharpe)

    # Fetch throws exception
    scanner.fetcher.fetch = MagicMock(side_effect=Exception("API Error"))
    sharpe = scanner._calculate_sharpe("ERROR")
    assert np.isnan(sharpe)

def test_scan_sectors(scanner):
    # Mock constituents
    constituents = pd.DataFrame({
        "Symbol": ["AAPL", "MSFT", "JPM", "BAC"],
        "GICS Sector": ["Information Technology", "Information Technology", "Financials", "Financials"]
    })
    scanner.fetch_sp500_list = MagicMock(return_value=constituents)

    # Mock _calculate_sharpe directly
    def mock_sharpe(ticker):
        scores = {"AAPL": 1.5, "MSFT": 1.2, "JPM": 0.8, "BAC": 1.1}
        return scores.get(ticker, np.nan)

    scanner._calculate_sharpe = MagicMock(side_effect=mock_sharpe)

    results = scanner.scan_sectors(stocks_per_sector=1, max_workers=2)

    assert results["total_stocks"] == 2
    assert "Information Technology" in results["sectors"]
    assert "Financials" in results["sectors"]

    # Apple should win IT, BAC should win Financials
    assert results["sectors"]["Information Technology"] == ["AAPL"]
    assert results["sectors"]["Financials"] == ["BAC"]

    # Verify top stocks extraction
    top_stocks = scanner.get_top_stocks(results)
    assert sorted(top_stocks) == sorted(["AAPL", "BAC"])

from pathlib import Path

@patch("pathlib.Path.mkdir")
def test_save_results(mock_mkdir, scanner):
    results = {"total_stocks": 2, "sectors": {"IT": ["AAPL"]}}

    m = mock_open()
    with patch("builtins.open", m):
        scanner.save_results(results, "dummy_path.json")

    m.assert_called_once_with(Path("dummy_path.json"), 'w', encoding='utf-8')
    handle = m()
    # Check that json.dump wrote string representation
    written = "".join([call.args[0] for call in handle.write.call_args_list])
    assert "total_stocks" in written
    assert "AAPL" in written

@patch("pathlib.Path.exists")
def test_load_cached_results_no_file(mock_exists, scanner):
    mock_exists.return_value = False
    assert scanner.load_cached_results() is None

@patch("pathlib.Path.exists")
@patch("pathlib.Path.stat")
def test_load_cached_results_too_old(mock_stat, mock_exists, scanner):
    mock_exists.return_value = True

    # Simulate file from 60 days ago
    import time
    old_time = time.time() - (60 * 24 * 3600)

    stat_mock = MagicMock()
    stat_mock.st_mtime = old_time
    mock_stat.return_value = stat_mock

    assert scanner.load_cached_results(max_age_days=30) is None

@patch("pathlib.Path.exists")
@patch("pathlib.Path.stat")
def test_load_cached_results_valid(mock_stat, mock_exists, scanner):
    mock_exists.return_value = True

    import time
    recent_time = time.time() - (5 * 24 * 3600)  # 5 days ago

    stat_mock = MagicMock()
    stat_mock.st_mtime = recent_time
    mock_stat.return_value = stat_mock

    mock_data = '{"total_stocks": 5}'
    m = mock_open(read_data=mock_data)

    with patch("builtins.open", m):
        cached = scanner.load_cached_results(max_age_days=30)

    assert cached is not None
    assert cached["total_stocks"] == 5