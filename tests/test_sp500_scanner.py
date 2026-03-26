# ruff: noqa: E402
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from core.sp500_scanner import SP500Scanner


@pytest.fixture
def scanner():
    return SP500Scanner(cache_dir="test_cache", lookback_days=100)

from pathlib import Path


@patch("pathlib.Path.mkdir")
def test_init(mock_mkdir):
    s = SP500Scanner(cache_dir="test_cache2", lookback_days=50)
    assert s.cache_dir == Path("test_cache2")
    assert s.lookback_days == 50
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

@patch("pandas.read_html")
@patch("requests.get")
def test_fetch_sp500_list(mock_get, mock_read_html, scanner):
    mock_response = MagicMock()
    mock_response.text = "<html></html>"
    mock_get.return_value = mock_response

    mock_df = pd.DataFrame({
        "Symbol": ["AAPL", "BRK.B"],
        "Security": ["Apple", "Berkshire Hathaway"],
        "GICS Sector": ["Information Technology", "Financials"],
        "GICS Sub-Industry": ["Hardware", "Insurance"]
    })
    mock_read_html.return_value = [mock_df]

    result = scanner.fetch_sp500_list()

    assert len(result) == 2
    assert "AAPL" in result["Symbol"].values
    assert "BRK-B" in result["Symbol"].values
    mock_get.assert_called_once()
    mock_read_html.assert_called_once()

@patch("core.sp500_scanner.UniversalDataFetcher.fetch")
def test_calculate_sharpe(mock_fetch, scanner):
    dates = pd.date_range("2023-01-01", periods=100, freq="1d")
    prices = np.linspace(100, 150, 100) # Steady increase
    df = pd.DataFrame({"Close": prices}, index=dates)
    mock_fetch.return_value = df

    sharpe = scanner._calculate_sharpe("AAPL")

    assert isinstance(sharpe, float)
    assert not np.isnan(sharpe)
    assert sharpe > 0

@patch("core.sp500_scanner.UniversalDataFetcher.fetch")
def test_calculate_sharpe_no_data(mock_fetch, scanner):
    mock_fetch.side_effect = Exception("No data")
    sharpe = scanner._calculate_sharpe("INVALID")
    assert np.isnan(sharpe)

@patch.object(SP500Scanner, "fetch_sp500_list")
@patch.object(SP500Scanner, "_calculate_sharpe")
def test_scan_sectors(mock_sharpe, mock_fetch, scanner):
    mock_fetch.return_value = pd.DataFrame({
        "Symbol": ["AAPL", "MSFT", "JNJ"],
        "GICS Sector": ["Information Technology", "Information Technology", "Health Care"]
    })

    # MSFT > AAPL
    def side_effect(ticker):
        return {"AAPL": 1.5, "MSFT": 2.0, "JNJ": 1.2}[ticker]
    mock_sharpe.side_effect = side_effect

    results = scanner.scan_sectors(stocks_per_sector=1, max_workers=1)

    assert results["total_stocks"] == 2
    assert "sectors" in results
    assert results["sectors"]["Information Technology"] == ["MSFT"]
    assert results["sectors"]["Health Care"] == ["JNJ"]
    assert results["sharpe_ratios"]["MSFT"] == 2.0

def test_get_top_stocks(scanner):
    scan_results = {
        "sectors": {
            "IT": ["AAPL", "MSFT"],
            "Health": ["JNJ"]
        }
    }

    top = scanner.get_top_stocks(scan_results)
    assert len(top) == 3
    assert "AAPL" in top
    assert "JNJ" in top

@patch("pathlib.Path.mkdir")
@patch("builtins.open")
def test_save_results(mock_open, mock_mkdir, scanner):
    results = {"test": 123}
    scanner.save_results(results, "test.json")
    mock_open.assert_called_once_with(Path("test.json"), "w", encoding="utf-8")
    mock_mkdir.assert_called_once()

@patch("pathlib.Path.exists")
@patch("pathlib.Path.stat")
@patch("builtins.open")
def test_load_cached_results(mock_open, mock_stat, mock_exists, scanner):
    mock_exists.return_value = True
    import time
    mock_stat_obj = MagicMock()
    mock_stat_obj.st_mtime = time.time() - 3600
    mock_stat.return_value = mock_stat_obj

    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file

    with patch("json.load") as mock_json_load:
        mock_json_load.return_value = {"cached": True}
        res = scanner.load_cached_results()

    assert res == {"cached": True}
