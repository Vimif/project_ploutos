import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
from core.sp500_scanner import SP500Scanner

@pytest.fixture
def mock_scanner(tmp_path):
    with patch("core.sp500_scanner.UniversalDataFetcher"):
        scanner = SP500Scanner(cache_dir=str(tmp_path), lookback_days=100)
        yield scanner

def test_init_creates_dir(tmp_path):
    with patch("core.sp500_scanner.UniversalDataFetcher"):
        SP500Scanner(cache_dir=str(tmp_path / "new_dir"))
        assert (tmp_path / "new_dir").exists()

@patch("pandas.read_html")
def test_fetch_sp500_list(mock_read_html, mock_scanner):
    mock_df = pd.DataFrame([{
        "Symbol": "AAPL",
        "Security": "Apple Inc.",
        "GICS Sector": "Information Technology",
        "GICS Sub-Industry": "Technology Hardware"
    }])
    mock_read_html.return_value = [mock_df]
    df = mock_scanner.fetch_sp500_list()
    assert not df.empty
    assert "Symbol" in df.columns
    assert "GICS Sector" in df.columns
    assert df.iloc[0]["Symbol"] == "AAPL"

def test_calculate_sharpe_success(mock_scanner):
    # Mock data fetcher return
    mock_df = pd.DataFrame({
        "Close": [100, 101, 102, 103, 104]
    })
    mock_scanner.fetcher.fetch_data.return_value = mock_df

    sharpe = mock_scanner._calculate_sharpe("AAPL")
    import numpy as np
    assert isinstance(sharpe, float)

def test_calculate_sharpe_failure(mock_scanner):
    mock_scanner.fetcher.fetch_data.side_effect = Exception("API Error")
    sharpe = mock_scanner._calculate_sharpe("INVALID")
    import numpy as np
    assert np.isnan(sharpe) or sharpe == -999.0

def test_save_and_load_results(mock_scanner, tmp_path):
    results = {"sectors": {"Information Technology": [{"ticker": "AAPL"}]}}

    # Save
    filepath = str(tmp_path / "test_scan.json")
    mock_scanner.save_results(results, filepath=filepath)
    assert Path(filepath).exists()

    # Load
    with patch.object(mock_scanner, "cache_dir", tmp_path):
        mock_scanner.save_results(results) # save to default
        loaded = mock_scanner.load_cached_results(max_age_days=1)
        assert loaded == results

def test_load_cached_results_expired(mock_scanner, tmp_path):
    # Simulate old file using patch on getmtime
    import time
    with patch("os.path.getmtime", return_value=time.time() - 86400 * 40):
        with patch("pathlib.Path.exists", return_value=True):
            # mock stat
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat_result = MagicMock()
                mock_stat_result.st_mtime = time.time() - 86400 * 40
                mock_stat.return_value = mock_stat_result
                res = mock_scanner.load_cached_results(max_age_days=30)
                assert res is None

def test_get_top_stocks(mock_scanner):
    scan_results = {
        "sectors": {
            "IT": ["AAPL", "MSFT"],
            "Health": ["JNJ"]
        }
    }
    stocks = mock_scanner.get_top_stocks(scan_results)
    assert set(stocks) == {"AAPL", "MSFT", "JNJ"}

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from core.sp500_scanner import SP500Scanner

@pytest.fixture
def mock_scanner(tmp_path):
    with patch("core.sp500_scanner.UniversalDataFetcher"):
        scanner = SP500Scanner(cache_dir=str(tmp_path), lookback_days=100)
        yield scanner

def test_scan_sectors(mock_scanner):
    # Mock self.fetch_sp500_list
    mock_df = pd.DataFrame([
        {"Symbol": "AAPL", "Security": "Apple", "GICS Sector": "Information Technology", "GICS Sub-Industry": "Hardware"},
        {"Symbol": "MSFT", "Security": "Microsoft", "GICS Sector": "Information Technology", "GICS Sub-Industry": "Software"},
        {"Symbol": "JNJ", "Security": "J&J", "GICS Sector": "Health Care", "GICS Sub-Industry": "Pharma"}
    ])
    mock_scanner.fetch_sp500_list = MagicMock(return_value=mock_df)

    # Mock self._calculate_sharpe
    def mock_sharpe(ticker):
        if ticker == "AAPL": return 1.5
        elif ticker == "MSFT": return 2.0
        elif ticker == "JNJ": return 1.0
        return -999.0
    mock_scanner._calculate_sharpe = MagicMock(side_effect=mock_sharpe)

    results = mock_scanner.scan_sectors(stocks_per_sector=1, max_workers=2)
    assert "sectors" in results

    # MSFT has higher sharpe than AAPL
    assert "Information Technology" in results["sectors"]
    it_stocks = results["sectors"]["Information Technology"]
    assert len(it_stocks) == 1
    assert it_stocks[0] == "MSFT"

def test_scan_sectors_handles_fetch_error(mock_scanner):
    mock_scanner.fetch_sp500_list = MagicMock(side_effect=Exception("Wiki down"))
    try:
        results = mock_scanner.scan_sectors()
        assert False, "Should have raised exception"
    except Exception as e:
        assert str(e) == "Wiki down"
