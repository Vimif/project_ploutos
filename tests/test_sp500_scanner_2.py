# ruff: noqa: E402
import time
from unittest.mock import MagicMock, patch

from core.sp500_scanner import SP500Scanner


def test_sp500_scanner_save_results():
    with patch("core.sp500_scanner.Path.mkdir"), \
         patch("core.sp500_scanner.json.dump"), \
         patch("builtins.open", new_callable=MagicMock):

        scanner = SP500Scanner()
        # Mock fetch to not do work
        scanner.save_results({"a": 1})
        # Passes without error

def test_sp500_scanner_load_results():
    with patch("core.sp500_scanner.Path.exists") as mock_exists, \
         patch("core.sp500_scanner.Path.stat") as mock_stat, \
         patch("builtins.open", new_callable=MagicMock), \
         patch("core.sp500_scanner.json.load") as mock_load, \
         patch("core.sp500_scanner.Path.mkdir"), \
         patch("core.sp500_scanner.Path.is_dir") as mock_isdir:

        mock_exists.return_value = True
        mock_isdir.return_value = True

        current_time = time.time()

        class StatResult:
            st_mtime = current_time
            st_mode = 16877

        mock_stat.return_value = StatResult()

        mock_load.return_value = {"a": 1}

        scanner = SP500Scanner()

        results = scanner.load_cached_results()
        assert results == {"a": 1}

def test_sp500_scanner_get_top_stocks():
    scanner = SP500Scanner()
    results = {
        "sectors": {
            "IT": {"AAPL": {"sharpe": 1.5, "symbol": "AAPL"}, "MSFT": {"sharpe": 1.2, "symbol": "MSFT"}},
            "Health": {"JNJ": {"sharpe": 1.0, "symbol": "JNJ"}}
        }
    }
    top = scanner.get_top_stocks(results)
    assert len(top) == 3
    assert "AAPL" in top
