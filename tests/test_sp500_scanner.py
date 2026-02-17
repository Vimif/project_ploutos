import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
from core.sp500_scanner import SP500Scanner
from datetime import datetime

class TestSP500Scanner:
    @pytest.fixture
    def scanner(self):
        with patch("core.sp500_scanner.UniversalDataFetcher") as mock_fetcher:
            scanner = SP500Scanner(cache_dir="tests/cache_test")
            scanner.fetcher = mock_fetcher.return_value
            yield scanner

    def test_initialization(self, scanner):
        assert scanner.lookback_days == 252
        assert scanner.risk_free_rate == 0.04
        assert isinstance(scanner.cache_dir, Path)

    @patch("requests.get")
    @patch("pandas.read_html")
    def test_fetch_sp500_list(self, mock_read_html, mock_get, scanner):
        # Mock requests
        mock_resp = MagicMock()
        mock_resp.text = "fake html"
        mock_get.return_value = mock_resp

        # Mock pandas
        mock_df = pd.DataFrame({
            "Symbol": ["AAPL", "BRK.B", "MSFT"],
            "Security": ["Apple", "Berkshire", "Microsoft"],
            "GICS Sector": ["Information Technology", "Financials", "Information Technology"],
            "GICS Sub-Industry": ["Tech", "Finance", "Tech"]
        })
        mock_read_html.return_value = [mock_df]

        df = scanner.fetch_sp500_list()

        assert len(df) == 3
        assert "BRK-B" in df["Symbol"].values # Check cleaning
        assert "AAPL" in df["Symbol"].values

    def test_calculate_sharpe_valid(self, scanner):
        # Mock data fetcher
        dates = pd.date_range("2023-01-01", periods=300, freq="D")
        # Generate trend with volatility
        prices = np.linspace(100, 150, 300) + np.random.normal(0, 2, 300)
        df = pd.DataFrame({"Close": prices}, index=dates)

        scanner.fetcher.fetch.return_value = df

        sharpe = scanner._calculate_sharpe("TEST")
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
        assert sharpe > 0 # Positive trend

    def test_calculate_sharpe_insufficient_data(self, scanner):
        scanner.fetcher.fetch.return_value = None
        assert np.isnan(scanner._calculate_sharpe("TEST"))

        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        df = pd.DataFrame({"Close": np.random.randn(50)}, index=dates)
        scanner.fetcher.fetch.return_value = df
        # Should be too short vs 252 lookback (0.7 * 252 = 176)
        assert np.isnan(scanner._calculate_sharpe("TEST"))

    @patch("core.sp500_scanner.SP500Scanner.fetch_sp500_list")
    @patch("core.sp500_scanner.SP500Scanner._calculate_sharpe")
    def test_scan_sectors(self, mock_sharpe, mock_list, scanner):
        # Mock List
        mock_list.return_value = pd.DataFrame({
            "Symbol": ["A", "B", "C", "D"],
            "GICS Sector": ["Information Technology", "Information Technology", "Health Care", "Health Care"]
        })

        # Mock Sharpe
        # A=2.0, B=1.0, C=0.5, D=NaN
        def side_effect(ticker):
            scores = {"A": 2.0, "B": 1.0, "C": 0.5, "D": np.nan}
            return scores.get(ticker, np.nan)

        mock_sharpe.side_effect = side_effect

        results = scanner.scan_sectors(stocks_per_sector=1, max_workers=1)

        assert "Information Technology" in results["sectors"]
        assert results["sectors"]["Information Technology"] == ["A"]
        assert "Health Care" in results["sectors"]
        assert results["sectors"]["Health Care"] == ["C"]
        assert results["total_stocks"] == 2

    def test_get_top_stocks(self, scanner):
        scan_results = {
            "sectors": {
                "Tech": ["A", "B"],
                "Health": ["C"]
            }
        }
        top = scanner.get_top_stocks(scan_results)
        assert len(top) == 3
        assert "A" in top and "C" in top

    def test_save_and_load_results(self, scanner):
        results = {"total_stocks": 10, "sectors": {}}

        # Mock open
        with patch("builtins.open", mock_open(read_data=u'{"total_stocks": 10}')) as mock_file:
            with patch("pathlib.Path.exists") as mock_exists:
                with patch("pathlib.Path.stat") as mock_stat:
                    with patch("pathlib.Path.mkdir") as mock_mkdir:
                        # Test Save
                        scanner.save_results(results, "test.json")
                        mock_file.assert_called_with(Path("test.json"), "w", encoding="utf-8")

                        # Test Load
                        mock_exists.return_value = True
                        mock_stat.return_value.st_mtime = datetime.now().timestamp()

                        loaded = scanner.load_cached_results()
                        assert loaded is not None
                        assert loaded["total_stocks"] == 10
