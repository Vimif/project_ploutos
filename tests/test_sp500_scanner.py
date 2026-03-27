import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from core.sp500_scanner import SP500Scanner
from datetime import datetime

# Sample HTML representing a simplified Wikipedia S&P 500 table
DUMMY_HTML = """
<html>
<body>
<table id="constituents">
  <thead>
    <tr>
      <th>Symbol</th>
      <th>Security</th>
      <th>GICS Sector</th>
      <th>GICS Sub-Industry</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>AAPL</td><td>Apple Inc.</td><td>Information Technology</td><td>Technology Hardware</td></tr>
    <tr><td>MSFT</td><td>Microsoft Corp.</td><td>Information Technology</td><td>Systems Software</td></tr>
    <tr><td>JNJ</td><td>Johnson & Johnson</td><td>Health Care</td><td>Pharmaceuticals</td></tr>
    <tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td><td>Multi-Sector Holdings</td></tr>
  </tbody>
</table>
</body>
</html>
"""


@pytest.fixture
def mock_scanner():
    with patch("core.sp500_scanner.Path.mkdir"):
        return SP500Scanner(cache_dir="dummy_cache", lookback_days=252)


class TestSP500Scanner:
    @patch("requests.get")
    def test_fetch_sp500_list(self, mock_get, mock_scanner):
        # Setup mock response
        mock_resp = MagicMock()
        mock_resp.text = DUMMY_HTML
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        df = mock_scanner.fetch_sp500_list()

        # Verify
        mock_get.assert_called_once()
        assert len(df) == 4
        assert list(df.columns) == ["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]
        # Check replacement of '.' with '-'
        assert df.iloc[3]["Symbol"] == "BRK-B"

    @patch("core.sp500_scanner.UniversalDataFetcher.fetch")
    def test_calculate_sharpe(self, mock_fetch, mock_scanner):
        # Create dummy returns dataframe: steady growth
        dates = pd.date_range("2023-01-01", periods=300)
        prices = np.exp(np.linspace(0, 1, 300))  # steady exponential growth
        df = pd.DataFrame({"Close": prices}, index=dates)
        mock_fetch.return_value = df

        sharpe = mock_scanner._calculate_sharpe("AAPL")

        # Steady growth will have low volatility and high returns, leading to a high Sharpe
        assert not np.isnan(sharpe)
        assert sharpe > 0

    @patch("core.sp500_scanner.UniversalDataFetcher.fetch")
    def test_calculate_sharpe_insufficient_data(self, mock_fetch, mock_scanner):
        # Not enough data (less than required lookback)
        dates = pd.date_range("2023-01-01", periods=10)
        prices = np.linspace(100, 110, 10)
        df = pd.DataFrame({"Close": prices}, index=dates)
        mock_fetch.return_value = df

        sharpe = mock_scanner._calculate_sharpe("AAPL")
        assert np.isnan(sharpe)

    @patch.object(SP500Scanner, "fetch_sp500_list")
    @patch.object(SP500Scanner, "_calculate_sharpe")
    def test_scan_sectors(self, mock_sharpe, mock_fetch_list, mock_scanner):
        # Mock constituency list
        mock_fetch_list.return_value = pd.DataFrame(
            {
                "Symbol": ["AAPL", "MSFT", "JNJ", "PFE", "V", "MA"],
                "GICS Sector": [
                    "Information Technology",
                    "Information Technology",
                    "Health Care",
                    "Health Care",
                    "Information Technology",
                    "Information Technology",
                ],
            }
        )

        # Mock sharpe ratios (AAPL > MSFT, V > MA)
        sharpe_vals = {"AAPL": 1.5, "MSFT": 1.2, "V": 1.0, "MA": 0.8, "JNJ": 0.5, "PFE": 0.2}
        mock_sharpe.side_effect = lambda t: sharpe_vals[t]

        results = mock_scanner.scan_sectors(stocks_per_sector=2, max_workers=2)

        # Verify IT sector output (top 2 out of 4)
        it_stocks = results["sectors"]["Information Technology"]
        assert len(it_stocks) == 2
        assert "AAPL" in it_stocks
        assert "MSFT" in it_stocks

        # Verify Health Care sector
        hc_stocks = results["sectors"]["Health Care"]
        assert len(hc_stocks) == 2
        assert "JNJ" in hc_stocks

    def test_get_top_stocks(self, mock_scanner):
        scan_results = {
            "sectors": {"Information Technology": ["AAPL", "MSFT"], "Health Care": ["JNJ", "PFE"]}
        }
        top_stocks = mock_scanner.get_top_stocks(scan_results)
        assert set(top_stocks) == {"AAPL", "MSFT", "JNJ", "PFE"}

    @patch("core.sp500_scanner.Path.exists")
    @patch("core.sp500_scanner.Path.stat")
    @patch("builtins.open")
    def test_load_cached_results_valid(self, mock_open, mock_stat, mock_exists, mock_scanner):
        mock_exists.return_value = True

        # Mock file stat to be recent (now)
        mock_stat_obj = MagicMock()
        mock_stat_obj.st_mtime = datetime.now().timestamp()
        mock_stat.return_value = mock_stat_obj

        # Mock file content
        import io
        import json

        dummy_data = {"scan_date": "2024-01-01", "total_stocks": 2}
        mock_open.return_value = io.StringIO(json.dumps(dummy_data))

        res = mock_scanner.load_cached_results(max_age_days=30)
        assert res is not None
        assert res["total_stocks"] == 2

    @patch("core.sp500_scanner.Path.exists")
    @patch("core.sp500_scanner.Path.stat")
    def test_load_cached_results_expired(self, mock_stat, mock_exists, mock_scanner):
        mock_exists.return_value = True

        # Mock file stat to be old (60 days ago)
        mock_stat_obj = MagicMock()
        mock_stat_obj.st_mtime = (datetime.now() - pd.Timedelta(days=60)).timestamp()
        mock_stat.return_value = mock_stat_obj

        res = mock_scanner.load_cached_results(max_age_days=30)
        assert res is None
