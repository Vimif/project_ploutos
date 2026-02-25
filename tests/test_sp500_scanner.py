import pytest
import shutil
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from core.sp500_scanner import SP500Scanner

class TestSP500Scanner:
    @pytest.fixture
    def cache_dir(self):
        # Create a temporary cache directory
        path = Path("tests/temp_sp500_cache")
        path.mkdir(exist_ok=True, parents=True)
        yield path
        # Cleanup
        if path.exists():
            shutil.rmtree(path)

    @pytest.fixture
    def scanner(self, cache_dir):
        return SP500Scanner(cache_dir=str(cache_dir))

    def test_initialization(self, scanner, cache_dir):
        assert scanner.cache_dir == cache_dir
        assert scanner.lookback_days == 252

    @patch("core.sp500_scanner.UniversalDataFetcher")
    def test_fetch_sp500_list(self, mock_fetcher_cls, scanner):
        import pandas as pd

        # Mock requests.get to return fake HTML table
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = """
            <table>
                <tr>
                    <th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th>
                </tr>
                <tr>
                    <td>AAPL</td><td>Apple Inc.</td><td>Information Technology</td><td>Consumer Electronics</td>
                </tr>
                <tr>
                    <td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td><td>Multi-Sector Holdings</td>
                </tr>
            </table>
            """
            mock_get.return_value = mock_response

            df = scanner.fetch_sp500_list()

            assert len(df) == 2
            assert df.iloc[0]["Symbol"] == "AAPL"
            assert df.iloc[1]["Symbol"] == "BRK-B" # Check symbol cleaning

    @patch("core.sp500_scanner.UniversalDataFetcher")
    def test_calculate_sharpe(self, mock_fetcher_cls, scanner):
        import pandas as pd
        import numpy as np

        # Mock fetcher
        scanner.fetcher.fetch = MagicMock()

        # Create dummy price data (uptrend)
        dates = pd.date_range(start="2023-01-01", periods=300) # > 252 lookback
        close = np.linspace(100, 150, 300)
        df = pd.DataFrame({"Close": close}, index=dates)

        scanner.fetcher.fetch.return_value = df

        sharpe = scanner._calculate_sharpe("AAPL")

        # Sharpe should be positive for steady uptrend
        assert sharpe > 0
        assert isinstance(sharpe, float)

        # Test Not Found / Error
        scanner.fetcher.fetch.side_effect = Exception("Error")
        sharpe_err = scanner._calculate_sharpe("INVALID")
        assert np.isnan(sharpe_err)

    def test_save_and_load_results(self, scanner):
        results = {
            "sectors": {"Technology": ["AAPL"]},
            "timestamp": "2023-01-01"
        }

        scanner.save_results(results)

        # Verify file exists
        expected_file = scanner.cache_dir / "latest_scan.json"
        assert expected_file.exists()

        # Verify content
        with open(expected_file, "r") as f:
            loaded = json.load(f)
        assert loaded == results

        # Test load
        loaded_res = scanner.load_cached_results()
        assert loaded_res == results
