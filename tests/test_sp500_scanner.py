import unittest
from unittest.mock import MagicMock, patch
from core.sp500_scanner import SP500Scanner

class TestSP500Scanner(unittest.TestCase):
    def setUp(self):
        # Prevent mkdir
        with patch('pathlib.Path.mkdir'):
            self.scanner = SP500Scanner()

    @patch('core.sp500_scanner.UniversalDataFetcher')
    def test_scan_market(self, mock_fetcher_cls):
        # We need to mock the fetcher instance inside scanner
        mock_fetcher_instance = mock_fetcher_cls.return_value
        self.scanner.fetcher = mock_fetcher_instance

        # Mock SP500 list
        import pandas as pd
        self.scanner.fetch_sp500_list = MagicMock(return_value=pd.DataFrame({
            "Symbol": ["AAPL", "MSFT"],
            "GICS Sector": ["Information Technology", "Information Technology"]
        }))

        # Mock calculate_sharpe to avoid threading complexity or actual calculation
        self.scanner._calculate_sharpe = MagicMock(side_effect=[2.0, 1.5])

        results = self.scanner.scan_sectors(stocks_per_sector=1, max_workers=1)

        self.assertIn("Information Technology", results["sectors"])
        self.assertEqual(results["sectors"]["Information Technology"], ["AAPL"])
        self.assertEqual(results["total_stocks"], 1)

    def test_calculate_sharpe(self):
        # Mock fetcher
        self.scanner.fetcher.fetch = MagicMock()
        import pandas as pd
        import numpy as np

        # DataFrame with constant growth
        df = pd.DataFrame({
            "Close": [100, 101, 102, 103] + [103 + i for i in range(250)]
        })
        self.scanner.fetcher.fetch.return_value = df

        sharpe = self.scanner._calculate_sharpe("TEST")
        # Should be a number
        self.assertIsInstance(sharpe, float)
        self.assertFalse(np.isnan(sharpe))

if __name__ == "__main__":
    unittest.main()
