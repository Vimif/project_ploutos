import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from core.sp500_scanner import SP500Scanner


class TestSP500Scanner(unittest.TestCase):
    def setUp(self):
        # Patch UniversalDataFetcher inside the scanner module
        self.fetcher_patcher = patch("core.sp500_scanner.UniversalDataFetcher")
        self.mock_fetcher_cls = self.fetcher_patcher.start()
        self.mock_fetcher_instance = self.mock_fetcher_cls.return_value

        # Scanner instance
        self.scanner = SP500Scanner()
        # Mock fetch_sp500_list to avoid HTTP requests
        self.scanner.fetch_sp500_list = MagicMock(
            return_value=pd.DataFrame(
                {
                    "Symbol": ["AAPL", "MSFT"],
                    "GICS Sector": ["Information Technology", "Information Technology"],
                }
            )
        )

    def tearDown(self):
        self.fetcher_patcher.stop()

    def test_scan_sectors(self):
        # Mock data return from fetcher
        dates = pd.date_range("2023-01-01", periods=300)
        df = pd.DataFrame(
            {
                "Close": [100 + i * 0.1 for i in range(300)],  # Steady uptrend
                "Volume": [1000000] * 300,
            },
            index=dates,
        )

        self.mock_fetcher_instance.fetch.return_value = df

        # Run scan
        results = self.scanner.scan_sectors(stocks_per_sector=1, max_workers=1)

        self.assertIn("Information Technology", results["sectors"])
        self.assertTrue(len(results["sectors"]["Information Technology"]) > 0)
        self.assertEqual(results["total_stocks"], 1)

    def test_calculate_sharpe(self):
        # Mock data for sharpe calculation
        dates = pd.date_range("2023-01-01", periods=300)
        df = pd.DataFrame(
            {"Close": [100 + i * 0.1 + (i % 2) for i in range(300)], "Volume": [1000000] * 300},
            index=dates,
        )
        self.mock_fetcher_instance.fetch.return_value = df

        sharpe = self.scanner._calculate_sharpe("AAPL")
        self.assertIsInstance(sharpe, float)
        self.assertFalse(pd.isna(sharpe))
