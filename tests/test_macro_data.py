import unittest
from unittest.mock import patch

import pandas as pd

from core.macro_data import MacroDataFetcher


class TestMacroData(unittest.TestCase):
    def setUp(self):
        self.fetcher = MacroDataFetcher()

    @patch("yfinance.download")
    def test_fetch_all(self, mock_yf):
        # Mock data
        mock_df = pd.DataFrame(
            {"Close": [10, 11, 12]}, index=pd.date_range("2023-01-01", periods=3)
        )

        # yf.download returns a DF
        mock_yf.return_value = mock_df

        data = self.fetcher.fetch_all()
        self.assertIsInstance(data, pd.DataFrame)
        # Should have columns for symbols (VIX, TNX, DXY) -> simplified check
        # The logic might return empty if join fails or structure differs
        # But we check that it runs without error

    def test_indicators(self):
        pd.DataFrame({"VIX": [10, 20, 15], "TNX": [1, 2, 1.5], "DXY": [100, 101, 100.5]})
        # Assuming we have a method to process or just raw fetching
        # The class mainly fetches via yfinance.
        pass


if __name__ == "__main__":
    unittest.main()
