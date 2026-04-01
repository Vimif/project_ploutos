# ruff: noqa: E402
import sys
from unittest.mock import MagicMock, patch

import pandas as pd

mock_lxml = MagicMock()
if "lxml" not in sys.modules:
    sys.modules["lxml"] = mock_lxml

from core.sp500_scanner import SP500Scanner


def test_sp500_scanner_fetch_list():
    with patch("pandas.read_html") as mock_read_html:
        mock_df = pd.DataFrame(
            {
                "Symbol": ["AAPL", "MSFT", "GOOGL"],
                "Security": ["Apple", "MS", "Ggl"],
                "GICS Sector": [
                    "Information Technology",
                    "Information Technology",
                    "Information Technology",
                ],
                "GICS Sub-Industry": ["Tech", "Tech", "Tech"],
            }
        )
        mock_read_html.return_value = [mock_df]

        scanner = SP500Scanner()
        df = scanner.fetch_sp500_list()

        assert len(df) == 3
        assert "AAPL" in df["Symbol"].values


def test_sp500_scanner_calculate_sharpe():
    with patch("core.sp500_scanner.UniversalDataFetcher.fetch") as mock_fetch:
        mock_df = pd.DataFrame({"Close": [150.0, 151.0, 153.0, 150.0, 155.0]})
        mock_df.index = pd.date_range("2023-01-01", periods=5)
        mock_fetch.return_value = mock_df

        scanner = SP500Scanner()
        sharpe = scanner._calculate_sharpe("AAPL")

        assert sharpe != 0.0


def test_sp500_scanner_scan_sectors():
    with (
        patch.object(SP500Scanner, "fetch_sp500_list") as mock_fetch_list,
        patch.object(SP500Scanner, "_calculate_sharpe") as mock_sharpe,
    ):

        mock_df = pd.DataFrame(
            {
                "Symbol": ["AAPL", "MSFT", "JNJ"],
                "Security": ["Apple", "MS", "JNJ"],
                "GICS Sector": ["IT", "IT", "Health"],
                "GICS Sub-Industry": ["Tech", "Tech", "Health"],
            }
        )
        mock_fetch_list.return_value = mock_df
        mock_sharpe.return_value = 1.5

        scanner = SP500Scanner()
        results = scanner.scan_sectors()

        assert len(results) > 0
