import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import pathlib

# Mock Path early
import core.sp500_scanner
core.sp500_scanner.Path = pathlib.Path

from core.sp500_scanner import SP500Scanner

@patch("core.sp500_scanner.pd.read_html")
def test_fetch_sp500_list(mock_read_html):
    mock_df = pd.DataFrame({
        "Symbol": ["AAPL", "MSFT", "NVDA"],
        "Security": ["Apple", "Microsoft", "Nvidia"],
        "GICS Sector": ["Tech", "Tech", "Tech"],
        "GICS Sub-Industry": ["Hardware", "Software", "Semiconductors"]
    })
    mock_read_html.return_value = [mock_df]
    scanner = SP500Scanner()
    df = scanner.fetch_sp500_list()
    assert len(df) == 3
    assert "Symbol" in df.columns

def test_get_top_stocks():
    scanner = SP500Scanner()
    scan_results = {
        "sectors": {
            "Technology": [
                "AAPL",
                "MSFT"
            ],
            "Health Care": [
                "JNJ"
            ]
        }
    }
    top = scanner.get_top_stocks(scan_results)
    assert len(top) == 3
    assert "AAPL" in top
    assert "MSFT" in top
    assert "JNJ" in top
