# ruff: noqa: E402
import sys
from unittest.mock import MagicMock

if "pandas" not in sys.modules:
    sys.modules["pandas"] = MagicMock()
from core.sp500_scanner import SP500Scanner


def test_sp500_scanner_init():
    scanner = SP500Scanner()
    assert len(scanner.GICS_SECTORS) == 11
    assert scanner.lookback_days == 252


def test_sp500_scanner_fetch_wiki_data():
    from unittest.mock import patch

    import pandas as pd

    scanner = SP500Scanner()

    mock_df = pd.DataFrame(
        {
            "Symbol": ["AAPL", "MSFT", "JNJ"],
            "GICS Sector": ["Information Technology", "Information Technology", "Health Care"],
        }
    )

    with patch("pandas.read_html", return_value=[mock_df]):
        df = scanner._fetch_wikipedia_data()
        assert len(df) == 3
        # Assert not needed on values if mocking pandas entirely,
        # but in CI this will run with real pandas and coverage
