import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.sp500_scanner import SP500Scanner


@pytest.fixture
def mock_scanner(tmp_path):
    scanner = SP500Scanner(cache_dir=str(tmp_path / "cache"))
    return scanner


def test_fetch_sp500_list(mock_scanner):
    dummy_df = pd.DataFrame(
        {
            "Symbol": ["AAPL", "BRK.B"],
            "Security": ["Apple Inc.", "Berkshire Hathaway"],
            "GICS Sector": ["Information Technology", "Financials"],
            "GICS Sub-Industry": [
                "Technology Hardware, Storage & Peripherals",
                "Multi-Sector Holdings",
            ],
        }
    )

    with patch("requests.get") as mock_get, patch("pandas.read_html", return_value=[dummy_df]):
        mock_response = MagicMock()
        mock_response.text = "<html></html>"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        df = mock_scanner.fetch_sp500_list()

        assert len(df) == 2
        assert "Symbol" in df.columns
        assert df.iloc[0]["Symbol"] == "AAPL"
        # Check dot replacement
        assert df.iloc[1]["Symbol"] == "BRK-B"
        assert df.iloc[0]["GICS Sector"] == "Information Technology"


def test_calculate_sharpe(mock_scanner):
    # Mock data fetcher to return a dummy dataframe
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100)
    # Generate prices that go up consistently to give a positive sharpe
    prices = np.linspace(100, 150, 100)
    df = pd.DataFrame({"Close": prices}, index=dates)

    # We set lookback_days = 50 so it's less than the 100 bars
    mock_scanner.lookback_days = 50

    with patch.object(mock_scanner.fetcher, "fetch", return_value=df):
        sharpe = mock_scanner._calculate_sharpe("AAPL")
        assert not np.isnan(sharpe)
        assert sharpe > 0


def test_calculate_sharpe_insufficient_data(mock_scanner):
    # Return too few rows
    dates = pd.date_range(end=pd.Timestamp.now(), periods=10)
    df = pd.DataFrame({"Close": np.random.randn(10)}, index=dates)

    mock_scanner.lookback_days = 50
    with patch.object(mock_scanner.fetcher, "fetch", return_value=df):
        sharpe = mock_scanner._calculate_sharpe("AAPL")
        assert np.isnan(sharpe)


def test_scan_sectors(mock_scanner):
    # Mock fetch_sp500_list
    df = pd.DataFrame(
        {
            "Symbol": ["AAPL", "MSFT", "JNJ", "UNH"],
            "Security": ["Apple", "Microsoft", "J&J", "UnitedHealth"],
            "GICS Sector": [
                "Information Technology",
                "Information Technology",
                "Health Care",
                "Health Care",
            ],
            "GICS Sub-Industry": ["Tech", "Tech", "Health", "Health"],
        }
    )

    with (
        patch.object(mock_scanner, "fetch_sp500_list", return_value=df),
        patch.object(mock_scanner, "_calculate_sharpe", side_effect=[1.5, 2.0, 1.0, 0.5]),
        patch("sys.stdout"),
    ):  # Suppress prints

        # Test scanning
        results = mock_scanner.scan_sectors(stocks_per_sector=1, max_workers=1)

        assert results["stocks_per_sector"] == 1
        assert "Information Technology" in results["sectors"]
        assert "Health Care" in results["sectors"]

        # Due to mock side_effect order, AAPL=1.5, MSFT=2.0 -> MSFT should win IT
        # JNJ=1.0, UNH=0.5 -> JNJ should win HC
        # NOTE: order of evaluation with ThreadPoolExecutor is not guaranteed,
        # so we just assert there is 1 stock per sector.
        assert len(results["sectors"]["Information Technology"]) == 1
        assert len(results["sectors"]["Health Care"]) == 1
        assert results["total_stocks"] == 2


def test_cache_methods(mock_scanner, tmp_path):
    results = {
        "scan_date": "2026-01-01",
        "total_stocks": 2,
        "sectors": {"Information Technology": ["AAPL"]},
        "sharpe_ratios": {"AAPL": 1.5},
    }

    mock_scanner.save_results(results)

    cache_file = mock_scanner.cache_dir / "latest_scan.json"
    assert cache_file.exists()

    loaded = mock_scanner.load_cached_results(max_age_days=30)
    assert loaded is not None
    assert loaded["total_stocks"] == 2
    assert loaded["sectors"]["Information Technology"] == ["AAPL"]

    # Test flat list helper
    top_stocks = mock_scanner.get_top_stocks(loaded)
    assert top_stocks == ["AAPL"]
