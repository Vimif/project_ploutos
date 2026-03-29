import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd

from core.sp500_scanner import SP500Scanner


@patch("pathlib.Path.exists")
@patch("pathlib.Path.mkdir")
@patch("requests.get")
@patch("pandas.read_html")
@patch("pandas.DataFrame.to_csv")
def test_fetch_sp500_list(mock_to_csv, mock_read_html, mock_get, mock_mkdir, mock_exists):
    scanner = SP500Scanner()
    mock_exists.return_value = False
    mock_response = MagicMock()
    mock_response.text = (
        "<html><table><tr><th>Symbol</th></tr><tr><td>NVDA</td></tr></table></html>"
    )
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    dummy_df = pd.DataFrame(
        {
            "Symbol": ["NVDA.B"],
            "Security": ["Nvidia"],
            "GICS Sector": ["Information Technology"],
            "GICS Sub-Industry": ["Semiconductors"],
        }
    )
    mock_read_html.return_value = [dummy_df]

    df = scanner.fetch_sp500_list()
    assert len(df) == 1
    assert df["Symbol"].iloc[0] == "NVDA-B"


@patch("pathlib.Path.exists")
@patch("pandas.read_csv")
@patch("os.path.getmtime")
def test_fetch_sp500_list_cached(mock_getmtime, mock_read_csv, mock_exists):
    scanner = SP500Scanner()

    def mock_exists_side_effect(*args, **kwargs):
        return True

    mock_exists.side_effect = mock_exists_side_effect
    mock_getmtime.return_value = datetime.now().timestamp()

    dummy_df = pd.DataFrame(
        {
            "Symbol": ["AAPL", "MSFT"],
            "Security": ["Apple", "Microsoft"],
            "GICS Sector": ["Information Technology", "Information Technology"],
            "GICS Sub-Industry": ["Hardware", "Software"],
        }
    )
    mock_read_csv.return_value = dummy_df

    with patch("pandas.read_html", return_value=[dummy_df]):
        df = scanner.fetch_sp500_list()
        assert len(df) == 2
        assert "AAPL" in df["Symbol"].values


def test_calculate_sharpe_direct():
    scanner = SP500Scanner()
    scanner.lookback_days = 252
    scanner.risk_free_rate = 0.0
    dates = pd.date_range(start="2020-01-01", periods=260, freq="D")
    prices = [100.0 * (1.001**i) for i in range(260)]
    dummy_df = pd.DataFrame({"Close": prices}, index=dates)

    with patch.object(scanner.fetcher, "fetch", return_value=dummy_df):
        sharpe = scanner._calculate_sharpe("AAPL")
        assert pd.notna(sharpe)
        assert sharpe > 0.0

    # Test flat data -> vol = 0 -> sharpe = 0
    prices = [100.0] * 260
    dummy_df = pd.DataFrame({"Close": prices}, index=dates)
    with patch.object(scanner.fetcher, "fetch", return_value=dummy_df):
        sharpe = scanner._calculate_sharpe("AAPL")
        assert sharpe == 0.0


@patch.object(SP500Scanner, "fetch_sp500_list")
@patch.object(SP500Scanner, "_calculate_sharpe")
def test_scan_sectors(mock_sharpe, mock_fetch_list):
    scanner = SP500Scanner()

    dummy_df = pd.DataFrame(
        {
            "Symbol": ["AAPL", "MSFT", "JNJ", "PFE"],
            "Security": ["Apple", "Microsoft", "Johnson", "Pfizer"],
            "GICS Sector": [
                "Information Technology",
                "Information Technology",
                "Health Care",
                "Health Care",
            ],
            "GICS Sub-Industry": ["Hardware", "Software", "Pharma", "Pharma"],
        }
    )
    mock_fetch_list.return_value = dummy_df

    def sharpe_side_effect(ticker):
        return {"AAPL": 2.0, "MSFT": 1.0, "JNJ": 1.5, "PFE": -0.5}.get(ticker, 0)

    mock_sharpe.side_effect = sharpe_side_effect

    results = scanner.scan_sectors(stocks_per_sector=1)
    assert "sectors" in results
    sectors = results["sectors"]
    assert "Information Technology" in sectors
    assert "AAPL" in sectors["Information Technology"]
    assert "MSFT" not in sectors["Information Technology"]
    assert "JNJ" in sectors["Health Care"]


@patch("pathlib.Path.exists")
@patch("builtins.open")
def test_load_cached_results(mock_open, mock_exists):
    scanner = SP500Scanner()
    mock_exists.return_value = True

    stat_result = MagicMock()
    stat_result.st_mtime = datetime.now().timestamp()

    with patch("pathlib.Path.stat", return_value=stat_result):
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = json.dumps(
            {"sectors": {"Information Technology": ["AAPL"]}}
        )
        mock_open.return_value = mock_file

        res = scanner.load_cached_results(max_age_days=30)
        assert res["sectors"] == {"Information Technology": ["AAPL"]}

    stat_result_old = MagicMock()
    stat_result_old.st_mtime = (datetime.now() - timedelta(days=40)).timestamp()
    with patch("pathlib.Path.stat", return_value=stat_result_old):
        res = scanner.load_cached_results(max_age_days=30)
        assert res is None
