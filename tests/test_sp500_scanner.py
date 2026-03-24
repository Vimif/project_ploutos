import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import pandas as pd
import numpy as np
import json
import time

from core.sp500_scanner import SP500Scanner

@pytest.fixture
def mock_cache_dir(tmp_path):
    cache = tmp_path / "sp500_cache"
    return str(cache)

def test_sp500_scanner_init(mock_cache_dir):
    scanner = SP500Scanner(cache_dir=mock_cache_dir, lookback_days=100)
    assert scanner.lookback_days == 100
    assert scanner.cache_dir.exists()

@patch('requests.get')
@patch('pandas.read_html')
def test_fetch_sp500_list(mock_read_html, mock_get, mock_cache_dir):
    mock_resp = MagicMock()
    mock_resp.text = "mock html"
    mock_get.return_value = mock_resp

    mock_df = pd.DataFrame({
        'Symbol': ['AAPL', 'BRK.B', 'JNJ'],
        'Security': ['Apple Inc.', 'Berkshire', 'Johnson & Johnson'],
        'GICS Sector': ['Information Technology', 'Financials', 'Health Care'],
        'GICS Sub-Industry': ['Tech', 'Fin', 'Pharma']
    })
    mock_read_html.return_value = [mock_df]

    scanner = SP500Scanner(cache_dir=mock_cache_dir)
    constituents = scanner.fetch_sp500_list()

    assert len(constituents) == 3
    assert 'AAPL' in constituents['Symbol'].values
    assert 'BRK-B' in constituents['Symbol'].values

@patch('core.sp500_scanner.UniversalDataFetcher')
def test_calculate_sharpe(mock_fetcher_cls, mock_cache_dir):
    mock_fetcher = mock_fetcher_cls.return_value
    scanner = SP500Scanner(cache_dir=mock_cache_dir, lookback_days=100)
    scanner.fetcher = mock_fetcher

    dates = pd.date_range(start="2023-01-01", periods=100, freq="1d")
    mock_df = pd.DataFrame({
        'Close': np.linspace(100, 200, 100)
    }, index=dates)
    mock_fetcher.fetch.return_value = mock_df

    sharpe = scanner._calculate_sharpe('AAPL')
    assert not np.isnan(sharpe)

    mock_fetcher.fetch.return_value = pd.DataFrame({'Close': [100]*10})
    sharpe = scanner._calculate_sharpe('AAPL')
    assert np.isnan(sharpe)

    mock_fetcher.fetch.return_value = None
    sharpe = scanner._calculate_sharpe('AAPL')
    assert np.isnan(sharpe)

    mock_fetcher.fetch.side_effect = Exception("Fetch failed")
    sharpe = scanner._calculate_sharpe('AAPL')
    assert np.isnan(sharpe)

@patch('core.sp500_scanner.SP500Scanner.fetch_sp500_list')
@patch('core.sp500_scanner.SP500Scanner._calculate_sharpe')
def test_scan_sectors(mock_sharpe, mock_fetch_list, mock_cache_dir):
    scanner = SP500Scanner(cache_dir=mock_cache_dir, lookback_days=100)

    mock_df = pd.DataFrame({
        'Symbol': ['AAPL', 'MSFT', 'JNJ', 'PFE', 'BAD'],
        'Security': ['Apple', 'Microsoft', 'Johnson', 'Pfizer', 'Bad'],
        'GICS Sector': ['Information Technology', 'Information Technology', 'Health Care', 'Health Care', 'Energy'],
        'GICS Sub-Industry': ['Tech', 'Soft', 'Pharma', 'Pharma', 'Oil']
    })
    mock_fetch_list.return_value = mock_df

    def mock_calc(symbol):
        if symbol == 'BAD': raise Exception("calc failed")
        sharpes = {'AAPL': 2.0, 'MSFT': 1.0, 'JNJ': 1.5, 'PFE': 0.5}
        return sharpes.get(symbol, np.nan)

    mock_sharpe.side_effect = mock_calc

    results = scanner.scan_sectors(stocks_per_sector=1, max_workers=2)

    assert results['total_stocks'] == 2
    assert len(results['sectors']) == 2
    assert results['sectors']['Information Technology'] == ['AAPL']
    assert results['sectors']['Health Care'] == ['JNJ']
    assert 'Energy' not in results['sectors']

def test_helpers(mock_cache_dir):
    scanner = SP500Scanner(cache_dir=mock_cache_dir, lookback_days=100)
    res = {
        'total_stocks': 2,
        'sectors': {'Tech': ['AAPL'], 'Health': ['JNJ']}
    }

    # get_top_stocks
    top = scanner.get_top_stocks(res)
    assert top == ['AAPL', 'JNJ']

    with patch.object(scanner, 'scan_sectors') as mock_scan:
        mock_scan.return_value = res
        assert scanner.get_top_stocks() == ['AAPL', 'JNJ']

    # save / load results
    scanner.save_results(res)
    loaded = scanner.load_cached_results(max_age_days=1)
    assert loaded['total_stocks'] == 2

    # max_age testing
    time.sleep(0.01)
    loaded_old = scanner.load_cached_results(max_age_days=-1)
    assert loaded_old is None

def test_load_cached_results_no_file(mock_cache_dir):
    scanner = SP500Scanner(cache_dir=mock_cache_dir)
    assert scanner.load_cached_results() is None
