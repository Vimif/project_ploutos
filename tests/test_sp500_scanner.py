import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from core.sp500_scanner import SP500Scanner

class TestSP500Scanner:
    @pytest.fixture
    def scanner(self, tmp_path):
        return SP500Scanner(cache_dir=str(tmp_path), lookback_days=10)

    @patch('core.sp500_scanner.SP500Scanner.fetch_sp500_list')
    def test_scan_sectors_mocked(self, mock_fetch_sp500_list, scanner):
        # Setup mocks
        mock_df = pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT'],
            'GICS Sector': ['Information Technology', 'Information Technology']
        })
        mock_fetch_sp500_list.return_value = mock_df

        # Mock fetch return
        dates = pd.date_range(start='2020-01-01', periods=20, freq='D')
        df = pd.DataFrame({
            'Close': [100 + i for i in range(20)],
            'Volume': [1000] * 20
        }, index=dates)

        scanner.fetcher = MagicMock()
        scanner.fetcher.fetch.return_value = df

        # Run scan
        results = scanner.scan_sectors(stocks_per_sector=1, max_workers=1)

        assert isinstance(results, dict)
        assert 'sectors' in results
        assert 'total_stocks' in results

    def test_calculate_sharpe(self, scanner):
        dates = pd.date_range(start='2020-01-01', periods=252, freq='D')
        # Create a series with consistent positive returns to guarantee positive Sharpe
        # 0.5% daily return -> ~3.5 Sharpe
        prices = [100.0]
        for _ in range(251):
            prices.append(prices[-1] * 1.05)

        df = pd.DataFrame({
            'Close': prices
        }, index=dates)

        scanner.fetcher = MagicMock()
        scanner.fetcher.fetch.return_value = df

        # Override lookback to match data length perfectly
        scanner.lookback_days = 250

        sharpe = scanner._calculate_sharpe("TEST")
        assert isinstance(sharpe, float)
        assert sharpe > 0

    def test_calculate_sharpe_empty(self, scanner):
        scanner.fetcher = MagicMock()
        scanner.fetcher.fetch.return_value = None
        sharpe = scanner._calculate_sharpe("TEST")
        assert pd.isna(sharpe)

    # Patch requests where it is imported inside the method
    @patch('requests.get')
    @patch('core.sp500_scanner.pd.read_html')
    def test_fetch_sp500_list(self, mock_read_html, mock_requests, scanner):
        # Mock requests
        mock_response = MagicMock()
        mock_response.text = "<html></html>"
        mock_requests.return_value = mock_response

        # Mock table
        mock_df = pd.DataFrame({
            'Symbol': ['AAPL', 'BRK.B'],
            'Security': ['Apple', 'Berkshire'],
            'GICS Sector': ['Information Technology', 'Financials'],
            'GICS Sub-Industry': ['Tech', 'Insurance']
        })
        mock_read_html.return_value = [mock_df]

        df = scanner.fetch_sp500_list()
        assert 'AAPL' in df['Symbol'].values
        assert 'BRK-B' in df['Symbol'].values # Check replacement
        assert len(df) == 2
