from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from core.sp500_scanner import SP500Scanner


@pytest.fixture
def scanner():
    return SP500Scanner(lookback_days=10)

def test_sp500_scanner_init(scanner):
    assert scanner.lookback_days == 10
    assert str(scanner.cache_dir) == 'data/sp500_cache'

@patch('core.sp500_scanner.pd.read_html')
def test_sp500_scanner_fetch_list(mock_read_html, scanner):
    mock_df = pd.DataFrame({
        'Symbol': ['AAPL', 'BRK.B', 'BF.B'],
        'Security': ['Apple', 'Berkshire', 'Brown Forman'],
        'GICS Sector': ['Tech', 'Financials', 'Consumer'],
        'GICS Sub-Industry': ['Tech Hardware', 'Insurance', 'Beverages']
    })
    mock_read_html.return_value = [mock_df]

    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.text = '<html></html>'
        mock_get.return_value = mock_response

        df = scanner.fetch_sp500_list()

    assert len(df) == 3
    # BRK.B -> BRK-B mapping should occur
    assert df.iloc[1]['Symbol'] == 'BRK-B'
    assert df.iloc[2]['Symbol'] == 'BF-B'
