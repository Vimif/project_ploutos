import pytest
from unittest.mock import patch, MagicMock
from core.sp500_scanner import SP500Scanner
from pathlib import Path

def test_save_results():
    scanner = SP500Scanner()
    results = {'sectors': {}}
    with patch('builtins.open') as mock_open:
        with patch('json.dump') as mock_json:
            scanner.save_results(results, filepath='test_output.json')
            mock_open.assert_called_once_with(Path('test_output.json'), 'w', encoding='utf-8')
            mock_json.assert_called_once()

        with patch('json.dump') as mock_json:
            scanner.save_results(results)
            # Default saves to cache dir
            assert mock_open.call_count == 2

def test_get_top_stocks_from_scan():
    scanner = SP500Scanner()
    scan_results = {
        'sectors': {'Tech': ['AAPL', 'MSFT'], 'Health': ['JNJ']},
        'sharpe_ratios': {'AAPL': 2.0, 'MSFT': 1.0, 'JNJ': 1.5}
    }
    top = scanner.get_top_stocks(scan_results=scan_results)
    assert len(top) == 3
    assert 'AAPL' in top
    assert 'JNJ' in top

    # No scan results
    with patch.object(scanner, 'scan_sectors') as mock_scan:
        mock_scan.return_value = scan_results
        top = scanner.get_top_stocks()
        assert len(top) == 3
