import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
from unittest.mock import MagicMock, patch
import os

# Mock before importing core.data_fetcher
mock_polygon = MagicMock()
mock_alpaca = MagicMock()

with patch.dict('sys.modules', {
    'polygon': mock_polygon,
    'alpaca': mock_alpaca,
    'alpaca.data': mock_alpaca.data,
    'alpaca.data.historical': mock_alpaca.data.historical,
    'alpaca.data.requests': mock_alpaca.data.requests,
    'alpaca.data.timeframe': mock_alpaca.data.timeframe
}):
    from core.data_fetcher import UniversalDataFetcher

def test_logging():
    print("🧪 Starting logging test...")

    # Setup mocks to raise exceptions
    mock_polygon.RESTClient.side_effect = Exception("Polygon forced failure")
    mock_alpaca.data.historical.StockHistoricalDataClient.side_effect = Exception("Alpaca forced failure")

    # Set fake env keys
    os.environ['POLYGON_API_KEY'] = 'fake'
    os.environ['ALPACA_API_KEY'] = 'fake'
    os.environ['ALPACA_SECRET_KEY'] = 'fake'

    with patch('core.data_fetcher.logger') as mock_logger:
        print("🔄 Initializing UniversalDataFetcher...")
        fetcher = UniversalDataFetcher()

        # Check for exception logs
        exception_calls = [call[0][0] for call in mock_logger.exception.call_args_list]
        print(f"  Exception logs: {exception_calls}")

        alpaca_logged = any("Alpaca échec" in msg for msg in exception_calls)
        polygon_logged = any("Polygon échec" in msg for msg in exception_calls)

        if alpaca_logged and polygon_logged:
            print("✅ Initialization exceptions logged correctly")
        else:
            if not alpaca_logged: print("❌ Alpaca exception NOT logged")
            if not polygon_logged: print("❌ Polygon exception NOT logged")
            sys.exit(1)

        # Test fetch fallback and its logging
        print("\n🔄 Testing fetch fallback...")
        mock_logger.reset_mock()

        # Mock yfinance to succeed
        with patch('yfinance.download') as mock_yf:
            import pandas as pd
            import numpy as np
            mock_df = pd.DataFrame(
                np.random.randn(200, 5),
                columns=['Open', 'High', 'Low', 'Close', 'Volume'],
                index=pd.date_range('2023-01-01', periods=200)
            )
            mock_yf.return_value = mock_df

            df = fetcher.fetch("AAPL", interval='1d')

            if df is not None:
                print(f"✅ Fetch successful: {len(df)} rows")
            else:
                print("❌ Fetch returned None")
                sys.exit(1)

            # Check info logs in fetch
            info_logs = [call[0][0] for call in mock_logger.info.call_args_list]
            print(f"  Info logs: {info_logs}")

            fetch_started = any("Fetch AAPL" in msg for msg in info_logs)
            source_success = any("yfinance : 200 bougies" in msg for msg in info_logs)

            if fetch_started and source_success:
                print("✅ Fetch logs generated correctly")
            else:
                if not fetch_started: print("❌ Fetch start log missing")
                if not source_success: print("❌ Source success log missing")
                sys.exit(1)

    print("\n🎉 All logging tests passed!")

if __name__ == "__main__":
    try:
        test_logging()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
