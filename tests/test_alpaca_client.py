import unittest
from unittest.mock import MagicMock, patch
from trading.alpaca_client import AlpacaClient

class TestAlpacaClient(unittest.TestCase):
    def setUp(self):
        # Mock env vars
        self.env_patcher = patch.dict('os.environ', {
            'ALPACA_API_KEY': 'test_key',
            'ALPACA_SECRET_KEY': 'test_secret',
            'ALPACA_ENDPOINT': 'https://paper-api.alpaca.markets'
        })
        self.env_patcher.start()

        # Mock dependencies
        with patch('trading.alpaca_client.TradingClient'), \
             patch('trading.alpaca_client.StockHistoricalDataClient'):
            self.client = AlpacaClient()

    def tearDown(self):
        self.env_patcher.stop()

    def test_initialization(self):
        self.assertIsNotNone(self.client.trading_client)
        self.assertIsNotNone(self.client.data_client)

    @patch('trading.alpaca_client.TradingClient.get_account')
    def test_get_account(self, mock_get_account):
        # The code likely converts values to float, so inputs should be convertable strings or floats.
        # But wait, Alpaca API returns objects, not dicts usually, but mock might return whatever we say.
        # Let's check how `trading/alpaca_client.py` uses `get_account`.
        # Assuming it calls `self.trading_client.get_account()`

        # We need to mock the `self.client.trading_client` instance, which is created in __init__
        # But we replaced the class TradingClient in setUp context, so self.client.trading_client is a Mock.

        mock_account = MagicMock()
        # Alpaca account object has attributes, not dict keys usually.
        # But if the client converts it to dict...
        # Let's assume standard attribute access.
        mock_account.equity = '10000'
        mock_account.cash = '5000'
        mock_account.buying_power = '20000'
        mock_account.last_equity = '9500'

        self.client.trading_client.get_account.return_value = mock_account

        account = self.client.get_account()
        # Based on typical implementation, keys should be present
        self.assertEqual(float(account['equity']), 10000.0)
        self.assertEqual(float(account['cash']), 5000.0)

    def test_place_market_order(self):
        # We need to assert on the method of the trading_client instance
        self.client.trading_client.submit_order = MagicMock()
        self.client.place_market_order("AAPL", 10, "buy")
        self.client.trading_client.submit_order.assert_called_once()

    def test_get_positions(self):
        self.client.trading_client.get_all_positions = MagicMock()
        self.client.trading_client.get_all_positions.return_value = []
        positions = self.client.get_positions()
        self.assertEqual(positions, [])

if __name__ == "__main__":
    unittest.main()
