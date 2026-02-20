import unittest
from unittest.mock import MagicMock, patch
from trading.alpaca_client import AlpacaClient

class TestAlpacaClient(unittest.TestCase):
    @patch('alpaca.trading.client.TradingClient')
    @patch('alpaca.data.historical.StockHistoricalDataClient')
    def setUp(self, mock_data_client, mock_trading_client):
        self.client = AlpacaClient(paper_trading=True)
        self.client.trading_client = mock_trading_client.return_value
        self.client.data_client = mock_data_client.return_value

    def test_get_account(self):
        # Mock the account object, not a dict
        mock_account = MagicMock()
        mock_account.id = "123"
        mock_account.cash = "10000"
        mock_account.portfolio_value = "15000"
        mock_account.buying_power = "20000"
        mock_account.equity = "15000"
        mock_account.currency = "USD"

        # Configure the return value directly on the instance
        self.client.trading_client.get_account.return_value = mock_account

        account = self.client.get_account()
        # AlpacaClient returns dict from object attributes
        # Based on implementation, it might be returning the dict directly if dict() was called on object
        # or manual mapping. Let's check how it's implemented.
        # Assuming get_account calls dict(account) or similar

        # If implementation uses dict(account_obj), we need to ensure our mock supports it
        # But mocks don't support dict() by default.
        # Let's see if the client converts it.

        # If client does: return {"id": account.id, ...}
        # Then mock attributes work.

        # If failure is KeyError: 'id', then account is a dict but 'id' is missing?
        # Or account is NOT a dict but we try to access it like one?
        # "TypeError: 'Mock' object is not subscriptable" would mean it's not a dict.
        # "KeyError: 'id'" means it IS a dict, but 'id' is missing.

        # The debug output shows that AlpacaClient.get_account returns a specific dictionary
        # keys: ['cash', 'portfolio_value', 'buying_power', 'equity', 'last_equity', 'daytrade_count', 'pattern_day_trader']
        # 'id' is NOT present in the returned dictionary.

        # We should check for 'cash' or 'portfolio_value' instead.
        self.assertEqual(account.get("cash"), 10000.0)
        self.assertEqual(account.get("portfolio_value"), 15000.0)

    def test_get_position(self):
        mock_position = MagicMock()
        mock_position.symbol = "AAPL"
        mock_position.qty = "10"
        mock_position.avg_entry_price = "150.0"
        mock_position.current_price = "155.0"
        mock_position.market_value = "1550.0"
        mock_position.unrealized_pl = "50.0"
        mock_position.unrealized_plpc = "0.033"

        self.client.trading_client.get_open_position.return_value = mock_position

        position = self.client.get_position("AAPL")
        self.assertEqual(position["symbol"], "AAPL")

    def test_place_market_order(self):
        mock_order = MagicMock()
        mock_order.id = "order_1"
        mock_order.status = "filled"
        self.client.trading_client.submit_order.return_value = mock_order
        self.client.trading_client.get_order_by_id.return_value = mock_order

        with patch('time.sleep', return_value=None):  # Skip sleep
            order = self.client.place_market_order("AAPL", 10, "buy")

        self.assertEqual(order["id"], "order_1")

    def test_place_limit_order(self):
        mock_order = MagicMock()
        mock_order.id = "order_2"
        self.client.trading_client.submit_order.return_value = mock_order

        order = self.client.place_limit_order("AAPL", 10, 150.0, "buy")
        self.assertEqual(order["id"], "order_2")

    def test_close_position(self):
        self.client.trading_client.close_position.return_value = None
        result = self.client.close_position("AAPL")
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()
