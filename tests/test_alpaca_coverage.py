import unittest
from unittest.mock import MagicMock, patch, ANY
import os
import sys
from datetime import datetime

# Mock environment variables
os.environ['ALPACA_PAPER_API_KEY'] = 'test_key'
os.environ['ALPACA_PAPER_SECRET_KEY'] = 'test_secret'

# Mock alpaca modules before import
sys.modules['alpaca.trading.client'] = MagicMock()
sys.modules['alpaca.trading.requests'] = MagicMock()
sys.modules['alpaca.trading.enums'] = MagicMock()
sys.modules['alpaca.data.historical'] = MagicMock()
sys.modules['alpaca.data.requests'] = MagicMock()
sys.modules['alpaca.data.timeframe'] = MagicMock()

# Mock dotenv
sys.modules['dotenv'] = MagicMock()

# Now import
from trading.alpaca_client import AlpacaClient

class TestAlpacaClient(unittest.TestCase):
    def setUp(self):
        self.client = AlpacaClient(paper_trading=True)
        self.client.trading_client = MagicMock()
        self.client.data_client = MagicMock()

    def test_initialization(self):
        self.assertTrue(self.client.paper_trading)
        # Check logs dir created
        self.assertTrue(os.path.exists('logs/trades'))

    def test_get_account(self):
        mock_account = MagicMock()
        mock_account.cash = '10000.0'
        mock_account.portfolio_value = '10000.0'
        mock_account.buying_power = '20000.0'
        mock_account.equity = '10000.0'
        mock_account.last_equity = '9900.0'
        mock_account.daytrade_count = 0
        mock_account.pattern_day_trader = False

        self.client.trading_client.get_account.return_value = mock_account

        account = self.client.get_account()
        self.assertEqual(account['cash'], 10000.0)
        self.assertEqual(account['equity'], 10000.0)

    def test_get_positions(self):
        mock_pos = MagicMock()
        mock_pos.symbol = 'AAPL'
        mock_pos.qty = '10'
        mock_pos.market_value = '1500.0'
        mock_pos.cost_basis = '1400.0'
        mock_pos.unrealized_pl = '100.0'
        mock_pos.unrealized_plpc = '0.07'
        mock_pos.current_price = '150.0'
        mock_pos.avg_entry_price = '140.0'

        self.client.trading_client.get_all_positions.return_value = [mock_pos]

        positions = self.client.get_positions()
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]['symbol'], 'AAPL')
        self.assertEqual(positions[0]['qty'], 10.0)

    def test_get_position(self):
        mock_pos = MagicMock()
        mock_pos.symbol = 'AAPL'
        mock_pos.qty = '10'
        mock_pos.market_value = '1500.0'
        mock_pos.unrealized_pl = '100.0'
        mock_pos.unrealized_plpc = '0.07'
        mock_pos.current_price = '150.0'
        mock_pos.avg_entry_price = '140.0'

        self.client.trading_client.get_open_position.return_value = mock_pos

        pos = self.client.get_position('AAPL')
        self.assertEqual(pos['symbol'], 'AAPL')

    def test_get_current_price(self):
        mock_quote = MagicMock()
        mock_quote.bid_price = '149.0'
        mock_quote.ask_price = '151.0'

        self.client.data_client.get_stock_latest_quote.return_value = {'AAPL': mock_quote}

        price = self.client.get_current_price('AAPL')
        self.assertEqual(price, 150.0)

    def test_cancel_orders(self):
        mock_order = {'id': '123', 'symbol': 'AAPL'}
        # self.client.get_orders = MagicMock(return_value=[mock_order])
        # Need to patch get_orders method of self.client or trading_client?
        # get_orders on AlpacaClient calls trading_client.get_orders

        # We mock trading_client.get_orders response object
        mock_alpaca_order = MagicMock()
        mock_alpaca_order.id = '123'
        mock_alpaca_order.symbol = 'AAPL'
        mock_alpaca_order.qty = '1'
        mock_alpaca_order.side = MagicMock()
        mock_alpaca_order.side.value = 'buy'
        mock_alpaca_order.status = MagicMock()
        mock_alpaca_order.status.value = 'new'
        mock_alpaca_order.order_type = MagicMock()
        mock_alpaca_order.order_type.value = 'market'

        self.client.trading_client.get_orders.return_value = [mock_alpaca_order]

        count = self.client.cancel_orders_for_symbol('AAPL')
        self.assertEqual(count, 1)
        self.client.trading_client.cancel_order_by_id.assert_called_with('123')

    @patch('time.sleep')
    @patch('time.time')
    def test_place_market_order(self, mock_time, mock_sleep):
        # Mock successful order submission
        mock_order = MagicMock()
        mock_order.id = 'order_123'
        mock_order.symbol = 'AAPL'
        mock_order.qty = '1'
        mock_order.side = 'buy'
        mock_order.status = 'filled'
        mock_order.filled_avg_price = '150.0'

        self.client.trading_client.submit_order.return_value = mock_order
        self.client.trading_client.get_order_by_id.return_value = mock_order

        # Mock time to avoid loop
        mock_time.side_effect = [0, 1, 2, 3]

        result = self.client.place_market_order('AAPL', 1, 'buy')

        self.assertIsNotNone(result)
        self.assertEqual(result['id'], 'order_123')
        self.assertEqual(result['status'], 'filled')

    def test_place_limit_order(self):
        mock_order = MagicMock()
        mock_order.id = 'order_lim_123'
        mock_order.symbol = 'AAPL'
        mock_order.qty = '1'
        mock_order.limit_price = '140.0'
        mock_order.status = 'new'

        self.client.trading_client.submit_order.return_value = mock_order

        result = self.client.place_limit_order('AAPL', 1, 140.0, 'buy')
        self.assertEqual(result['id'], 'order_lim_123')

    @patch('time.sleep')
    @patch('time.time')
    def test_close_position(self, mock_time, mock_sleep):
        # Mock existing position
        mock_pos = MagicMock()
        mock_pos.qty = 10
        mock_pos.current_price = 150.0

        # Mock get_position
        with patch.object(self.client, 'get_position', return_value={'qty': 10, 'current_price': 150.0}):
            # Mock get_orders (empty)
            self.client.trading_client.get_orders.return_value = []

            # Mock close_position response
            mock_resp = MagicMock()
            mock_resp.id = 'close_123'
            self.client.trading_client.close_position.return_value = mock_resp

            # Mock order check
            mock_order = MagicMock()
            mock_order.status = 'filled'
            self.client.trading_client.get_order_by_id.return_value = mock_order

            mock_time.side_effect = [0, 1, 2, 3]

            success = self.client.close_position('AAPL')
            self.assertTrue(success)

    def test_close_all_positions(self):
        self.client.trading_client.close_all_positions.return_value = []
        success = self.client.close_all_positions()
        self.assertTrue(success)
        self.client.trading_client.close_all_positions.assert_called_once()

if __name__ == '__main__':
    unittest.main()
