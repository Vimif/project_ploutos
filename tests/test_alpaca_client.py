import unittest
from unittest.mock import MagicMock, patch

from trading.alpaca_client import AlpacaClient


class TestAlpacaClient(unittest.TestCase):
    def setUp(self):
        # Patch environment variables
        self.env_patcher = patch.dict(
            "os.environ",
            {"ALPACA_PAPER_API_KEY": "fake_key", "ALPACA_PAPER_SECRET_KEY": "fake_secret"},
        )
        self.env_patcher.start()

        # Patch Alpaca API clients
        self.trading_client_patcher = patch("trading.alpaca_client.TradingClient")
        self.data_client_patcher = patch("trading.alpaca_client.StockHistoricalDataClient")

        self.MockTradingClient = self.trading_client_patcher.start()
        self.MockDataClient = self.data_client_patcher.start()

        self.client = AlpacaClient(paper_trading=True)

    def tearDown(self):
        self.env_patcher.stop()
        self.trading_client_patcher.stop()
        self.data_client_patcher.stop()

    def test_init(self):
        self.assertIsNotNone(self.client.trading_client)
        self.assertIsNotNone(self.client.data_client)

    def test_get_account(self):
        mock_account = MagicMock()
        mock_account.cash = "10000"
        mock_account.portfolio_value = "10000"
        mock_account.buying_power = "20000"
        mock_account.equity = "10000"
        mock_account.last_equity = "9000"
        mock_account.daytrade_count = 0
        mock_account.pattern_day_trader = False

        self.client.trading_client.get_account.return_value = mock_account

        account = self.client.get_account()
        self.assertEqual(account["cash"], 10000.0)
        self.assertEqual(account["equity"], 10000.0)

    def test_get_positions(self):
        mock_pos = MagicMock()
        mock_pos.symbol = "AAPL"
        mock_pos.qty = "10"
        mock_pos.market_value = "1500"
        mock_pos.cost_basis = "1400"
        mock_pos.unrealized_pl = "100"
        mock_pos.unrealized_plpc = "0.07"
        mock_pos.current_price = "150"
        mock_pos.avg_entry_price = "140"

        self.client.trading_client.get_all_positions.return_value = [mock_pos]

        positions = self.client.get_positions()
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]["symbol"], "AAPL")
        self.assertEqual(positions[0]["qty"], 10.0)

    def test_close_position(self):
        # Mock get_position
        self.client.get_position = MagicMock(return_value={"qty": 10, "current_price": 150})
        self.client.cancel_orders_for_symbol = MagicMock()
        self.client.wait_for_order_fill = MagicMock(return_value=True)

        mock_response = MagicMock()
        mock_response.id = "order_123"
        self.client.trading_client.close_position.return_value = mock_response

        with patch("trading.alpaca_client.log_trade_to_json"):
            result = self.client.close_position("AAPL")
            self.assertTrue(result)
            self.client.trading_client.close_position.assert_called_with("AAPL")

    def test_place_market_order(self):
        self.client.wait_for_order_fill = MagicMock(return_value=True)
        mock_order = MagicMock()
        mock_order.id = "order_123"
        mock_order.symbol = "AAPL"
        mock_order.qty = "10"
        mock_order.side = "buy"
        mock_order.status = "filled"
        mock_order.filled_avg_price = "150.0"

        self.client.trading_client.submit_order.return_value = mock_order
        self.client.trading_client.get_order_by_id.return_value = mock_order

        with patch("trading.alpaca_client.log_trade_to_json"):
            order = self.client.place_market_order("AAPL", 10, "buy")
            self.assertIsNotNone(order)
            self.assertEqual(order["symbol"], "AAPL")
            self.assertEqual(order["status"], "filled")

    def test_get_current_price(self):
        mock_quote = MagicMock()
        mock_quote.bid_price = "149.0"
        mock_quote.ask_price = "151.0"
        self.client.data_client.get_stock_latest_quote.return_value = {"AAPL": mock_quote}

        price = self.client.get_current_price("AAPL")
        self.assertEqual(price, 150.0)

    def test_get_orders(self):
        mock_order = MagicMock()
        mock_order.id = "1"
        mock_order.symbol = "AAPL"
        mock_order.qty = "10"
        self.client.trading_client.get_orders.return_value = [mock_order]

        orders = self.client.get_orders()
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]["symbol"], "AAPL")
