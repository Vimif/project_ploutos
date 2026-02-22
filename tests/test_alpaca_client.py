import unittest
from unittest.mock import MagicMock, patch, ANY
import os

from trading.alpaca_client import AlpacaClient


class TestAlpacaClient(unittest.TestCase):

    @patch.dict(
        os.environ, {"ALPACA_PAPER_API_KEY": "test_key", "ALPACA_PAPER_SECRET_KEY": "test_secret"}
    )
    @patch("trading.alpaca_client.StockHistoricalDataClient")
    @patch("trading.alpaca_client.TradingClient")
    def setUp(self, mock_trading_client_cls, mock_data_client_cls):
        # mock_trading_client_cls is the class Mock.
        # Its return_value is the instance that will be returned when TradingClient() is called.
        self.mock_trading_instance = mock_trading_client_cls.return_value

        self.client = AlpacaClient(paper_trading=True)

        # In AlpacaClient.__init__, self.trading_client = TradingClient(...)
        # So self.client.trading_client SHOULD be self.mock_trading_instance

        self.mock_trading = self.client.trading_client
        self.mock_data = self.client.data_client

    def test_get_account(self):
        # Mock account response
        mock_account = MagicMock()
        mock_account.cash = "10000.0"
        mock_account.portfolio_value = "10000.0"
        mock_account.buying_power = "20000.0"
        mock_account.equity = "10000.0"
        mock_account.last_equity = "9900.0"
        mock_account.daytrade_count = 0
        mock_account.pattern_day_trader = False

        self.mock_trading.get_account.return_value = mock_account

        account = self.client.get_account()

        self.assertIsNotNone(account)
        self.assertEqual(account["cash"], 10000.0)
        self.assertEqual(account["equity"], 10000.0)

    def test_get_positions(self):
        # Mock positions
        mock_pos = MagicMock()
        mock_pos.symbol = "AAPL"
        mock_pos.qty = "10"
        mock_pos.market_value = "1500"
        mock_pos.cost_basis = "1450"
        mock_pos.unrealized_pl = "50"
        mock_pos.unrealized_plpc = "0.034"
        mock_pos.current_price = "150"
        mock_pos.avg_entry_price = "145"

        self.mock_trading.get_all_positions.return_value = [mock_pos]

        positions = self.client.get_positions()

        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]["symbol"], "AAPL")
        self.assertEqual(positions[0]["qty"], 10.0)

    def test_place_market_order_buy(self):
        # Mock order response
        mock_order = MagicMock()
        mock_order.id = "order_123"
        mock_order.symbol = "AAPL"
        mock_order.qty = "10"
        mock_order.side = "buy"
        mock_order.status = "filled"
        mock_order.filled_avg_price = "150.0"

        # Setup mocks
        self.mock_trading.submit_order.return_value = mock_order
        self.mock_trading.get_order_by_id.return_value = mock_order

        # Mock wait_for_order_fill to return True immediately
        with patch.object(self.client, "wait_for_order_fill", return_value=True):
            order = self.client.place_market_order("AAPL", 10, side="buy")

            self.assertIsNotNone(order)
            self.assertEqual(order["status"], "filled")
            self.assertEqual(order["qty"], 10.0)
            self.assertEqual(order["filled_avg_price"], 150.0)

    def test_close_position(self):
        # Setup
        mock_pos = {"qty": 10, "current_price": 150.0}

        with (
            patch.object(self.client, "get_position", return_value=mock_pos),
            patch.object(self.client, "cancel_orders_for_symbol"),
            patch.object(self.client, "wait_for_order_fill", return_value=True),
            patch("time.sleep"),
        ):  # skip sleep

            mock_close_resp = MagicMock()
            mock_close_resp.id = "close_order_123"
            self.mock_trading.close_position.return_value = mock_close_resp

            result = self.client.close_position("AAPL")

            self.assertTrue(result)
            self.mock_trading.close_position.assert_called_with("AAPL")

    def test_wait_for_order_fill_success(self):
        mock_order = MagicMock()
        mock_order.status = "filled"
        self.mock_trading.get_order_by_id.return_value = mock_order

        result = self.client.wait_for_order_fill("order_123", timeout=1)
        self.assertTrue(result)

    def test_wait_for_order_fill_timeout(self):
        mock_order = MagicMock()
        mock_order.status = "new"
        self.mock_trading.get_order_by_id.return_value = mock_order

        with patch("time.time") as mock_time, patch("time.sleep"):
            # Simulate time passing
            mock_time.side_effect = [0, 0.5, 1.1, 1.2]

            result = self.client.wait_for_order_fill("order_123", timeout=1)
            self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
