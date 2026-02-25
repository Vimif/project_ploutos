from unittest.mock import MagicMock, patch

import pytest

from trading.alpaca_client import AlpacaClient


@pytest.fixture
def mock_alpaca_deps():
    with (
        patch("trading.alpaca_client.TradingClient") as mock_trading,
        patch("trading.alpaca_client.StockHistoricalDataClient") as mock_data,
    ):

        # Setup account mock
        mock_account = MagicMock()
        mock_account.cash = "10000"
        mock_account.portfolio_value = "10000"
        mock_account.equity = "10000"
        mock_account.buying_power = "10000"
        mock_account.last_equity = "9900"
        mock_account.daytrade_count = 0
        mock_account.pattern_day_trader = False

        mock_trading.return_value.get_account.return_value = mock_account

        yield mock_trading, mock_data


def test_initialization(mock_alpaca_deps):
    with patch("os.getenv", return_value="fake_key"):
        client = AlpacaClient(paper_trading=True)
        assert client.paper_trading is True
        assert client.trading_client is not None


def test_get_account(mock_alpaca_deps):
    mock_trading, _ = mock_alpaca_deps

    with patch("os.getenv", return_value="fake_key"):
        client = AlpacaClient()
        account = client.get_account()

        assert account["cash"] == 10000.0
        assert account["equity"] == 10000.0
        mock_trading.return_value.get_account.assert_called_once()


def test_get_positions(mock_alpaca_deps):
    mock_trading, _ = mock_alpaca_deps

    # Mock positions
    pos1 = MagicMock()
    pos1.symbol = "AAPL"
    pos1.qty = "10"
    pos1.market_value = "1500"
    pos1.cost_basis = "1400"
    pos1.unrealized_pl = "100"
    pos1.unrealized_plpc = "0.07"
    pos1.current_price = "150"
    pos1.avg_entry_price = "140"

    mock_trading.return_value.get_all_positions.return_value = [pos1]

    with patch("os.getenv", return_value="fake_key"):
        client = AlpacaClient()
        positions = client.get_positions()

        assert len(positions) == 1
        assert positions[0]["symbol"] == "AAPL"
        assert positions[0]["qty"] == 10.0


def test_place_market_order(mock_alpaca_deps):
    mock_trading, _ = mock_alpaca_deps

    # Mock order response
    mock_order = MagicMock()
    mock_order.id = "123"
    mock_order.symbol = "AAPL"
    mock_order.qty = "10"
    mock_order.side = "buy"
    mock_order.status = "filled"
    mock_order.filled_avg_price = "150.0"

    mock_trading.return_value.submit_order.return_value = mock_order
    mock_trading.return_value.get_order_by_id.return_value = mock_order

    with patch("os.getenv", return_value="fake_key"):
        client = AlpacaClient()
        # Mock wait_for_order_fill to return True immediately
        with patch.object(client, "wait_for_order_fill", return_value=True):
            order = client.place_market_order("AAPL", 10, "buy")

            assert order is not None
            assert order["id"] == "123"
            assert order["status"] == "filled"


def test_close_position(mock_alpaca_deps):
    mock_trading, _ = mock_alpaca_deps

    # Mock get_position
    pos = MagicMock()
    pos.qty = "10"
    pos.current_price = "150"
    mock_trading.return_value.get_open_position.return_value = pos

    # Mock close response
    mock_close_resp = MagicMock()
    mock_close_resp.id = "order_close_123"
    mock_trading.return_value.close_position.return_value = mock_close_resp

    with patch("os.getenv", return_value="fake_key"):
        client = AlpacaClient()
        with patch.object(client, "wait_for_order_fill", return_value=True):
            with patch("time.sleep"):  # Skip sleep
                result = client.close_position("AAPL")
                assert result is True
                mock_trading.return_value.close_position.assert_called_with("AAPL")
