import os
from unittest.mock import MagicMock, patch

import pytest

from trading.alpaca_client import AlpacaClient


@pytest.fixture
def mock_alpaca_env():
    with patch.dict(
        os.environ,
        {
            "ALPACA_PAPER_API_KEY": "fake_key",
            "ALPACA_PAPER_SECRET_KEY": "fake_secret",
            "ALPACA_LIVE_API_KEY": "live_key",
            "ALPACA_LIVE_SECRET_KEY": "live_secret",
        },
    ):
        yield


@pytest.fixture
def mock_trading_client():
    with patch("trading.alpaca_client.TradingClient") as mock:
        yield mock


@pytest.fixture
def mock_data_client():
    with patch("trading.alpaca_client.StockHistoricalDataClient") as mock:
        yield mock


def test_initialization(mock_alpaca_env, mock_trading_client, mock_data_client):
    client = AlpacaClient(paper_trading=True)
    assert client.paper_trading is True
    mock_trading_client.assert_called_once()


def test_initialization_live(mock_alpaca_env, mock_trading_client, mock_data_client):
    client = AlpacaClient(paper_trading=False)
    assert client.paper_trading is False


def test_get_account(mock_alpaca_env, mock_trading_client, mock_data_client):
    client = AlpacaClient()
    mock_account = MagicMock()
    mock_account.cash = "10000"
    mock_account.portfolio_value = "10000"
    mock_account.buying_power = "20000"
    mock_account.equity = "10000"
    mock_account.last_equity = "9000"
    mock_account.daytrade_count = 3
    mock_account.pattern_day_trader = False

    client.trading_client.get_account.return_value = mock_account

    account = client.get_account()
    assert account["cash"] == 10000.0
    assert account["daytrade_count"] == 3


def test_get_positions(mock_alpaca_env, mock_trading_client, mock_data_client):
    client = AlpacaClient()
    mock_pos = MagicMock()
    mock_pos.symbol = "AAPL"
    mock_pos.qty = "10"
    mock_pos.market_value = "1500"
    mock_pos.cost_basis = "1400"
    mock_pos.unrealized_pl = "100"
    mock_pos.unrealized_plpc = "0.07"
    mock_pos.current_price = "150"
    mock_pos.avg_entry_price = "140"

    client.trading_client.get_all_positions.return_value = [mock_pos]

    positions = client.get_positions()
    assert len(positions) == 1
    assert positions[0]["symbol"] == "AAPL"
    assert positions[0]["qty"] == 10.0


def test_place_market_order(mock_alpaca_env, mock_trading_client, mock_data_client):
    client = AlpacaClient()
    mock_order = MagicMock()
    mock_order.id = "123"
    mock_order.symbol = "AAPL"
    mock_order.qty = "10"
    mock_order.side = "buy"
    mock_order.status = "filled"
    mock_order.filled_avg_price = "150"

    client.trading_client.submit_order.return_value = mock_order
    # Mock wait_for_order_fill to return True immediately
    client.wait_for_order_fill = MagicMock(return_value=True)
    client.trading_client.get_order_by_id.return_value = mock_order

    order = client.place_market_order("AAPL", 10, "buy")
    assert order["id"] == "123"
    assert order["status"] == "filled"


def test_close_position(mock_alpaca_env, mock_trading_client, mock_data_client):
    client = AlpacaClient()

    # Mock existing position
    mock_pos = MagicMock()
    mock_pos.qty = "10"
    mock_pos.current_price = "150"
    client.get_position = MagicMock(return_value={"qty": 10, "current_price": 150})

    # Mock close response
    mock_response = MagicMock()
    mock_response.id = "456"
    client.trading_client.close_position.return_value = mock_response
    client.wait_for_order_fill = MagicMock(return_value=True)

    result = client.close_position("AAPL")
    assert result is True


def test_get_orders(mock_alpaca_env, mock_trading_client, mock_data_client):
    client = AlpacaClient()
    mock_order = MagicMock()
    mock_order.id = "123"
    mock_order.symbol = "AAPL"
    mock_order.qty = "10"
    mock_order.side = "buy"
    mock_order.status = "filled"

    client.trading_client.get_orders.return_value = [mock_order]

    orders = client.get_orders()
    assert len(orders) == 1
    assert orders[0]["symbol"] == "AAPL"


def test_cancel_order(mock_alpaca_env, mock_trading_client, mock_data_client):
    client = AlpacaClient()
    result = client.cancel_order("123")
    assert result is True
    client.trading_client.cancel_order_by_id.assert_called_with("123")


def test_cancel_orders_for_symbol(mock_alpaca_env, mock_trading_client, mock_data_client):
    client = AlpacaClient()
    mock_obj = MagicMock()
    mock_obj.id = "123"
    mock_obj.symbol = "AAPL"

    client.trading_client.get_orders.return_value = [mock_obj]

    count = client.cancel_orders_for_symbol("AAPL")
    assert count == 1
    client.trading_client.cancel_order_by_id.assert_called_with("123")
