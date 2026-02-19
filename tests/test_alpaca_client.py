from unittest.mock import MagicMock, patch

import pytest

from trading.alpaca_client import AlpacaClient


@pytest.fixture
def mock_alpaca_env():
    with patch.dict(
        "os.environ",
        {
            "ALPACA_PAPER_API_KEY": "test_key",
            "ALPACA_PAPER_SECRET_KEY": "test_secret",
            "ALPACA_API_KEY": "test_key",
            "ALPACA_SECRET_KEY": "test_secret",
            "ALPACA_PAPER": "True",
        },
    ):
        yield


@patch("trading.alpaca_client.TradingClient")
@patch("trading.alpaca_client.StockHistoricalDataClient")
def test_alpaca_client_init(mock_hist_client, mock_trading_client, mock_alpaca_env):
    """Test AlpacaClient initialization."""
    client = AlpacaClient()
    # Check internal trading client config if possible, or just successful init
    assert client.paper_trading is True

    mock_trading_client.assert_called_once()
    mock_hist_client.assert_called_once()


@patch("trading.alpaca_client.TradingClient")
@patch("trading.alpaca_client.StockHistoricalDataClient")
def test_get_account(mock_hist_client, mock_trading_client, mock_alpaca_env):
    """Test get_account method."""
    client = AlpacaClient()

    # Mock return value
    mock_account = MagicMock()
    # The actual implementation accesses attributes directly, not via dict()
    mock_account.id = "123"
    mock_account.cash = "10000"
    mock_account.portfolio_value = "10000"
    mock_account.buying_power = "20000"
    mock_account.equity = "10000"
    mock_account.last_equity = "9900"
    mock_account.daytrade_count = 0
    mock_account.pattern_day_trader = False

    client.trading_client.get_account.return_value = mock_account

    account = client.get_account()
    assert account["cash"] == 10000.0
    assert account["buying_power"] == 20000.0


@patch("trading.alpaca_client.TradingClient")
@patch("trading.alpaca_client.StockHistoricalDataClient")
def test_get_positions(mock_hist_client, mock_trading_client, mock_alpaca_env):
    """Test get_positions method."""
    client = AlpacaClient()

    # Mock return value
    mock_pos = MagicMock()
    mock_pos.symbol = "AAPL"
    mock_pos.qty = "10"
    mock_pos.market_value = "1500"
    mock_pos.cost_basis = "1400"
    mock_pos.unrealized_pl = "100"
    mock_pos.unrealized_plpc = "0.05"
    mock_pos.current_price = "150"
    mock_pos.avg_entry_price = "140"
    # Mock attribute access for created_at
    mock_pos.created_at = "2023-01-01T12:00:00Z"

    client.trading_client.get_all_positions.return_value = [mock_pos]

    positions = client.get_positions()
    assert len(positions) == 1
    assert positions[0]["symbol"] == "AAPL"
    assert positions[0]["qty"] == 10.0
