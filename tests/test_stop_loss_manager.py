
import pytest
from unittest.mock import MagicMock
from trading.stop_loss_manager import StopLossManager

# ============================================================================
# StopLossManager Tests
# ============================================================================

@pytest.fixture
def mock_broker():
    return MagicMock()

@pytest.fixture
def mock_metrics():
    return MagicMock()

@pytest.fixture
def manager(mock_broker):
    return StopLossManager(mock_broker, stop_loss_pct=0.05, take_profit_pct=0.10)

def test_check_all_positions_no_trigger(manager, mock_broker):
    positions = [
        {"symbol": "AAPL", "unrealized_plpc": 0.02, "unrealized_pl": 100},
        {"symbol": "MSFT", "unrealized_plpc": -0.01, "unrealized_pl": -50}
    ]
    manager.check_all_positions(positions)
    mock_broker.close_position.assert_not_called()

def test_check_all_positions_trigger_sl(manager, mock_broker, mock_metrics):
    positions = [
        {"symbol": "AAPL", "unrealized_plpc": -0.06, "unrealized_pl": -300}
    ]
    mock_broker.close_position.return_value = True

    manager.check_all_positions(positions, metrics=mock_metrics)

    mock_broker.close_position.assert_called_once()
    args, kwargs = mock_broker.close_position.call_args
    assert args[0] == "AAPL"
    assert "Stop Loss" in kwargs["reason"]

    mock_metrics.record_trade.assert_called_once_with("AAPL", "SELL", 300, result="loss")

def test_check_all_positions_trigger_tp(manager, mock_broker, mock_metrics):
    positions = [
        {"symbol": "NVDA", "unrealized_plpc": 0.15, "unrealized_pl": 1500}
    ]
    mock_broker.close_position.return_value = True

    manager.check_all_positions(positions, metrics=mock_metrics)

    mock_broker.close_position.assert_called_once()
    args, kwargs = mock_broker.close_position.call_args
    assert args[0] == "NVDA"
    assert "Take Profit" in kwargs["reason"]

    mock_metrics.record_trade.assert_called_once_with("NVDA", "SELL", 1500, result="win")

def test_check_all_positions_close_fail(manager, mock_broker, mock_metrics):
    positions = [
        {"symbol": "TSLA", "unrealized_plpc": 0.20, "unrealized_pl": 2000}
    ]
    mock_broker.close_position.return_value = False # Simulate failure

    manager.check_all_positions(positions, metrics=mock_metrics)

    mock_broker.close_position.assert_called_once()
    mock_metrics.record_trade.assert_not_called()
