import sys
from unittest.mock import MagicMock

if "core.utils" not in sys.modules:
    mock_utils = MagicMock()
    mock_logger = MagicMock()
    mock_utils.setup_logging.return_value = mock_logger
    sys.modules["core.utils"] = mock_utils

from trading.stop_loss_manager import StopLossManager


def test_stop_loss_basic():
    mock_broker = MagicMock()
    slm = StopLossManager(mock_broker, stop_loss_pct=0.05, take_profit_pct=0.15)

    positions = [
        {"symbol": "AAPL", "unrealized_plpc": -0.06, "unrealized_pl": -100.0},  # SL
        {"symbol": "MSFT", "unrealized_plpc": 0.20, "unrealized_pl": 500.0},  # TP
        {"symbol": "GOOG", "unrealized_plpc": 0.02, "unrealized_pl": 50.0},  # HOLD
    ]

    mock_metrics = MagicMock()
    slm.check_all_positions(positions, metrics=mock_metrics)

    assert mock_broker.close_position.call_count == 2
    mock_broker.close_position.assert_any_call("AAPL", reason="Stop Loss -6.0%")
    mock_broker.close_position.assert_any_call("MSFT", reason="Take Profit 20.0%")

    assert mock_metrics.record_trade.call_count == 2
