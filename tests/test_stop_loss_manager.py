from unittest.mock import MagicMock

import pytest

from trading.stop_loss_manager import StopLossManager


class TestStopLossManager:
    @pytest.fixture
    def mock_broker(self):
        broker = MagicMock()
        broker.close_position.return_value = True
        return broker

    @pytest.fixture
    def mock_metrics(self):
        metrics = MagicMock()
        return metrics

    @pytest.fixture
    def sl_manager(self, mock_broker):
        return StopLossManager(broker_client=mock_broker, stop_loss_pct=0.05, take_profit_pct=0.10)

    def test_check_no_action(self, sl_manager, mock_broker):
        positions = [
            {"symbol": "AAPL", "unrealized_plpc": 0.01, "unrealized_pl": 10},
            {"symbol": "GOOG", "unrealized_plpc": -0.01, "unrealized_pl": -10},
        ]
        sl_manager.check_all_positions(positions)
        mock_broker.close_position.assert_not_called()

    def test_execute_stop_loss(self, sl_manager, mock_broker, mock_metrics):
        positions = [{"symbol": "AAPL", "unrealized_plpc": -0.06, "unrealized_pl": -60}]
        sl_manager.check_all_positions(positions, metrics=mock_metrics)

        mock_broker.close_position.assert_called_once()
        args, kwargs = mock_broker.close_position.call_args
        assert args[0] == "AAPL"
        assert "Stop Loss" in kwargs["reason"]

        mock_metrics.record_trade.assert_called_once_with("AAPL", "SELL", 60, result="loss")

    def test_execute_take_profit(self, sl_manager, mock_broker, mock_metrics):
        positions = [{"symbol": "NVDA", "unrealized_plpc": 0.12, "unrealized_pl": 120}]
        sl_manager.check_all_positions(positions, metrics=mock_metrics)

        mock_broker.close_position.assert_called_once()
        args, kwargs = mock_broker.close_position.call_args
        assert args[0] == "NVDA"
        assert "Take Profit" in kwargs["reason"]

        mock_metrics.record_trade.assert_called_once_with("NVDA", "SELL", 120, result="win")

    def test_broker_fail_close(self, sl_manager, mock_broker, mock_metrics):
        mock_broker.close_position.return_value = False
        positions = [{"symbol": "AAPL", "unrealized_plpc": -0.10, "unrealized_pl": -100}]
        sl_manager.check_all_positions(positions, metrics=mock_metrics)

        mock_broker.close_position.assert_called_once()
        mock_metrics.record_trade.assert_not_called()
