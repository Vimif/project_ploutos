from datetime import date
from unittest.mock import MagicMock, patch

import pytest

import database.db as db


@pytest.fixture
def mock_conn():
    with patch('database.db.get_connection') as mock_get_conn:
        conn = MagicMock()
        mock_get_conn.return_value.__enter__.return_value = conn
        yield conn

def test_init_database(mock_conn):
    db.init_database()
    mock_conn.cursor().execute.assert_called()

def test_log_trade(mock_conn):
    mock_conn.cursor().fetchone.return_value = [1]
    trade_id = db.log_trade('AAPL', 'BUY', 10.0, 150.0, 1500.0)
    assert trade_id == 1
    mock_conn.cursor().execute.assert_called()

def test_get_trade_history(mock_conn):
    mock_conn.cursor().fetchall.return_value = [{'id': 1, 'symbol': 'AAPL'}]
    history = db.get_trade_history(days=30, symbol='AAPL')
    assert len(history) == 1
    assert history[0]['symbol'] == 'AAPL'

    # Test without symbol
    history_all = db.get_trade_history(days=30)
    assert len(history_all) == 1

def test_log_position(mock_conn):
    mock_conn.cursor().fetchone.return_value = [1]
    pos_id = db.log_position('AAPL', 10.0, 150.0, 155.0, 1550.0, 50.0, 0.03)
    assert pos_id == 1

def test_get_position_history(mock_conn):
    mock_conn.cursor().fetchall.return_value = [{'id': 1, 'symbol': 'AAPL'}]
    history = db.get_position_history('AAPL', days=30)
    assert len(history) == 1

def test_log_all_positions(mock_conn):
    positions = [{'symbol': 'AAPL', 'qty': 10, 'avg_entry_price': 150, 'current_price': 155, 'market_value': 1550, 'unrealized_pl': 50, 'unrealized_plpc': 0.03}]
    db.log_all_positions(positions)
    assert mock_conn.cursor().execute.call_count == 1

def test_save_daily_summary(mock_conn):
    db.save_daily_summary(date.today(), 100000, 50000, 50000, 500, 5, 10)
    mock_conn.cursor().execute.assert_called()

def test_get_daily_summary(mock_conn):
    mock_conn.cursor().fetchall.return_value = [{'date': date.today(), 'portfolio_value': 100000}]
    summary = db.get_daily_summary(days=30)
    assert len(summary) == 1

def test_log_prediction(mock_conn):
    mock_conn.cursor().fetchone.return_value = [1]
    pred_id = db.log_prediction('AAPL', 'Technology', 1, 0.8, 'BUY', {'f1': 1})
    assert pred_id == 1

def test_get_prediction_history(mock_conn):
    mock_conn.cursor().fetchall.return_value = [{'id': 1, 'symbol': 'AAPL'}]
    history = db.get_prediction_history(symbol='AAPL', days=7)
    assert len(history) == 1

    # Test without symbol
    history_all = db.get_prediction_history(days=7)
    assert len(history_all) == 1

def test_get_trade_statistics(mock_conn):
    mock_conn.cursor().fetchone.return_value = {'total_trades': 10}
    stats = db.get_trade_statistics(days=30)
    assert stats['total_trades'] == 10

def test_get_top_symbols(mock_conn):
    mock_conn.cursor().fetchall.return_value = [{'symbol': 'AAPL', 'trade_count': 5}]
    symbols = db.get_top_symbols(days=30, limit=10)
    assert len(symbols) == 1
    assert symbols[0]['symbol'] == 'AAPL'

def test_get_portfolio_evolution(mock_conn):
    mock_conn.cursor().fetchall.return_value = [{'date': date.today(), 'portfolio_value': 100000}]
    evolution = db.get_portfolio_evolution(days=30)
    assert len(evolution) == 1

def test_get_win_loss_ratio(mock_conn):
    mock_conn.cursor().fetchone.return_value = {'wins': 6, 'losses': 4}
    ratio = db.get_win_loss_ratio(days=30)
    assert ratio['wins'] == 6
    assert ratio['losses'] == 4
    assert ratio['total'] == 10
    assert ratio['win_rate'] == 60.0

def test_get_win_loss_ratio_empty(mock_conn):
    mock_conn.cursor().fetchone.return_value = None
    ratio = db.get_win_loss_ratio(days=30)
    assert ratio['wins'] == 0
    assert ratio['losses'] == 0
    assert ratio['total'] == 0
    assert ratio['win_rate'] == 0
