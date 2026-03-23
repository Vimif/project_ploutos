import pytest
from unittest.mock import patch, MagicMock
from database import db

def test_interval_type_safety_get_trade_history():
    """
    Vérifie que les injections de chaînes malveillantes via les paramètres temporels
    échouent de manière sécurisée (ValueError due au cast entier au lieu d'une erreur SQL).
    L'exception ValueError est attrapée par la fonction et une liste vide est renvoyée.
    """
    assert db.get_trade_history(days="30; DROP TABLE trades; --") == []

def test_interval_type_safety_get_position_history():
    assert db.get_position_history(symbol="AAPL", days="10 OR 1=1") == []

def test_interval_type_safety_get_daily_summary():
    assert db.get_daily_summary(days="abc") == []

def test_interval_type_safety_get_prediction_history():
    assert db.get_prediction_history(symbol="AAPL", days="999 UNION ALL SELECT * FROM users") == []

def test_limit_type_safety_get_top_symbols():
    """
    Vérifie que les injections de chaînes malveillantes via les paramètres de limite
    échouent de manière sécurisée au niveau de l'application et renvoient une liste vide.
    """
    assert db.get_top_symbols(days=30, limit="10 OFFSET 5") == []

def test_interval_type_safety_get_portfolio_evolution():
    assert db.get_portfolio_evolution(days="1 year") == []

def test_interval_type_safety_get_win_loss_ratio():
    # Retourne {'wins': 0, 'losses': 0, 'total': 0, 'win_rate': 0} par défaut
    assert db.get_win_loss_ratio(days="30; SELECT pg_sleep(10);") == {'wins': 0, 'losses': 0, 'total': 0, 'win_rate': 0}

def test_interval_type_safety_get_trade_statistics():
    # Retourne {} par défaut
    assert db.get_trade_statistics(days="30 days") == {}
