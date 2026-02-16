# database/__init__.py
"""Package de gestion de la base de données"""

from .db import (
    get_daily_summary,
    get_portfolio_evolution,
    get_position_history,
    get_trade_history,
    get_trade_statistics,
    init_database,
    log_position,
    log_prediction,
    log_trade,
    save_daily_summary,
)

__all__ = [
    "init_database",
    "log_trade",
    "log_position",
    "log_prediction",
    "get_trade_history",
    "get_position_history",
    "get_daily_summary",
    "save_daily_summary",
    "get_trade_statistics",
    "get_portfolio_evolution",
]
