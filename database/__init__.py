# ruff: noqa: F403, F405
# flake8: noqa
# database/__init__.py
"""Package de gestion de la base de données"""

from .db import *

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
