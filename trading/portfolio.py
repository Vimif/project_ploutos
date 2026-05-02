# trading/portfolio.py
"""Gestion du portefeuille de trading"""

from datetime import datetime
from typing import Dict, List
import json
from pathlib import Path
from config.settings import TRADES_DIR
from core.utils import setup_logging

logger = setup_logging(__name__)


class Portfolio:
    """Gestionnaire de portefeuille"""

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trades_history = []
        self.portfolio_value_history = []

    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """Calculer la valeur totale du portefeuille"""
        positions_value = sum(
            pos["shares"] * current_prices.get(ticker, pos["entry_price"])
            for ticker, pos in self.positions.items()
        )
        return self.cash + positions_value

    def buy(self, ticker: str, price: float, amount: float):
        """Acheter des actions"""
        if amount > self.cash:
            logger.warning(f"⚠️  Fonds insuffisants pour {ticker}")
            return False

        shares = amount / price
        cost = shares * price

        if ticker in self.positions:
            old_shares = self.positions[ticker]["shares"]
            old_entry = self.positions[ticker]["entry_price"]
            new_shares = old_shares + shares
            new_entry = (old_shares * old_entry + cost) / new_shares

            self.positions[ticker] = {"shares": new_shares, "entry_price": new_entry}
        else:
            self.positions[ticker] = {"shares": shares, "entry_price": price}

        self.cash -= cost

        self.trades_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "action": "BUY",
                "ticker": ticker,
                "price": price,
                "shares": shares,
                "amount": cost,
            }
        )

        logger.info(f"🟢 BUY {ticker}: {shares:.4f} @ ${price:.2f}")
        return True

    def sell(self, ticker: str, price: float, percentage: float = 1.0):
        """Vendre des actions"""
        if ticker not in self.positions:
            logger.warning(f"⚠️  Aucune position sur {ticker}")
            return False

        shares_to_sell = self.positions[ticker]["shares"] * percentage
        proceeds = shares_to_sell * price

        entry_price = self.positions[ticker]["entry_price"]
        pnl = (price - entry_price) * shares_to_sell
        pnl_pct = ((price / entry_price) - 1) * 100

        self.cash += proceeds

        if percentage >= 1.0:
            del self.positions[ticker]
        else:
            self.positions[ticker]["shares"] -= shares_to_sell

        self.trades_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "action": "SELL",
                "ticker": ticker,
                "price": price,
                "shares": shares_to_sell,
                "amount": proceeds,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            }
        )

        logger.info(f"🔴 SELL {ticker}: {shares_to_sell:.4f} @ ${price:.2f}")
        return True

    def get_position(self, ticker: str):
        """Obtenir la position sur un ticker"""
        return self.positions.get(ticker)

    def get_summary(self, current_prices: Dict[str, float] = None):
        """Résumé du portefeuille"""
        current_prices = current_prices or {}

        total_value = self.get_total_value(current_prices)
        total_return = total_value - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100

        return {
            "cash": self.cash,
            "positions_count": len(self.positions),
            "positions": self.positions.copy(),
            "total_value": total_value,
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "trades_count": len(self.trades_history),
        }

    def save_state(self, filename: str = None):
        """Sauvegarder l'état du portefeuille"""
        if filename is None:
            filename = f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = TRADES_DIR / filename

        state = {
            "timestamp": datetime.now().isoformat(),
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "positions": self.positions,
            "trades_history": self.trades_history[-100:],
        }

        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"💾 Portefeuille sauvegardé: {filepath}")

    def load_state(self, filename: str):
        """Charger l'état du portefeuille"""
        filepath = TRADES_DIR / filename

        if not filepath.exists():
            logger.error(f"❌ Fichier introuvable: {filepath}")
            return False

        with open(filepath, "r") as f:
            state = json.load(f)

        self.initial_capital = state["initial_capital"]
        self.cash = state["cash"]
        self.positions = state["positions"]
        self.trades_history = state["trades_history"]

        logger.info(f"📂 Portefeuille chargé: {filepath}")
        return True
