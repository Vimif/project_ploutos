"""Routes API pour les donnees de portfolio."""

import logging
from flask import Blueprint, jsonify

logger = logging.getLogger(__name__)

portfolio_bp = Blueprint("portfolio", __name__, url_prefix="/api/portfolio")

# Essayer Alpaca
try:
    from trading.alpaca_client import AlpacaClient

    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False


@portfolio_bp.route("/account")
def account():
    """Informations du compte Alpaca."""
    if not ALPACA_AVAILABLE:
        return jsonify(
            {
                "portfolio_value": 0,
                "cash": 0,
                "buying_power": 0,
                "available": False,
            }
        )

    try:
        client = AlpacaClient()
        account_info = client.get_account()
        return jsonify(account_info)
    except Exception as e:
        logger.error(f"Erreur account: {e}")
        return jsonify({"error": str(e)}), 500


@portfolio_bp.route("/positions")
def positions():
    """Positions ouvertes."""
    if not ALPACA_AVAILABLE:
        return jsonify([])

    try:
        client = AlpacaClient()
        pos = client.get_positions()
        return jsonify(pos)
    except Exception as e:
        logger.error(f"Erreur positions: {e}")
        return jsonify({"error": str(e)}), 500


@portfolio_bp.route("/analytics")
def analytics():
    """Metriques de performance du portfolio."""
    try:
        from dashboard.analytics import PortfolioAnalytics

        pa = PortfolioAnalytics()
        metrics = pa.get_all_metrics()
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Erreur analytics: {e}")
        return jsonify({"error": str(e), "available": False}), 500
