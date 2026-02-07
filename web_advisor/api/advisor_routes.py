"""Routes API pour les analyses advisory."""

import logging
from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

advisor_bp = Blueprint("advisor", __name__, url_prefix="/api/advisor")

# Reference au moteur (injectee au register)
_engine = None


def init_engine(engine):
    global _engine
    _engine = engine


@advisor_bp.route("/analyze/<symbol>")
def analyze_symbol(symbol):
    """Analyse complete pour un symbole."""
    period = request.args.get("period", "3mo")
    interval = request.args.get("interval", "1h")

    try:
        result = _engine.analyze(symbol, period=period, interval=interval)
        return jsonify(result.to_dict())
    except Exception as e:
        logger.error(f"Erreur analyse {symbol}: {e}")
        return jsonify({"error": str(e)}), 500


@advisor_bp.route("/watchlist")
def watchlist():
    """Analyse de tous les tickers de la watchlist."""
    period = request.args.get("period", "3mo")
    interval = request.args.get("interval", "1h")

    try:
        results = _engine.analyze_watchlist(period=period, interval=interval)
        return jsonify([r.to_dict() for r in results])
    except Exception as e:
        logger.error(f"Erreur watchlist: {e}")
        return jsonify({"error": str(e)}), 500


@advisor_bp.route("/top-picks")
def top_picks():
    """Top 5 achats et top 5 ventes."""
    n = request.args.get("n", 5, type=int)

    try:
        picks = _engine.get_top_picks(n=n)
        return jsonify(
            {
                "top_buys": [r.to_dict() for r in picks["top_buys"]],
                "top_sells": [r.to_dict() for r in picks["top_sells"]],
            }
        )
    except Exception as e:
        logger.error(f"Erreur top-picks: {e}")
        return jsonify({"error": str(e)}), 500


@advisor_bp.route("/history/<symbol>")
def analysis_history(symbol):
    """Historique des analyses pour un symbole."""
    days = request.args.get("days", 30, type=int)

    try:
        from database.advisory_db import get_analysis_history

        history = get_analysis_history(symbol, days=days)
        return jsonify(history)
    except Exception as e:
        logger.error(f"Erreur historique {symbol}: {e}")
        return jsonify({"error": str(e)}), 500
