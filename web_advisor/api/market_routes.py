"""Routes API pour les donnees de marche (OHLCV, indicateurs)."""

import logging
from flask import Blueprint, jsonify, request

import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)

market_bp = Blueprint("market", __name__, url_prefix="/api/market")


@market_bp.route("/ohlcv/<symbol>")
def ohlcv(symbol):
    """Donnees OHLCV pour graphiques en chandelier."""
    period = request.args.get("period", "3mo")
    interval = request.args.get("interval", "1h")
    limit = request.args.get("limit", 200, type=int)

    try:
        ticker = yf.Ticker(symbol.upper())
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            return jsonify({"error": f"Pas de donnees pour {symbol}"}), 404

        # Limiter le nombre de barres
        df = df.tail(limit)

        data = []
        for idx, row in df.iterrows():
            data.append(
                {
                    "date": idx.isoformat(),
                    "open": round(float(row["Open"]), 2),
                    "high": round(float(row["High"]), 2),
                    "low": round(float(row["Low"]), 2),
                    "close": round(float(row["Close"]), 2),
                    "volume": int(row["Volume"]),
                }
            )

        return jsonify(data)

    except Exception as e:
        logger.error(f"Erreur OHLCV {symbol}: {e}")
        return jsonify({"error": str(e)}), 500


@market_bp.route("/indicators/<symbol>")
def indicators(symbol):
    """Series temporelles d'indicateurs techniques pour overlays."""
    period = request.args.get("period", "3mo")
    interval = request.args.get("interval", "1h")

    try:
        ticker = yf.Ticker(symbol.upper())
        df = ticker.history(period=period, interval=interval)

        if df.empty or len(df) < 50:
            return jsonify({"error": "Donnees insuffisantes"}), 404

        close = df["Close"]

        # SMA
        sma_20 = close.rolling(window=20).mean()
        sma_50 = close.rolling(window=50).mean()

        # Bollinger Bands
        bb_mid = sma_20
        bb_std = close.rolling(window=20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - macd_signal

        # Volume
        volume = df["Volume"]

        # Construire les series
        dates = [idx.isoformat() for idx in df.index]

        def to_list(series):
            return [
                round(float(v), 4) if pd.notna(v) else None
                for v in series.values
            ]

        return jsonify(
            {
                "dates": dates,
                "sma_20": to_list(sma_20),
                "sma_50": to_list(sma_50),
                "bb_upper": to_list(bb_upper),
                "bb_lower": to_list(bb_lower),
                "bb_middle": to_list(bb_mid),
                "rsi": to_list(rsi),
                "macd_line": to_list(macd_line),
                "macd_signal": to_list(macd_signal),
                "macd_histogram": to_list(macd_hist),
                "volume": to_list(volume),
            }
        )

    except Exception as e:
        logger.error(f"Erreur indicateurs {symbol}: {e}")
        return jsonify({"error": str(e)}), 500


@market_bp.route("/sectors")
def sectors():
    """Score moyen par secteur."""
    try:
        from config.tickers import SECTORS

        sector_data = {}
        for sector_name, sector_info in SECTORS.items():
            sector_data[sector_name] = {
                "tickers": sector_info["tickers"],
                "allocation": sector_info["allocation"],
            }

        return jsonify(sector_data)

    except Exception as e:
        logger.error(f"Erreur secteurs: {e}")
        return jsonify({"error": str(e)}), 500
