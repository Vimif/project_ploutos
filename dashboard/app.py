#!/usr/bin/env python3
"""
🏛️ PLOUTOS DASHBOARD V4 – Version adaptée à la DB existante

Compatible avec le schéma PostgreSQL actuel :

Tables :
  - trades(id, timestamp, symbol, action, quantity, price, amount, reason, portfolio_value)
  - daily_summary(id, date, portfolio_value, cash, total_pl, positions_count, trades_count)
  - predictions(id, ticker, action, confidence, timestamp, features JSONB)
  - positions (non utilisée pour l’instant car schéma non précisé)

Endpoints principaux :
  - /            : page d’accueil (HTML)
  - /api/status  : statut portfolio + trades du jour + prédictions
  - /api/trades  : historique des trades (adapté à ta table trades)
  - /api/metrics : métriques calculées depuis daily_summary
  - /api/health  : healthcheck DB
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta
import os
import numpy as np

app = Flask(__name__)
CORS(app)

# =============================================================================
# CONFIG
# =============================================================================

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "ploutos"),
    "user": os.getenv("DB_USER", "ploutos"),
    "password": os.getenv("DB_PASSWORD", "changeme"),  # à surcharger en prod
}

INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", "100000"))


# =============================================================================
# DB HELPERS
# =============================================================================

def get_db_connection():
    """Ouvre une connexion PostgreSQL."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"❌ Erreur connexion DB: {e}")
        return None


def execute_query(query, params=None, fetch=True):
    """Exécute une requête SQL et renvoie les résultats sous forme de dict."""
    conn = get_db_connection()
    if not conn:
        return None

    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(query, params or [])
        if fetch:
            result = cur.fetchall()
        else:
            conn.commit()
            result = True
        cur.close()
        conn.close()
        return result
    except Exception as e:
        print(f"❌ Erreur query: {e}")
        try:
            conn.close()
        except Exception:
            pass
        return None


# =============================================================================
# ROUTES HTML
# =============================================================================

@app.context_processor
def inject_now():
    """Injecte la date actuelle dans les templates."""
    return {"now": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


@app.route("/")
def index():
    return render_template("index.html")


# =============================================================================
# API: STATUS
# =============================================================================

@app.route("/api/status")
def api_status():
    """
    Retourne :
      - dernier daily_summary comme état du portfolio
      - nb de trades du jour (table trades)
      - dernières prédictions (table predictions)
    """
    # Dernier daily_summary
    portfolio_query = """
        SELECT
            date,
            portfolio_value,
            cash,
            total_pl,
            positions_count,
            trades_count
        FROM daily_summary
        ORDER BY date DESC
        LIMIT 1
    """
    portfolio_rows = execute_query(portfolio_query) or []
    portfolio = None

    if portfolio_rows:
        row = portfolio_rows[0]
        pv = float(row.get("portfolio_value") or 0.0)
        cash = float(row.get("cash") or 0.0)

        # Return = (PV / INITIAL_BALANCE - 1) * 100
        if INITIAL_BALANCE > 0:
            return_pct = (pv - INITIAL_BALANCE) / INITIAL_BALANCE * 100.0
        else:
            return_pct = 0.0

        portfolio = {
            "date": row.get("date").isoformat() if row.get("date") else None,
            "portfolio_value": pv,
            "cash": cash,
            "return_pct": return_pct,
            "total_pl": float(row.get("total_pl") or 0.0),
            "positions_count": int(row.get("positions_count") or 0),
            "trades_count": int(row.get("trades_count") or 0),
        }

    # Trades du jour
    trades_today_query = """
        SELECT COUNT(*) AS count
        FROM trades
        WHERE DATE(timestamp) = CURRENT_DATE
    """
    trades_today_rows = execute_query(trades_today_query) or []
    trades_today = (
        int(trades_today_rows[0]["count"]) if trades_today_rows else 0
    )

    # Prédictions récentes
    predictions_query = """
        SELECT
            ticker,
            action,
            confidence,
            timestamp
        FROM predictions
        ORDER BY timestamp DESC
        LIMIT 10
    """
    predictions_rows = execute_query(predictions_query) or []

    predictions = []
    for row in predictions_rows:
        predictions.append(
            {
                "ticker": row.get("ticker"),
                "action": row.get("action"),
                "confidence": float(row.get("confidence") or 0.0),
                "timestamp": (
                    row.get("timestamp").isoformat()
                    if row.get("timestamp")
                    else None
                ),
            }
        )

    # Pour l’instant, on ne lit pas la table positions (schéma non défini)
    positions = []

    return jsonify(
        {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "portfolio": portfolio,
            "trades_today": trades_today,
            "positions": positions,
            "recent_predictions": predictions,
        }
    )


# =============================================================================
# API: TRADES
# =============================================================================

@app.route("/api/trades")
def api_trades():
    """
    Historique des trades basé sur ta table trades :

    trades(id, timestamp, symbol, action, quantity, price, amount, reason, portfolio_value)

    On adapte le JSON pour coller à peu près à l’ancien format sans utiliser des
    colonnes qui n’existent pas (`ticker`, `pnl`, etc.).
    """
    days = request.args.get("days", 30, type=int)
    ticker = request.args.get("ticker", None, type=str)
    limit = request.args.get("limit", 100, type=int)

    base_query = """
        SELECT
            id,
            timestamp,
            symbol,
            action,
            quantity,
            price,
            amount,
            reason,
            portfolio_value
        FROM trades
        WHERE timestamp >= NOW() - INTERVAL %s
    """
    params = [f"{days} days"]

    if ticker:
        base_query += " AND symbol = %s"
        params.append(ticker)

    base_query += " ORDER BY timestamp DESC LIMIT %s"
    params.append(limit)

    rows = execute_query(base_query, params) or []

    trades = []
    for row in rows:
        trades.append(
            {
                "id": row.get("id"),
                "ticker": row.get("symbol"),  # mapping symbol -> ticker
                "action": row.get("action"),
                "shares": float(row.get("quantity") or 0.0),
                "price": float(row.get("price") or 0.0),
                "amount": float(row.get("amount") or 0.0),
                "reason": row.get("reason"),
                "entry_time": (
                    row.get("timestamp").isoformat()
                    if row.get("timestamp")
                    else None
                ),
                # On ne dispose pas d’info de sortie par trade dans ce schéma
                "exit_time": None,
                "portfolio_value": float(row.get("portfolio_value") or 0.0),
            }
        )

    # Stats simples (sans pnl car pas de colonne dédiée)
    total_trades = len(trades)
    total_amount = float(sum(t["amount"] for t in trades)) if trades else 0.0

    stats = {
        "total_trades": total_trades,
        "total_amount": total_amount,
        # placeholders pour compatibilité :
        "winning_trades": None,
        "avg_pnl_pct": None,
        "max_win": None,
        "max_loss": None,
        "total_pnl": None,
    }

    return jsonify(
        {
            "success": True,
            "trades": trades,
            "stats": stats,
            "filters": {
                "days": days,
                "ticker": ticker,
                "limit": limit,
            },
        }
    )


# =============================================================================
# API: METRICS
# =============================================================================

@app.route("/api/metrics")
def api_metrics():
    """
    Calcule les métriques globales à partir de daily_summary :

    daily_summary(id, date, portfolio_value, cash, total_pl, positions_count, trades_count)
    """
    days = request.args.get("days", 90, type=int)

    daily_query = """
        SELECT
            date,
            portfolio_value,
            cash,
            total_pl,
            positions_count,
            trades_count
        FROM daily_summary
        WHERE date >= CURRENT_DATE - %s::INTERVAL
        ORDER BY date ASC
    """
    daily_rows = execute_query(daily_query, [f"{days} days"]) or []

    if not daily_rows:
        return jsonify(
            {
                "success": True,
                "metrics": {},
                "daily_data": [],
                "period_days": days,
            }
        )

    portfolio_values = [
        float(r.get("portfolio_value") or 0.0) for r in daily_rows
    ]

    # Total return sur la période = (dernier / premier - 1) * 100
    first_value = portfolio_values[0] if portfolio_values else INITIAL_BALANCE
    last_value = portfolio_values[-1] if portfolio_values else INITIAL_BALANCE

    if first_value > 0:
        total_return = (last_value - first_value) / first_value * 100.0
    else:
        total_return = 0.0

    # Returns journaliers
    daily_returns = []
    for i in range(1, len(portfolio_values)):
        prev = portfolio_values[i - 1]
        cur = portfolio_values[i]
        if prev > 0:
            daily_returns.append((cur - prev) / prev)
        else:
            daily_returns.append(0.0)

    # Sharpe
    if daily_returns:
        sharpe = (
            np.mean(daily_returns)
            / (np.std(daily_returns) + 1e-8)
            * np.sqrt(252.0)
        )
    else:
        sharpe = 0.0

    # Max drawdown
    peak = portfolio_values[0]
    max_dd = 0.0
    for v in portfolio_values:
        if v > peak:
            peak = v
        dd = (v - peak) / peak if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd
    max_drawdown = abs(max_dd) * 100.0

    total_trades = int(
        sum(int(r.get("trades_count") or 0) for r in daily_rows)
    )
    avg_trades_per_day = (
        float(total_trades) / len(daily_rows) if daily_rows else 0.0
    )

    metrics = {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "current_value": last_value,
        "total_trades": total_trades,
        "avg_trades_per_day": avg_trades_per_day,
    }

    # On renvoie aussi les données brutes pour les graphes
    daily_data = []
    for r in daily_rows:
        daily_data.append(
            {
                "date": r.get("date").isoformat() if r.get("date") else None,
                "portfolio_value": float(r.get("portfolio_value") or 0.0),
                "cash": float(r.get("cash") or 0.0),
                "total_pl": float(r.get("total_pl") or 0.0),
                "positions_count": int(r.get("positions_count") or 0),
                "trades_count": int(r.get("trades_count") or 0),
            }
        )

    return jsonify(
        {
            "success": True,
            "metrics": metrics,
            "daily_data": daily_data,
            "period_days": days,
        }
    )


# =============================================================================
# API: PREDICTIONS (inchangée)
# =============================================================================

@app.route("/api/predictions")
def api_predictions():
    """Retourne les dernières prédictions de la table predictions."""
    limit = request.args.get("limit", 50, type=int)

    query = """
        SELECT
            ticker,
            action,
            confidence,
            timestamp,
            features
        FROM predictions
        ORDER BY timestamp DESC
        LIMIT %s
    """
    rows = execute_query(query, [limit]) or []

    predictions = []
    for r in rows:
        predictions.append(
            {
                "ticker": r.get("ticker"),
                "action": r.get("action"),
                "confidence": float(r.get("confidence") or 0.0),
                "timestamp": (
                    r.get("timestamp").isoformat()
                    if r.get("timestamp")
                    else None
                ),
                "features": r.get("features"),
            }
        )

    return jsonify({"success": True, "predictions": predictions})


# =============================================================================
# API: HEALTH
# =============================================================================

@app.route("/api/health")
def api_health():
    conn = get_db_connection()
    db_ok = conn is not None
    if conn:
        conn.close()
    return jsonify(
        {
            "success": True,
            "status": "healthy" if db_ok else "degraded",
            "database": "ok" if db_ok else "error",
            "timestamp": datetime.now().isoformat(),
        }
    )


# =============================================================================
# MAIN (mode dev)
# =============================================================================

if __name__ == "__main__":
    print("⚡ Vérification connexion DB...")
    conn = get_db_connection()
    if conn:
        print("✅ DB connectée")
        conn.close()
    else:
        print("❌ DB non accessible (mode dégradé)")

    print("\n🚀 Démarrage dashboard (dev)...")
    print("   URL: http://localhost:5000")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
