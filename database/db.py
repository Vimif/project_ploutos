# database/db.py
"""Gestionnaire de base de données PostgreSQL pour Ploutos"""

import psycopg2
import psycopg2.extras
import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, date
from contextlib import contextmanager
from dotenv import load_dotenv
from core.utils import setup_logging

load_dotenv()
logger = setup_logging(__name__)

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "ploutos"),
    "user": os.getenv("DB_USER", "ploutos"),
    "password": os.getenv("DB_PASSWORD", ""),
}


@contextmanager
def get_connection():
    """Context manager pour connexion BDD"""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"❌ Erreur BDD: {e}")
        raise
    finally:
        if conn:
            conn.close()


def init_database():
    """Initialiser les tables"""
    schema = """
    -- Trades
    CREATE TABLE IF NOT EXISTS trades (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP DEFAULT NOW(),
        symbol VARCHAR(10) NOT NULL,
        action VARCHAR(10) NOT NULL,
        quantity DECIMAL(10,4),
        price DECIMAL(10,2),
        amount DECIMAL(12,2),
        reason TEXT,
        portfolio_value DECIMAL(12,2),
        order_id VARCHAR(100)
    );

    -- Positions (snapshots)
    CREATE TABLE IF NOT EXISTS positions (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP DEFAULT NOW(),
        symbol VARCHAR(10) NOT NULL,
        quantity DECIMAL(10,4),
        avg_entry_price DECIMAL(10,2),
        current_price DECIMAL(10,2),
        market_value DECIMAL(12,2),
        unrealized_pl DECIMAL(12,2),
        unrealized_plpc DECIMAL(8,4)
    );

    -- Résumés quotidiens
    CREATE TABLE IF NOT EXISTS daily_summary (
        id SERIAL PRIMARY KEY,
        date DATE UNIQUE NOT NULL,
        portfolio_value DECIMAL(12,2),
        cash DECIMAL(12,2),
        buying_power DECIMAL(12,2),
        total_pl DECIMAL(12,2),
        positions_count INTEGER,
        trades_count INTEGER
    );

    -- Prédictions du modèle
    CREATE TABLE IF NOT EXISTS predictions (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP DEFAULT NOW(),
        symbol VARCHAR(10) NOT NULL,
        sector VARCHAR(50),
        prediction INTEGER,
        confidence DECIMAL(5,4),
        action VARCHAR(10),
        features JSONB
    );

    -- Index
    CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
    CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
    CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
    CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON predictions(symbol);
    """

    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(schema)
            logger.info("✅ Base de données initialisée")
    except Exception as e:
        logger.error(f"❌ Erreur init DB: {e}")


# ==================== TRADES ====================


def log_trade(
    symbol: str,
    action: str,
    quantity: float,
    price: float,
    amount: float,
    reason: str = "",
    portfolio_value: float = None,
    order_id: str = None,
):
    """Enregistrer un trade"""
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO trades (symbol, action, quantity, price, amount, reason, 
                                  portfolio_value, order_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """,
                (symbol, action, quantity, price, amount, reason, portfolio_value, order_id),
            )

            trade_id = cur.fetchone()[0]
            logger.info(f"✅ Trade logged: {action} {quantity} {symbol} @ ${price:.2f}")
            return trade_id
    except Exception as e:
        logger.error(f"❌ Erreur log_trade: {e}")
        return None


def get_trade_history(days: int = 30, symbol: str = None) -> List[Dict]:
    """Récupérer l'historique des trades"""
    try:
        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            if symbol:
                cur.execute(
                    """
                    SELECT * FROM trades 
                    WHERE timestamp > NOW() - INTERVAL %s AND symbol = %s
                    ORDER BY timestamp DESC
                """,
                    (f"{days} days", symbol),
                )
            else:
                cur.execute(
                    """
                    SELECT * FROM trades 
                    WHERE timestamp > NOW() - INTERVAL %s
                    ORDER BY timestamp DESC
                """,
                    (f"{days} days",),
                )

            return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"❌ Erreur get_trade_history: {e}")
        return []


# ==================== POSITIONS ====================


def log_position(
    symbol: str,
    quantity: float,
    avg_entry_price: float,
    current_price: float,
    market_value: float,
    unrealized_pl: float,
    unrealized_plpc: float,
):
    """Enregistrer un snapshot de position"""
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO positions (symbol, quantity, avg_entry_price, current_price,
                                     market_value, unrealized_pl, unrealized_plpc)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """,
                (
                    symbol,
                    quantity,
                    avg_entry_price,
                    current_price,
                    market_value,
                    unrealized_pl,
                    unrealized_plpc,
                ),
            )

            return cur.fetchone()[0]
    except Exception as e:
        logger.error(f"❌ Erreur log_position: {e}")
        return None


def get_position_history(symbol: str, days: int = 30) -> List[Dict]:
    """Historique d'une position"""
    try:
        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                """
                SELECT * FROM positions 
                WHERE symbol = %s AND timestamp > NOW() - INTERVAL %s
                ORDER BY timestamp DESC
            """,
                (symbol, f"{days} days"),
            )

            return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"❌ Erreur get_position_history: {e}")
        return []


def log_all_positions(positions_list: List[Dict]):
    """Logger plusieurs positions en batch"""
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            for pos in positions_list:
                cur.execute(
                    """
                    INSERT INTO positions (symbol, quantity, avg_entry_price, current_price,
                                         market_value, unrealized_pl, unrealized_plpc)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                    (
                        pos["symbol"],
                        pos["qty"],
                        pos["avg_entry_price"],
                        pos["current_price"],
                        pos["market_value"],
                        pos["unrealized_pl"],
                        pos["unrealized_plpc"],
                    ),
                )

            logger.info(f"✅ {len(positions_list)} positions loggées")
    except Exception as e:
        logger.error(f"❌ Erreur log_all_positions: {e}")


# ==================== DAILY SUMMARY ====================


def save_daily_summary(
    date: date,
    portfolio_value: float,
    cash: float,
    buying_power: float,
    total_pl: float,
    positions_count: int,
    trades_count: int,
):
    """Sauvegarder le résumé quotidien"""
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO daily_summary (date, portfolio_value, cash, buying_power,
                                         total_pl, positions_count, trades_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (date) DO UPDATE SET
                    portfolio_value = EXCLUDED.portfolio_value,
                    cash = EXCLUDED.cash,
                    buying_power = EXCLUDED.buying_power,
                    total_pl = EXCLUDED.total_pl,
                    positions_count = EXCLUDED.positions_count,
                    trades_count = EXCLUDED.trades_count
            """,
                (
                    date,
                    portfolio_value,
                    cash,
                    buying_power,
                    total_pl,
                    positions_count,
                    trades_count,
                ),
            )

            logger.info(f"✅ Résumé quotidien sauvegardé: {date}")
    except Exception as e:
        logger.error(f"❌ Erreur save_daily_summary: {e}")


def get_daily_summary(days: int = 30) -> List[Dict]:
    """Récupérer les résumés quotidiens"""
    try:
        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                """
                SELECT * FROM daily_summary 
                WHERE date > CURRENT_DATE - INTERVAL %s
                ORDER BY date DESC
            """,
                (f"{days} days",),
            )

            return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"❌ Erreur get_daily_summary: {e}")
        return []


# ==================== PREDICTIONS ====================


def log_prediction(
    symbol: str, sector: str, prediction: int, confidence: float, action: str, features: dict = None
):
    """Enregistrer une prédiction"""
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO predictions (symbol, sector, prediction, confidence, action, features)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """,
                (symbol, sector, prediction, confidence, action, json.dumps(features or {})),
            )

            return cur.fetchone()[0]
    except Exception as e:
        logger.error(f"❌ Erreur log_prediction: {e}")
        return None


def get_prediction_history(symbol: str = None, days: int = 7) -> List[Dict]:
    """Récupérer l'historique des prédictions"""
    try:
        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            if symbol:
                cur.execute(
                    """
                    SELECT * FROM predictions 
                    WHERE symbol = %s AND timestamp > NOW() - INTERVAL %s
                    ORDER BY timestamp DESC
                """,
                    (symbol, f"{days} days"),
                )
            else:
                cur.execute(
                    """
                    SELECT * FROM predictions 
                    WHERE timestamp > NOW() - INTERVAL %s
                    ORDER BY timestamp DESC
                """,
                    (f"{days} days",),
                )

            return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"❌ Erreur get_prediction_history: {e}")
        return []


# ==================== ANALYTICS ====================


def get_trade_statistics(days: int = 30) -> Dict:
    """Statistiques des trades"""
    try:
        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                """
                SELECT 
                    COUNT(*) as total_trades,
                    COUNT(CASE WHEN action = 'BUY' THEN 1 END) as buy_count,
                    COUNT(CASE WHEN action = 'SELL' THEN 1 END) as sell_count,
                    SUM(amount) as total_volume,
                    AVG(amount) as avg_trade_size,
                    MAX(amount) as max_trade_size,
                    MIN(amount) as min_trade_size
                FROM trades
                WHERE timestamp > NOW() - INTERVAL %s
            """,
                (f"{days} days",),
            )

            result = cur.fetchone()
            return dict(result) if result else {}
    except Exception as e:
        logger.error(f"❌ Erreur get_trade_statistics: {e}")
        return {}


def get_top_symbols(days: int = 30, limit: int = 10) -> List[Dict]:
    """Top symboles tradés"""
    try:
        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                """
                SELECT 
                    symbol,
                    COUNT(*) as trade_count,
                    SUM(amount) as total_volume,
                    AVG(price) as avg_price
                FROM trades
                WHERE timestamp > NOW() - INTERVAL %s
                GROUP BY symbol
                ORDER BY total_volume DESC
                LIMIT %s
            """,
                (f"{days} days", limit),
            )

            return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"❌ Erreur get_top_symbols: {e}")
        return []


def get_portfolio_evolution(days: int = 30) -> List[Dict]:
    """Évolution du portfolio"""
    try:
        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                """
                SELECT date, portfolio_value, total_pl, cash
                FROM daily_summary
                WHERE date > CURRENT_DATE - INTERVAL %s
                ORDER BY date ASC
            """,
                (f"{days} days",),
            )

            return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"❌ Erreur get_portfolio_evolution: {e}")
        return []


def get_win_loss_ratio(days: int = 30) -> Dict:
    """Calculer le ratio gains/pertes"""
    try:
        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Trouver les paires buy/sell
            cur.execute(
                """
                WITH trade_pairs AS (
                    SELECT 
                        symbol,
                        action,
                        price,
                        amount,
                        timestamp,
                        LAG(price) OVER (PARTITION BY symbol ORDER BY timestamp) as prev_price,
                        LAG(action) OVER (PARTITION BY symbol ORDER BY timestamp) as prev_action
                    FROM trades
                    WHERE timestamp > NOW() - INTERVAL %s
                )
                SELECT 
                    COUNT(CASE WHEN action = 'SELL' AND price > prev_price THEN 1 END) as wins,
                    COUNT(CASE WHEN action = 'SELL' AND price <= prev_price THEN 1 END) as losses
                FROM trade_pairs
                WHERE action = 'SELL' AND prev_action = 'BUY'
            """,
                (f"{days} days",),
            )

            result = cur.fetchone()
            if result:
                wins = int(result["wins"] or 0)
                losses = int(result["losses"] or 0)
                total = wins + losses
                win_rate = (wins / total * 100) if total > 0 else 0

                return {"wins": wins, "losses": losses, "total": total, "win_rate": win_rate}

            return {"wins": 0, "losses": 0, "total": 0, "win_rate": 0}
    except Exception as e:
        logger.error(f"❌ Erreur get_win_loss_ratio: {e}")
        return {"wins": 0, "losses": 0, "total": 0, "win_rate": 0}
