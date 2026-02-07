"""
Fonctions database pour le systeme advisory.

Suit le meme pattern que database/db.py :
- get_connection() context manager
- try/except avec fallback JSON
- psycopg2 RealDictCursor
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Import DB avec fallback
try:
    from database.db import get_connection

    DB_AVAILABLE = True
except Exception:
    DB_AVAILABLE = False
    logger.info("PostgreSQL indisponible, fallback JSON pour advisory")

# Dossier JSON fallback
JSON_DIR = Path("data/advisory_results")
JSON_DIR.mkdir(parents=True, exist_ok=True)


def init_advisory_tables() -> bool:
    """Cree les tables advisory si elles n'existent pas."""
    if not DB_AVAILABLE:
        return False

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS advisory_analyses (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT NOW(),
                        symbol VARCHAR(10) NOT NULL,
                        composite_score DECIMAL(5,4),
                        recommendation VARCHAR(20),
                        confidence DECIMAL(5,4),
                        technical_signal DECIMAL(5,4),
                        technical_confidence DECIMAL(5,4),
                        ml_signal DECIMAL(5,4),
                        ml_confidence DECIMAL(5,4),
                        sentiment_signal DECIMAL(5,4),
                        sentiment_confidence DECIMAL(5,4),
                        statistical_signal DECIMAL(5,4),
                        statistical_confidence DECIMAL(5,4),
                        risk_signal DECIMAL(5,4),
                        risk_confidence DECIMAL(5,4),
                        explanation_fr TEXT,
                        details JSONB,
                        forecast_data JSONB
                    )
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_advisory_symbol
                    ON advisory_analyses(symbol)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_advisory_timestamp
                    ON advisory_analyses(timestamp)
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS sentiment_cache (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT NOW(),
                        symbol VARCHAR(10) NOT NULL,
                        source VARCHAR(50),
                        headline TEXT,
                        sentiment_score DECIMAL(5,4),
                        url TEXT
                    )
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sentiment_symbol
                    ON sentiment_cache(symbol)
                """)
                conn.commit()
                logger.info("Tables advisory creees/verifiees")
                return True
    except Exception as e:
        logger.error(f"Erreur creation tables advisory: {e}")
        return False


def save_advisory_analysis(result_dict: Dict) -> Optional[int]:
    """Sauvegarde un resultat d'analyse. Retourne l'id ou None."""
    # Extraire les sous-signaux par source
    sub_signals = {s["source"]: s for s in result_dict.get("sub_signals", [])}

    if DB_AVAILABLE:
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    tech = sub_signals.get("technical", {})
                    ml = sub_signals.get("ml_model", {})
                    sent = sub_signals.get("sentiment", {})
                    stat = sub_signals.get("statistical", {})
                    risk = sub_signals.get("risk", {})

                    cur.execute(
                        """
                        INSERT INTO advisory_analyses (
                            symbol, composite_score, recommendation, confidence,
                            technical_signal, technical_confidence,
                            ml_signal, ml_confidence,
                            sentiment_signal, sentiment_confidence,
                            statistical_signal, statistical_confidence,
                            risk_signal, risk_confidence,
                            explanation_fr, details, forecast_data
                        ) VALUES (
                            %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s
                        ) RETURNING id
                        """,
                        (
                            result_dict["symbol"],
                            result_dict["composite_score"],
                            result_dict["recommendation"],
                            result_dict["confidence"],
                            tech.get("signal", 0),
                            tech.get("confidence", 0),
                            ml.get("signal", 0),
                            ml.get("confidence", 0),
                            sent.get("signal", 0),
                            sent.get("confidence", 0),
                            stat.get("signal", 0),
                            stat.get("confidence", 0),
                            risk.get("signal", 0),
                            risk.get("confidence", 0),
                            result_dict.get("explanation_fr", ""),
                            json.dumps(result_dict.get("indicators", {})),
                            json.dumps(
                                [
                                    fp if isinstance(fp, dict) else fp
                                    for fp in result_dict.get("forecast", [])
                                ]
                            ),
                        ),
                    )
                    row = cur.fetchone()
                    conn.commit()
                    return row[0] if row else None
        except Exception as e:
            logger.error(f"Erreur sauvegarde advisory DB: {e}")

    # Fallback JSON
    try:
        filepath = JSON_DIR / f"{result_dict['symbol']}_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(filepath, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)
        return None
    except Exception as e:
        logger.error(f"Erreur sauvegarde advisory JSON: {e}")
        return None


def get_latest_analysis(symbol: str) -> Optional[Dict]:
    """Recupere la derniere analyse pour un symbole."""
    if DB_AVAILABLE:
        try:
            from psycopg2.extras import RealDictCursor

            with get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        """
                        SELECT * FROM advisory_analyses
                        WHERE symbol = %s
                        ORDER BY timestamp DESC
                        LIMIT 1
                        """,
                        (symbol.upper(),),
                    )
                    row = cur.fetchone()
                    return dict(row) if row else None
        except Exception as e:
            logger.error(f"Erreur lecture advisory DB: {e}")

    # Fallback JSON : chercher le fichier le plus recent
    try:
        files = sorted(JSON_DIR.glob(f"{symbol.upper()}_*.json"), reverse=True)
        if files:
            with open(files[0]) as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Erreur lecture advisory JSON: {e}")

    return None


def get_analysis_history(symbol: str, days: int = 30) -> List[Dict]:
    """Recupere l'historique des analyses pour un symbole."""
    if DB_AVAILABLE:
        try:
            from psycopg2.extras import RealDictCursor

            with get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        """
                        SELECT * FROM advisory_analyses
                        WHERE symbol = %s
                        AND timestamp > NOW() - INTERVAL '%s days'
                        ORDER BY timestamp DESC
                        """,
                        (symbol.upper(), days),
                    )
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Erreur historique advisory DB: {e}")

    # Fallback JSON
    results = []
    try:
        for f in sorted(JSON_DIR.glob(f"{symbol.upper()}_*.json"), reverse=True):
            with open(f) as fh:
                results.append(json.load(fh))
    except Exception as e:
        logger.error(f"Erreur historique advisory JSON: {e}")

    return results
