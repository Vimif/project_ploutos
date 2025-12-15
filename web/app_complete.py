#!/usr/bin/env python3
"""
🌐 PLOUTOS WEB DASHBOARD - V8 ORACLE + TRADER PRO + 5 TOOLS + WATCHLISTS + LIVE TRADING
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from dataclasses import is_dataclass, asdict
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COMPLETE_INDICATORS = False
TRADER_PRO = False

try:
    from web.utils.all_indicators import calculate_complete_indicators, get_indicator_signals
    from web.utils.advanced_ai import AdvancedAIAnalyzer
    COMPLETE_INDICATORS = True
    logger.info("✅ Indicateurs avancés chargés")
except Exception as e:
    logger.warning(f"⚠️  Indicateurs avancés non disponibles: {e}")
    import ta

try:
    from web.utils.pattern_detector import PatternDetector
    from web.utils.multi_timeframe import MultiTimeframeAnalyzer
    TRADER_PRO = True
    logger.info("✅ TRADER PRO modules chargés")
except Exception as e:
    logger.error(f"❌ TRADER PRO non disponible: {e}")

try:
    from web.utils.screener import StockScreener
    from web.utils.alerts import AlertSystem
    from web.utils.backtester import Backtester
    from web.utils.correlation_analyzer import CorrelationAnalyzer
    from web.utils.portfolio_tracker import PortfolioTracker
    TOOLS_AVAILABLE = True
    logger.info("✅ 5 TOOLS chargés")
except Exception as e:
    TOOLS_AVAILABLE = False
    logger.error(f"❌ TOOLS non disponibles: {e}")

try:
    from web.utils.chart_tools import ChartTools
    CHART_TOOLS_AVAILABLE = True
    logger.info("✅ Chart Tools chargés")
except Exception as e:
    CHART_TOOLS_AVAILABLE = False
    logger.error(f"❌ Chart Tools non disponibles: {e}")

try:
    from web.utils.pro_technical_analyzer import ProTechnicalAnalyzer
    PRO_ANALYZER_AVAILABLE = True
    logger.info("✅ Pro Technical Analyzer chargé")
except Exception as e:
    PRO_ANALYZER_AVAILABLE = False
    logger.error(f"❌ Pro Technical Analyzer non disponible: {e}")

# 📊 WATCHLISTS
try:
    from web.routes import watchlists_bp
    WATCHLISTS_AVAILABLE = True
    logger.info("✅ Watchlists module chargé")
except Exception as e:
    WATCHLISTS_AVAILABLE = False
    logger.error(f"❌ Watchlists non disponibles: {e}")

# 🔥 LIVE TRADING
try:
    from web.routes.live_trading import live_bp
    LIVE_TRADING_AVAILABLE = True
    logger.info("✅ Live Trading module chargé")
except Exception as e:
    LIVE_TRADING_AVAILABLE = False
    logger.error(f"❌ Live Trading non disponible: {e}")

# 🔥 LIVE WATCHLISTS
try:
    from web.routes.live_watchlists import live_watchlists_bp
    LIVE_WATCHLISTS_AVAILABLE = True
    logger.info("✅ Live Watchlists module chargé")
except Exception as e:
    LIVE_WATCHLISTS_AVAILABLE = False
    logger.error(f"❌ Live Watchlists non disponibles: {e}")

try:
    from trading.alpaca_client import AlpacaClient
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

try:
    from src.models.v8_oracle_ensemble import V8OracleEnsemble
    V8_ORACLE_AVAILABLE = True
except ImportError:
    V8_ORACLE_AVAILABLE = False

# 💾 DATABASE
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False


def clean_for_json(obj):
    if is_dataclass(obj) and not isinstance(obj, type):
        return clean_for_json(asdict(obj))
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [clean_for_json(x) for x in obj.tolist()]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(item) for item in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    return obj


app = Flask(__name__)
CORS(app)

# Register blueprints
if WATCHLISTS_AVAILABLE:
    app.register_blueprint(watchlists_bp)
    logger.info("✅ Watchlists blueprint enregistré")

if LIVE_TRADING_AVAILABLE:
    app.register_blueprint(live_bp)
    logger.info("✅ Live Trading blueprint enregistré")

if LIVE_WATCHLISTS_AVAILABLE:
    app.register_blueprint(live_watchlists_bp)
    logger.info("✅ Live Watchlists blueprint enregistré")

alpaca_client = None
if ALPACA_AVAILABLE:
    try:
        alpaca_client = AlpacaClient(paper_trading=True)
    except Exception as e:
        logger.warning(f"⚠️  Alpaca: {e}")

v8_oracle = None
if V8_ORACLE_AVAILABLE:
    try:
        v8_oracle = V8OracleEnsemble()
        if v8_oracle.load_models():
            logger.info("✅ V8 Oracle chargé")
        else:
            v8_oracle = None
    except Exception as e:
        logger.warning(f"⚠️  V8: {e}")
        v8_oracle = None

ai_analyzer = None
if COMPLETE_INDICATORS:
    try:
        ai_analyzer = AdvancedAIAnalyzer()
        logger.info("✅ AI Analyzer initialisé")
    except Exception as e:
        logger.error(f"❌ AI Analyzer error: {e}")

pattern_detector = None
mtf_analyzer = None

if TRADER_PRO:
    try:
        pattern_detector = PatternDetector()
        logger.info("✅ Pattern Detector initialisé")
    except Exception as e:
        logger.error(f"❌ Pattern Detector error: {e}")
    
    try:
        mtf_analyzer = MultiTimeframeAnalyzer()
        logger.info("✅ MTF Analyzer initialisé")
    except Exception as e:
        logger.error(f"❌ MTF Analyzer error: {e}")

screener = None
alert_system = None
backtester = None
corr_analyzer = None
portfolio = None

if TOOLS_AVAILABLE:
    try:
        screener = StockScreener()
        alert_system = AlertSystem()
        backtester = Backtester()
        corr_analyzer = CorrelationAnalyzer()
        portfolio = PortfolioTracker()
        logger.info("✅ Tous les TOOLS initialisés")
    except Exception as e:
        logger.error(f"❌ Erreur init TOOLS: {e}")

chart_tools = None
if CHART_TOOLS_AVAILABLE:
    chart_tools = ChartTools()
    logger.info("✅ Chart Tools initialisé")

pro_analyzer = None
if PRO_ANALYZER_AVAILABLE:
    pro_analyzer = ProTechnicalAnalyzer()
    logger.info("✅ Pro Technical Analyzer initialisé")

cache = {'account': None, 'positions': None, 'last_update': None, 'chart_cache': {}}
DEFAULT_WATCHLIST = ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'BAC']


def get_db_connection():
    if not DB_AVAILABLE:
        return None
    try:
        import os
        db_password = os.getenv('POSTGRES_PASSWORD', 'your_password_here')
        return psycopg2.connect(
            host="localhost",
            database="ploutos",
            user="ploutos",
            password=db_password
        )
    except Exception as e:
        logger.warning(f"⚠️  DB connexion: {e}")
        return None


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chart')
def chart_page():
    return render_template('advanced_chart.html')

@app.route('/tools')
def tools_page():
    return render_template('tools.html')

@app.route('/live')
def live_page():
    return render_template('live.html')


@app.route('/api/health')
def api_health():
    return jsonify({
        'status': 'healthy',
        'modules': {
            'live_trading': LIVE_TRADING_AVAILABLE,
            'live_watchlists': LIVE_WATCHLISTS_AVAILABLE,
            'alpaca': alpaca_client is not None
        }
    }), 200


if __name__ == '__main__':
    import os
    host = os.getenv('DASHBOARD_HOST', '0.0.0.0')
    port = int(os.getenv('DASHBOARD_PORT', 5000))
    print("\n" + "="*70)
    print("🌐 PLOUTOS - V8 ORACLE + LIVE TRADING + WATCHLISTS")
    print("="*70)
    print(f"\n🚀 http://{host}:{port}")
    print(f"🔥 Live Trading: http://{host}:{port}/live")
    if LIVE_WATCHLISTS_AVAILABLE:
        print(f"📊 9 Watchlists prédéfinies disponibles")
    print("\n" + "="*70 + "\n")
    app.run(host=host, port=port, debug=False)
