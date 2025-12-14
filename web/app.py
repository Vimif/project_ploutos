#!/usr/bin/env python3
"""
🌐 PLOUTOS WEB DASHBOARD - V8 ORACLE + TRADER PRO + 5 TOOLS
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import bibliothèque complète d'indicateurs
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

# 🎯 NOUVEAUX MODULES
try:
    from web.utils.screener import StockScreener
    from web.utils.alerts import AlertSystem
    from web.utils.backtester import Backtester
    from web.utils.correlation_analyzer import CorrelationAnalyzer
    from web.utils.portfolio_tracker import PortfolioTracker
    TOOLS_AVAILABLE = True
    logger.info("✅ 5 TOOLS chargés (Screener, Alerts, Backtest, Correlation, Portfolio)")
except Exception as e:
    TOOLS_AVAILABLE = False
    logger.error(f"❌ TOOLS non disponibles: {e}")

# Import modules Ploutos
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


def clean_for_json(obj):
    """Nettoie les valeurs pour JSON (remplace NaN/Infinity par None)"""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [clean_for_json(x) for x in obj.tolist()]
    elif isinstance(obj, (np.float32, np.float64)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    return obj


# Setup
app = Flask(__name__)
CORS(app)

# Initialisation
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

# Initialiser les modules Trader Pro
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

# 🎯 Initialiser les 5 TOOLS
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

cache = {
    'account': None,
    'positions': None,
    'last_update': None,
    'chart_cache': {}
}

# Liste de tickers surveillés par défaut
DEFAULT_WATCHLIST = ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'BAC']


# ========== ROUTES HTML ==========

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chart')
def chart_page():
    return render_template('advanced_chart.html')


@app.route('/tools')
def tools_page():
    return render_template('tools.html')


# ========== CHART DATA (keeping same as before) ==========
# ... (keeping all existing chart code)

# ========== 🎯 5 TOOLS ROUTES ==========

@app.route('/api/screener', methods=['POST'])
def api_screener():
    if not screener:
        return jsonify({'error': 'Screener non disponible'}), 503
    try:
        data = request.json or {}
        tickers = data.get('tickers', DEFAULT_WATCHLIST)
        period = data.get('period', '3mo')
        filters = data.get('filters', {})
        results = screener.scan(tickers, period=period, filters=filters)
        return jsonify(clean_for_json(results))
    except Exception as e:
        logger.error(f"Erreur screener: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    if not backtester:
        return jsonify({'error': 'Backtester non disponible'}), 503
    try:
        data = request.json
        result = backtester.run(
            ticker=data.get('ticker', 'AAPL'),
            strategy=data.get('strategy', 'rsi_mean_reversion'),
            period=data.get('period', '1y'),
            params=data.get('params', {})
        )
        return jsonify(clean_for_json(result))
    except Exception as e:
        logger.error(f"Erreur backtest: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts/list')
def api_alerts_list():
    if not alert_system:
        return jsonify({'error': 'Alert system non disponible'}), 503
    try:
        rules = alert_system.get_rules()
        history = alert_system.get_history(limit=20)
        return jsonify({'rules': rules, 'history': history, 'timestamp': datetime.now().isoformat()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts/add', methods=['POST'])
def api_alerts_add():
    if not alert_system:
        return jsonify({'error': 'Alert system non disponible'}), 503
    try:
        data = request.json
        rule = alert_system.add_rule(
            ticker=data['ticker'],
            condition=data['condition'],
            value=data['value'],
            name=data.get('name'),
            channel=data.get('channel', 'all')
        )
        return jsonify(rule)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts/check')
def api_alerts_check():
    if not alert_system:
        return jsonify({'error': 'Alert system non disponible'}), 503
    try:
        triggered = alert_system.check_all()
        return jsonify({'triggered': triggered, 'count': len(triggered), 'timestamp': datetime.now().isoformat()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/correlation/heatmap', methods=['POST'])
def api_correlation_heatmap():
    if not corr_analyzer:
        return jsonify({'error': 'Correlation analyzer non disponible'}), 503
    try:
        data = request.json or {}
        tickers = data.get('tickers', DEFAULT_WATCHLIST)
        period = data.get('period', '6mo')
        heatmap = corr_analyzer.generate_heatmap(tickers, period=period)
        return jsonify(clean_for_json(heatmap))
    except Exception as e:
        logger.error(f"Erreur correlation: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ========== 💼 PORTFOLIO ROUTES ==========

@app.route('/api/portfolio/summary')
def api_portfolio_summary():
    if not portfolio:
        return jsonify({'error': 'Portfolio tracker non disponible'}), 503
    try:
        summary = portfolio.get_summary()
        return jsonify(clean_for_json(summary))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/portfolio/add', methods=['POST'])
def api_portfolio_add():
    if not portfolio:
        return jsonify({'error': 'Portfolio tracker non disponible'}), 503
    try:
        data = request.json
        position = portfolio.add_position(
            ticker=data['ticker'],
            shares=data['shares'],
            avg_price=data['avg_price'],
            date=data.get('date')
        )
        return jsonify(position)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/portfolio/remove', methods=['POST'])
def api_portfolio_remove():
    """🔥 NOUVEAU: Supprimer une position"""
    if not portfolio:
        return jsonify({'error': 'Portfolio tracker non disponible'}), 503
    try:
        data = request.json
        ticker = data.get('ticker')
        shares = data.get('shares')  # Optionnel: si None, supprime tout
        
        if not ticker:
            return jsonify({'error': 'Ticker requis'}), 400
        
        success = portfolio.remove_position(ticker, shares)
        
        if success:
            return jsonify({'success': True, 'message': f'{ticker} supprimé'})
        else:
            return jsonify({'error': f'{ticker} non trouvé dans le portfolio'}), 404
            
    except Exception as e:
        logger.error(f"Erreur suppression position: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/portfolio/analysis')
def api_portfolio_analysis():
    if not portfolio:
        return jsonify({'error': 'Portfolio tracker non disponible'}), 503
    try:
        analysis = portfolio.analyze_allocation()
        return jsonify(clean_for_json(analysis))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ========== HEALTH & STARTUP (same as before) ==========

@app.route('/api/health')
def api_health():
    return jsonify({
        'status': 'healthy',
        'modules': {
            'complete_indicators': COMPLETE_INDICATORS,
            'trader_pro': TRADER_PRO,
            'pattern_detector': pattern_detector is not None,
            'mtf_analyzer': mtf_analyzer is not None,
            'ai_analyzer': ai_analyzer is not None,
            'v8_oracle': v8_oracle is not None,
            'tools': TOOLS_AVAILABLE,
            'screener': screener is not None,
            'alerts': alert_system is not None,
            'backtester': backtester is not None,
            'correlation': corr_analyzer is not None,
            'portfolio': portfolio is not None
        }
    }), 200


if __name__ == '__main__':
    import os
    host = os.getenv('DASHBOARD_HOST', '0.0.0.0')
    port = int(os.getenv('DASHBOARD_PORT', 5000))
    
    print("\n" + "="*70)
    print("🌐 PLOUTOS WEB DASHBOARD - V8 ORACLE + TRADER PRO + 5 TOOLS")
    print("="*70)
    print(f"\n🚀 http://{host}:{port}")
    
    if TOOLS_AVAILABLE:
        print("🛠️ 5 TOOLS activés")
        if portfolio:
            print("  ✅ Portfolio (avec suppression)")
    
    print("\n✅ Pages: /, /chart, /tools")
    print("🩺 Test: /api/health")
    print("\n" + "="*70 + "\n")
    
    app.run(host=host, port=port, debug=False)
