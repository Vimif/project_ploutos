#!/usr/bin/env python3
"""
🌐 PLOUTOS WEB DASHBOARD - V8 ORACLE + TRADER PRO + 5 TOOLS + CHART PRO
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

# 📈 CHART TOOLS
try:
    from web.utils.chart_tools import ChartTools
    CHART_TOOLS_AVAILABLE = True
    logger.info("✅ Chart Tools chargés")
except Exception as e:
    CHART_TOOLS_AVAILABLE = False
    logger.error(f"❌ Chart Tools non disponibles: {e}")

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


app = Flask(__name__)
CORS(app)

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

# 📈 Chart Tools
chart_tools = None
if CHART_TOOLS_AVAILABLE:
    chart_tools = ChartTools()
    logger.info("✅ Chart Tools initialisé")

cache = {'account': None, 'positions': None, 'last_update': None, 'chart_cache': {}}
DEFAULT_WATCHLIST = ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'BAC']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chart')
def chart_page():
    return render_template('advanced_chart.html')

@app.route('/tools')
def tools_page():
    return render_template('tools.html')


# ========== 📈 CHART TOOLS ROUTES ==========

@app.route('/api/chart/<ticker>/fibonacci')
def api_fibonacci(ticker):
    if not chart_tools:
        return jsonify({'error': 'Chart tools non disponible'}), 503
    try:
        period = request.args.get('period', '3mo')
        lookback = int(request.args.get('lookback', 90))
        df = yf.download(ticker.upper(), period=period, progress=False)
        if df is None or df.empty:
            return jsonify({'error': 'Aucune donnée'}), 404
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        out = chart_tools.calculate_fibonacci(df, lookback=lookback)
        return jsonify(clean_for_json(out))
    except Exception as e:
        logger.error(f"Erreur Fibonacci {ticker}: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/chart/<ticker>/volume-profile')
def api_volume_profile(ticker):
    if not chart_tools:
        return jsonify({'error': 'Chart tools non disponible'}), 503
    try:
        period = request.args.get('period', '3mo')
        bins = int(request.args.get('bins', 24))
        df = yf.download(ticker.upper(), period=period, progress=False)
        if df is None or df.empty:
            return jsonify({'error': 'Aucune donnée'}), 404
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        out = chart_tools.calculate_volume_profile(df, bins=bins)
        return jsonify(clean_for_json(out))
    except Exception as e:
        logger.error(f"Erreur Volume Profile {ticker}: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/chart/<ticker>/support-resistance')
def api_support_resistance(ticker):
    if not chart_tools:
        return jsonify({'error': 'Chart tools non disponible'}), 503
    try:
        period = request.args.get('period', '3mo')
        window = int(request.args.get('window', 20))
        df = yf.download(ticker.upper(), period=period, progress=False)
        if df is None or df.empty:
            return jsonify({'error': 'Aucune donnée'}), 404
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        out = chart_tools.detect_support_resistance(df, window=window)
        return jsonify(clean_for_json(out))
    except Exception as e:
        logger.error(f"Erreur Support/Resistance {ticker}: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ========== 5 TOOLS ROUTES ==========

@app.route('/api/screener', methods=['POST'])
def api_screener():
    if not screener:
        return jsonify({'error': 'Screener non disponible'}), 503
    try:
        data = request.json or {}
        results = screener.scan(data.get('tickers', DEFAULT_WATCHLIST), period=data.get('period', '3mo'), filters=data.get('filters', {}))
        return jsonify(clean_for_json(results))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    if not backtester:
        return jsonify({'error': 'Backtester non disponible'}), 503
    try:
        data = request.json
        result = backtester.run(data.get('ticker', 'AAPL'), data.get('strategy', 'rsi_mean_reversion'), data.get('period', '1y'), data.get('params', {}))
        return jsonify(clean_for_json(result))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/list')
def api_alerts_list():
    if not alert_system:
        return jsonify({'error': 'Alert system non disponible'}), 503
    try:
        return jsonify({'rules': alert_system.get_rules(), 'history': alert_system.get_history(limit=20), 'timestamp': datetime.now().isoformat()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/add', methods=['POST'])
def api_alerts_add():
    if not alert_system:
        return jsonify({'error': 'Alert system non disponible'}), 503
    try:
        data = request.json
        return jsonify(alert_system.add_rule(ticker=data['ticker'], condition=data['condition'], value=data['value'], name=data.get('name'), channel=data.get('channel', 'all')))
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
        return jsonify(clean_for_json(corr_analyzer.generate_heatmap(data.get('tickers', DEFAULT_WATCHLIST), period=data.get('period', '6mo'))))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/summary')
def api_portfolio_summary():
    if not portfolio:
        return jsonify({'error': 'Portfolio tracker non disponible'}), 503
    try:
        return jsonify(clean_for_json(portfolio.get_summary()))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/add', methods=['POST'])
def api_portfolio_add():
    if not portfolio:
        return jsonify({'error': 'Portfolio tracker non disponible'}), 503
    try:
        data = request.json
        return jsonify(portfolio.add_position(ticker=data['ticker'], shares=data['shares'], avg_price=data['avg_price'], date=data.get('date')))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/remove', methods=['POST'])
def api_portfolio_remove():
    if not portfolio:
        return jsonify({'error': 'Portfolio tracker non disponible'}), 503
    try:
        data = request.json
        ticker = data.get('ticker')
        if not ticker:
            return jsonify({'error': 'Ticker requis'}), 400
        success = portfolio.remove_position(ticker, data.get('shares'))
        if success:
            return jsonify({'success': True, 'message': f'{ticker} supprimé'})
        else:
            return jsonify({'error': f'{ticker} non trouvé'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/analysis')
def api_portfolio_analysis():
    if not portfolio:
        return jsonify({'error': 'Portfolio tracker non disponible'}), 503
    try:
        return jsonify(clean_for_json(portfolio.analyze_allocation()))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def api_health():
    return jsonify({'status': 'healthy', 'modules': {'complete_indicators': COMPLETE_INDICATORS, 'trader_pro': TRADER_PRO, 'pattern_detector': pattern_detector is not None, 'mtf_analyzer': mtf_analyzer is not None, 'ai_analyzer': ai_analyzer is not None, 'v8_oracle': v8_oracle is not None, 'tools': TOOLS_AVAILABLE, 'screener': screener is not None, 'alerts': alert_system is not None, 'backtester': backtester is not None, 'correlation': corr_analyzer is not None, 'portfolio': portfolio is not None, 'chart_tools': chart_tools is not None}}), 200

if __name__ == '__main__':
    import os
    host = os.getenv('DASHBOARD_HOST', '0.0.0.0')
    port = int(os.getenv('DASHBOARD_PORT', 5000))
    print("\n" + "="*70)
    print("🌐 PLOUTOS - V8 + TRADER PRO + 5 TOOLS + CHART PRO")
    print("="*70)
    print(f"\n🚀 http://{host}:{port}")
    if TOOLS_AVAILABLE:
        print("🛠️ 5 TOOLS activés")
    if CHART_TOOLS_AVAILABLE:
        print("📈 CHART PRO: Fibonacci / Volume Profile / S/R")
    print("\n✅ Pages: /, /chart, /tools")
    print("🩺 Test: /api/health")
    print("\n" + "="*70 + "\n")
    app.run(host=host, port=port, debug=False)
