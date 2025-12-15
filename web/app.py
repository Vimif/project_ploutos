#!/usr/bin/env python3
"""
🌐 PLOUTOS WEB DASHBOARD - V8 ORACLE + TRADER PRO + 5 TOOLS + CHART PRO + PRO ANALYSIS + WATCHLISTS
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
    """
    Convertit récursivement les objets Python en types JSON-sérialisables
    Gère : numpy, pandas, dataclasses, enums, etc.
    """
    # Dataclasses
    if is_dataclass(obj) and not isinstance(obj, type):
        return clean_for_json(asdict(obj))
    
    # Enums
    if isinstance(obj, Enum):
        return obj.value
    
    # Booléens numpy (CRITICAL FIX)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    
    # Numpy/Pandas types
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
    
    # Collections
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(item) for item in obj]
    
    # Float/Int standards
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    
    # Fallback
    return obj


app = Flask(__name__)
CORS(app)

# 🔗 Register blueprints
if WATCHLISTS_AVAILABLE:
    app.register_blueprint(watchlists_bp)
    logger.info("✅ Watchlists blueprint enregistré")

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
    """💾 Connexion PostgreSQL avec gestion d'erreurs"""
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


# ========== ACCOUNT & TRADING ENDPOINTS ==========

@app.route('/api/account')
def api_account():
    """💰 Récupère les infos du compte Alpaca"""
    if not alpaca_client:
        return jsonify({
            'mock': True,
            'portfolio_value': 100000,
            'cash': 50000,
            'buying_power': 50000,
            'equity': 100000,
            'profit_loss': 0,
            'profit_loss_pct': 0
        }), 200
    
    try:
        account = alpaca_client.get_account()
        
        if isinstance(account, dict):
            portfolio_value = float(account.get('portfolio_value', 0))
            cash = float(account.get('cash', 0))
            buying_power = float(account.get('buying_power', 0))
            equity = float(account.get('equity', 0))
            last_equity = float(account.get('last_equity', equity))
        else:
            portfolio_value = float(account.portfolio_value)
            cash = float(account.cash)
            buying_power = float(account.buying_power)
            equity = float(account.equity)
            last_equity = float(account.last_equity)
        
        profit_loss = equity - last_equity
        profit_loss_pct = (profit_loss / last_equity * 100) if last_equity > 0 else 0
        
        return jsonify({
            'portfolio_value': portfolio_value,
            'cash': cash,
            'buying_power': buying_power,
            'equity': equity,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct
        })
        
    except Exception as e:
        logger.error(f"❌ Erreur API account: {e}")
        return jsonify({
            'mock': True,
            'error': str(e),
            'portfolio_value': 100000,
            'cash': 50000,
            'buying_power': 50000,
            'equity': 100000
        }), 200


@app.route('/api/positions')
def api_positions():
    """💼 Récupère les positions Alpaca"""
    if not alpaca_client:
        return jsonify({
            'mock': True,
            'positions': [],
            'count': 0
        }), 200
    
    try:
        positions = alpaca_client.get_positions()
        
        result = []
        for pos in positions:
            if isinstance(pos, dict):
                result.append({
                    'symbol': pos.get('symbol', ''),
                    'qty': float(pos.get('qty', 0)),
                    'avg_entry_price': float(pos.get('avg_entry_price', 0)),
                    'current_price': float(pos.get('current_price', 0)),
                    'market_value': float(pos.get('market_value', 0)),
                    'cost_basis': float(pos.get('cost_basis', 0)),
                    'unrealized_pl': float(pos.get('unrealized_pl', 0)),
                    'unrealized_plpc': float(pos.get('unrealized_plpc', 0)) * 100,
                    'side': pos.get('side', 'long')
                })
            else:
                result.append({
                    'symbol': pos.symbol,
                    'qty': float(pos.qty),
                    'avg_entry_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'market_value': float(pos.market_value),
                    'cost_basis': float(pos.cost_basis),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc) * 100,
                    'side': pos.side
                })
        
        return jsonify({'positions': result, 'count': len(result)})
        
    except Exception as e:
        logger.error(f"❌ Erreur API positions: {e}")
        return jsonify({
            'mock': True,
            'error': str(e),
            'positions': [],
            'count': 0
        }), 200


@app.route('/api/trades')
def api_trades():
    """📊 Récupère les trades depuis PostgreSQL"""
    days = int(request.args.get('days', 7))
    
    conn = get_db_connection()
    if not conn:
        return jsonify({
            'mock': True,
            'error': 'PostgreSQL non disponible',
            'trades': [],
            'count': 0,
            'days': days
        }), 200
    
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT 
                id,
                ticker,
                action,
                quantity,
                price,
                timestamp,
                pnl,
                strategy
            FROM trades
            WHERE timestamp >= NOW() - INTERVAL '%s days'
            ORDER BY timestamp DESC
            LIMIT 50
        """
        
        cursor.execute(query, (days,))
        trades = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        result = []
        for trade in trades:
            result.append({
                'id': trade['id'],
                'ticker': trade['ticker'],
                'action': trade['action'],
                'quantity': float(trade['quantity']) if trade['quantity'] else 0,
                'price': float(trade['price']) if trade['price'] else 0,
                'timestamp': trade['timestamp'].isoformat() if trade['timestamp'] else None,
                'pnl': float(trade['pnl']) if trade['pnl'] else 0,
                'strategy': trade['strategy']
            })
        
        return jsonify({
            'trades': result,
            'count': len(result),
            'days': days
        })
        
    except Exception as e:
        logger.error(f"❌ Erreur API trades: {e}", exc_info=True)
        return jsonify({
            'mock': True,
            'error': str(e),
            'trades': [],
            'count': 0
        }), 200


# ========== V8 ORACLE ENDPOINTS ==========

@app.route('/api/v8/predict/<ticker>')
def api_v8_predict(ticker):
    """
    🤖 Prédiction V8 Oracle pour un ticker
    """
    if not v8_oracle:
        return jsonify({
            'error': 'V8 Oracle non disponible',
            'message': 'Les modèles V8 ne sont pas chargés sur ce serveur'
        }), 503
    
    try:
        ticker = ticker.upper()
        result = v8_oracle.predict_multi_horizon(ticker)
        
        if 'error' in result:
            return jsonify(result), 404
        
        logger.info(f"✅ V8 prediction pour {ticker}: {result.get('ensemble', {}).get('prediction', 'N/A')}")
        return jsonify(clean_for_json(result))
        
    except Exception as e:
        logger.error(f"❌ Erreur V8 predict {ticker}: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/v8/batch')
def api_v8_batch():
    """
    📊 Analyse batch V8 Oracle pour plusieurs tickers
    """
    if not v8_oracle:
        return jsonify({
            'error': 'V8 Oracle non disponible'
        }), 503
    
    try:
        tickers_param = request.args.get('tickers', 'AAPL,NVDA,MSFT')
        tickers = [t.strip().upper() for t in tickers_param.split(',')]
        
        logger.info(f"📊 Batch analysis pour {len(tickers)} tickers: {tickers}")
        
        results = {}
        bullish = 0
        bearish = 0
        high_confidence = 0
        
        for ticker in tickers:
            try:
                prediction = v8_oracle.predict_multi_horizon(ticker)
                
                if 'error' not in prediction:
                    results[ticker] = prediction
                    
                    ensemble = prediction.get('ensemble', {})
                    if ensemble.get('prediction') == 'UP':
                        bullish += 1
                    elif ensemble.get('prediction') == 'DOWN':
                        bearish += 1
                    
                    if ensemble.get('confidence', 0) >= 75:
                        high_confidence += 1
                else:
                    results[ticker] = {'error': prediction['error']}
                    
            except Exception as e:
                logger.error(f"❌ Erreur {ticker}: {e}")
                results[ticker] = {'error': str(e)}
        
        summary = {
            'total': len(tickers),
            'bullish': bullish,
            'bearish': bearish,
            'high_confidence_count': high_confidence
        }
        
        return jsonify({
            'tickers': clean_for_json(results),
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Erreur batch: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/v8/recommend/<ticker>')
def api_v8_recommend(ticker):
    """
    🎯 Recommandation de trading V8 avec gestion du risque
    """
    if not v8_oracle:
        return jsonify({'error': 'V8 Oracle non disponible'}), 503
    
    try:
        ticker = ticker.upper()
        risk = request.args.get('risk', 'moderate')
        
        # Seuils de confiance selon profil de risque
        thresholds = {
            'conservative': 80,
            'moderate': 65,
            'aggressive': 50
        }
        
        threshold = thresholds.get(risk, 65)
        
        prediction = v8_oracle.predict_multi_horizon(ticker)
        
        if 'error' in prediction:
            return jsonify(prediction), 404
        
        ensemble = prediction.get('ensemble', {})
        pred = ensemble.get('prediction', 'HOLD')
        conf = ensemble.get('confidence', 0)
        agreement = ensemble.get('agreement', 'WEAK')
        
        # Déterminer l'action
        if conf >= threshold:
            if pred == 'UP':
                action = 'BUY'
                strength = 'STRONG' if agreement == 'STRONG' else 'MODERATE'
            elif pred == 'DOWN':
                action = 'SELL'
                strength = 'STRONG' if agreement == 'STRONG' else 'MODERATE'
            else:
                action = 'HOLD'
                strength = 'WEAK'
        else:
            action = 'HOLD'
            strength = 'WEAK'
        
        return jsonify({
            'ticker': ticker,
            'action': action,
            'strength': strength,
            'prediction': pred,
            'confidence': conf,
            'agreement': agreement,
            'risk_profile': risk,
            'threshold_used': threshold,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Erreur recommendation: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ========== ROUTE CHART PRINCIPALE ==========

@app.route('/api/chart/<ticker>')
def api_chart_data(ticker):
    ticker = ticker.upper()
    period = request.args.get('period', '3mo')
    
    cache_key = f"{ticker}_{period}"
    if cache_key in cache['chart_cache']:
        cached = cache['chart_cache'][cache_key]
        if (datetime.now() - cached['timestamp']).seconds < 300:
            return jsonify(cached['data'])
    
    try:
        df = yf.download(ticker, period=period, progress=False)
        
        if df.empty:
            return jsonify({'error': 'Aucune donnée'}), 404
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if COMPLETE_INDICATORS:
            indicators = calculate_complete_indicators(df)
            signals = get_indicator_signals(df, indicators)
        else:
            indicators = calculate_basic_indicators(df)
            signals = {'overall': {'recommendation': 'HOLD', 'confidence': 50}}
        
        quick_stats = generate_quick_stats(df, indicators, signals)
        
        patterns = None
        if pattern_detector:
            try:
                patterns = pattern_detector.detect_all_patterns(df)
                logger.info(f"✅ Patterns détectés pour {ticker}")
            except Exception as e:
                logger.error(f"❌ Erreur patterns pour {ticker}: {e}")
        
        ai_analysis = None
        if ai_analyzer:
            try:
                v8_predictions = None
                if v8_oracle:
                    try:
                        v8_result = v8_oracle.predict_multi_horizon(ticker)
                        if 'error' not in v8_result:
                            v8_predictions = v8_result
                    except:
                        pass
                
                data_for_ai = {
                    'quick_stats': quick_stats,
                    'signals': signals,
                    'indicators': indicators
                }
                ai_analysis = ai_analyzer.generate_complete_analysis(
                    ticker, data_for_ai, v8_predictions
                )
            except Exception as e:
                logger.error(f"Erreur génération IA: {e}")
        
        response = {
            'ticker': ticker,
            'period': period,
            'current_price': clean_for_json(df['Close'].iloc[-1]),
            'dates': [d.isoformat() for d in df.index],
            'open': clean_for_json(df['Open'].values),
            'high': clean_for_json(df['High'].values),
            'low': clean_for_json(df['Low'].values),
            'close': clean_for_json(df['Close'].values),
            'volume': clean_for_json(df['Volume'].values),
            'indicators': clean_for_json(indicators),
            'signals': clean_for_json(signals),
            'quick_stats': clean_for_json(quick_stats),
            'patterns': clean_for_json(patterns) if patterns else None,
            'ai_analysis': ai_analysis,
            'timestamp': datetime.now().isoformat()
        }
        
        cache['chart_cache'][cache_key] = {
            'data': response,
            'timestamp': datetime.now()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Erreur chart {ticker}: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ========== PRO TECHNICAL ANALYSIS ==========

@app.route('/api/pro-analysis/<ticker>')
def api_pro_analysis(ticker):
    if not pro_analyzer:
        return jsonify({'error': 'Pro analyzer non disponible'}), 503
    
    try:
        ticker = ticker.upper()
        period = request.args.get('period', '1y')
        
        df = yf.download(ticker, period=period, progress=False)
        
        if df.empty:
            return jsonify({'error': 'Aucune donnée'}), 404
        
        report = pro_analyzer.analyze(df, ticker=ticker)
        result = clean_for_json(report)
        
        logger.info(f"✅ Pro analysis pour {ticker}: {report.overall_signal} ({report.confidence:.0f}%)")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Erreur pro-analysis {ticker}: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ========== CHART TOOLS ==========

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


# ========== TRADER PRO ==========

@app.route('/api/patterns/<ticker>')
def api_patterns(ticker):
    if not pattern_detector:
        return jsonify({'error': 'Pattern detector non disponible'}), 503
    try:
        period = request.args.get('period', '3mo')
        df = yf.download(ticker.upper(), period=period, progress=False)
        if df.empty:
            return jsonify({'error': 'Aucune donnée'}), 404
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        patterns = pattern_detector.detect_all_patterns(df)
        return jsonify(clean_for_json(patterns))
    except Exception as e:
        logger.error(f"Erreur patterns API: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/mtf/<ticker>')
def api_multi_timeframe(ticker):
    if not mtf_analyzer:
        return jsonify({'error': 'MTF analyzer non disponible'}), 503
    try:
        analysis = mtf_analyzer.analyze_multi_timeframe(ticker.upper())
        return jsonify(clean_for_json(analysis))
    except Exception as e:
        logger.error(f"Erreur MTF API: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ========== 5 TOOLS ==========

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


# ========== HELPERS ==========

def calculate_basic_indicators(df: pd.DataFrame) -> dict:
    import ta
    indicators = {}
    for p in [20, 50, 200]:
        if len(df) >= p:
            indicators[f'sma_{p}'] = ta.trend.sma_indicator(df['Close'], window=p).tolist()
    if len(df) >= 14:
        indicators['rsi'] = ta.momentum.rsi(df['Close'], window=14).tolist()
    if len(df) >= 26:
        macd = ta.trend.MACD(df['Close'])
        indicators['macd'] = macd.macd().tolist()
        indicators['macd_signal'] = macd.macd_signal().tolist()
        indicators['macd_hist'] = macd.macd_diff().tolist()
    if len(df) >= 20:
        bb = ta.volatility.BollingerBands(df['Close'])
        indicators['bb_upper'] = bb.bollinger_hband().tolist()
        indicators['bb_middle'] = bb.bollinger_mavg().tolist()
        indicators['bb_lower'] = bb.bollinger_lband().tolist()
    if len(df) >= 14:
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        indicators['stoch_k'] = stoch.stoch().tolist()
        indicators['stoch_d'] = stoch.stoch_signal().tolist()
        indicators['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close']).tolist()
        indicators['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close']).tolist()
    indicators['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume']).tolist()
    return indicators

def generate_quick_stats(df: pd.DataFrame, indicators: dict, signals: dict) -> dict:
    def safe_get(arr, default=0):
        if not arr or len(arr) == 0:
            return default
        val = arr[-1]
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return default
        return float(val)
    
    current_price = float(df['Close'].iloc[-1])
    prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else current_price
    change = current_price - prev_close
    change_pct = (change / prev_close * 100) if prev_close != 0 else 0
    current_volume = float(df['Volume'].iloc[-1])
    avg_volume = float(df['Volume'].rolling(20).mean().iloc[-1]) if len(df) >= 20 else current_volume
    volume_ratio = (current_volume / avg_volume) if avg_volume > 0 else 1.0
    
    return {
        'price': current_price,
        'change': change,
        'change_pct': change_pct,
        'high_52w': float(df['High'].tail(252).max()) if len(df) >= 252 else float(df['High'].max()),
        'low_52w': float(df['Low'].tail(252).min()) if len(df) >= 252 else float(df['Low'].min()),
        'rsi': safe_get(indicators.get('rsi', []), 50),
        'adx': safe_get(indicators.get('adx', []), 20),
        'volume': current_volume,
        'volume_ratio': volume_ratio,
        'recommendation': signals.get('overall', {}).get('recommendation', 'HOLD'),
        'confidence': signals.get('overall', {}).get('confidence', 50)
    }


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
            'portfolio': portfolio is not None, 
            'chart_tools': chart_tools is not None,
            'pro_analyzer': pro_analyzer is not None,
            'watchlists': WATCHLISTS_AVAILABLE,
            'alpaca': alpaca_client is not None,
            'database': DB_AVAILABLE
        }
    }), 200

if __name__ == '__main__':
    import os
    host = os.getenv('DASHBOARD_HOST', '0.0.0.0')
    port = int(os.getenv('DASHBOARD_PORT', 5000))
    print("\n" + "="*70)
    print("🌐 PLOUTOS - V8 ORACLE + TRADER PRO + 5 TOOLS + WATCHLISTS")
    print("="*70)
    print(f"\n🚀 http://{host}:{port}")
    if v8_oracle:
        print("🤖 V8 ORACLE: /api/v8/predict/<ticker>, /api/v8/batch")
    if TOOLS_AVAILABLE:
        print("🛠️ 5 TOOLS activés")
    if CHART_TOOLS_AVAILABLE:
        print("📈 CHART PRO: Fibonacci / Volume Profile / S/R")
    if PRO_ANALYZER_AVAILABLE:
        print("🎯 PRO ANALYSIS: 5 indicateurs clés + divergences")
    if WATCHLISTS_AVAILABLE:
        print("📊 WATCHLISTS: 20 listes (US + FR + International)")
    print("\n✅ Pages: /, /chart, /tools")
    print("🩺 Health: /api/health")
    print("📊 Watchlists: /api/watchlists")
    print("💰 Trading: /api/account, /api/positions, /api/trades")
    print("🤖 V8 Oracle: /api/v8/predict/<TICKER>, /api/v8/batch?tickers=...")
    print("\n" + "="*70 + "\n")
    app.run(host=host, port=port, debug=False)
