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


# ========== CHART DATA ==========

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
        
        # Calculer TOUS les indicateurs
        if COMPLETE_INDICATORS:
            indicators = calculate_complete_indicators(df)
            signals = get_indicator_signals(df, indicators)
        else:
            indicators = calculate_basic_indicators(df)
            signals = {'overall': {'recommendation': 'HOLD', 'confidence': 50}}
        
        # Quick stats
        quick_stats = generate_quick_stats(df, indicators, signals)
        
        # 🎯 TRADER PRO : Détection de patterns
        patterns = None
        if pattern_detector:
            try:
                patterns = pattern_detector.detect_all_patterns(df)
                logger.info(f"✅ Patterns détectés pour {ticker}")
            except Exception as e:
                logger.error(f"❌ Erreur patterns pour {ticker}: {e}")
        
        # 🚀 NOUVEAU : Générer analyse IA complète
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


# ========== TRADER PRO ROUTES ==========

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


# ========== 🎯 5 NOUVEAUX TOOLS ==========

@app.route('/api/screener', methods=['POST'])
def api_screener():
    """Screener automatique"""
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
    """Backtesting"""
    if not backtester:
        return jsonify({'error': 'Backtester non disponible'}), 503
    
    try:
        data = request.json
        ticker = data.get('ticker', 'AAPL')
        strategy = data.get('strategy', 'rsi_mean_reversion')
        period = data.get('period', '1y')
        params = data.get('params', {})
        
        result = backtester.run(ticker, strategy, period, params)
        return jsonify(clean_for_json(result))
        
    except Exception as e:
        logger.error(f"Erreur backtest: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts/list')
def api_alerts_list():
    """Liste des alertes"""
    if not alert_system:
        return jsonify({'error': 'Alert system non disponible'}), 503
    
    try:
        rules = alert_system.get_rules()
        history = alert_system.get_history(limit=20)
        return jsonify({
            'rules': rules,
            'history': history,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts/add', methods=['POST'])
def api_alerts_add():
    """Ajouter une alerte"""
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
    """Vérifier les alertes"""
    if not alert_system:
        return jsonify({'error': 'Alert system non disponible'}), 503
    
    try:
        triggered = alert_system.check_all()
        return jsonify({
            'triggered': triggered,
            'count': len(triggered),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/correlation/heatmap', methods=['POST'])
def api_correlation_heatmap():
    """Heatmap de corrélations"""
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


@app.route('/api/portfolio/summary')
def api_portfolio_summary():
    """Résumé portfolio"""
    if not portfolio:
        return jsonify({'error': 'Portfolio tracker non disponible'}), 503
    
    try:
        summary = portfolio.get_summary()
        return jsonify(clean_for_json(summary))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/portfolio/add', methods=['POST'])
def api_portfolio_add():
    """Ajouter position"""
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


@app.route('/api/portfolio/analysis')
def api_portfolio_analysis():
    """Analyse allocation"""
    if not portfolio:
        return jsonify({'error': 'Portfolio tracker non disponible'}), 503
    
    try:
        analysis = portfolio.analyze_allocation()
        return jsonify(clean_for_json(analysis))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ========== HELPERS ==========

def calculate_basic_indicators(df: pd.DataFrame) -> dict:
    """Fallback si bibliothèque complète non dispo"""
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


@app.route('/api/ai-chat', methods=['POST'])
def api_ai_chat():
    data = request.json
    message = data.get('message', '').lower()
    ticker = data.get('ticker', '')
    context = data.get('context', {})
    
    response = generate_smart_ai_response(message, ticker, context)
    
    return jsonify({
        'response': response,
        'timestamp': datetime.now().isoformat()
    })


def generate_smart_ai_response(message: str, ticker: str, context: dict) -> str:
    if 'analyse' in message or 'complet' in message or 'détail' in message:
        if ai_analyzer:
            try:
                return ai_analyzer.generate_complete_analysis(ticker, context, None)
            except:
                pass
    
    if 'rsi' in message:
        rsi = context.get('rsi', 50)
        if rsi > 70:
            return f"📈 Le RSI de {ticker} est à {rsi:.1f}, en **sur-achat**. ⚠️ Évitez d'acheter au plus haut."
        elif rsi < 30:
            return f"📉 Le RSI de {ticker} est à {rsi:.1f}, en **sur-vente**. 💡 Opportunité d'achat."
        else:
            return f"🟡 Le RSI de {ticker} est à {rsi:.1f}, en zone neutre."
    
    return f"🤖 Assistant IA Ploutos V8 ! Demandez-moi RSI, MACD, analyse complète pour {ticker}."


@app.route('/api/v8/predict/<ticker>')
def api_v8_predict_single(ticker):
    if not v8_oracle:
        return jsonify({'error': 'V8 Oracle non disponible'}), 503
    
    try:
        result = v8_oracle.predict_multi_horizon(ticker.upper())
        
        if 'error' not in result:
            return jsonify({
                'ticker': ticker.upper(),
                'timestamp': result['timestamp'],
                'predictions': clean_for_json(result['predictions']),
                'ensemble': clean_for_json(result.get('ensemble', {}))
            })
        else:
            return jsonify({'error': result['error']}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
    
    if COMPLETE_INDICATORS:
        print("✅ 50+ indicateurs professionnels")
    
    if TRADER_PRO:
        print("🎯 TRADER PRO activé")
        if pattern_detector:
            print("  ✅ Pattern Detector")
        if mtf_analyzer:
            print("  ✅ Multi-Timeframe Analyzer")
    
    if TOOLS_AVAILABLE:
        print("🛠️ 5 TOOLS activés")
        if screener:
            print("  ✅ Screener")
        if alert_system:
            print("  ✅ Alerts")
        if backtester:
            print("  ✅ Backtester")
        if corr_analyzer:
            print("  ✅ Correlation")
        if portfolio:
            print("  ✅ Portfolio")
    
    if ai_analyzer:
        print("🤖 IA avancée activée")
    
    if v8_oracle:
        print(f"⭐ V8 Oracle: {len(v8_oracle.models)} modèles")
    
    print("\n✅ Pages: /, /chart, /tools")
    print("🩺 Test: /api/health")
    print("\n" + "="*70 + "\n")
    
    app.run(host=host, port=port, debug=False)
