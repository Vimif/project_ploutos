#!/usr/bin/env python3
"""
🌐 PLOUTOS WEB DASHBOARD - V8 ORACLE + TRADER PRO + 5 TOOLS + CHART PRO + PRO ANALYSIS + WATCHLISTS + LIVE TRADING + SIGNALS
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

# 📊 TECHNICAL ANALYZER (pour endpoints /api/chart/*)
try:
    from dashboard.technical_analysis import TechnicalAnalyzer
    TECHNICAL_ANALYZER_AVAILABLE = True
    logger.info("✅ Technical Analyzer chargé (module dashboard)")
except Exception as e:
    TECHNICAL_ANALYZER_AVAILABLE = False
    logger.warning(f"⚠️  Technical Analyzer non disponible: {e}")

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


# ========== ROUTES PAGES HTML ==========

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
    """🔥 Page Live Trading Dashboard"""
    return render_template('live.html')

@app.route('/signals')
def signals_page():
    """🚦 Page Trading Signals Dashboard - Interface graphique avec signaux BUY/SELL"""
    return render_template('trading_signals.html')


# ========== API ENDPOINTS ==========

@app.route('/api/health')
def api_health():
    return jsonify({
        'status': 'healthy',
        'modules': {
            'live_trading': LIVE_TRADING_AVAILABLE,
            'live_watchlists': LIVE_WATCHLISTS_AVAILABLE,
            'alpaca': alpaca_client is not None,
            'technical_analyzer': TECHNICAL_ANALYZER_AVAILABLE,
            'chart_data': TECHNICAL_ANALYZER_AVAILABLE
        }
    }), 200


# ========== ENDPOINTS CHART (TECHNICAL ANALYSIS) ==========

@app.route('/api/chart/<symbol>')
def api_chart_data(symbol):
    """
    📊 Données OHLCV + tous indicateurs techniques pour affichage chart
    Utilisé par l'interface web chart_pro.js
    
    Query params:
        period: '1mo', '3mo', '6mo', '1y', '2y' (défaut: 3mo)
    
    Returns:
        JSON avec OHLCV + indicateurs techniques complets
    """
    if not TECHNICAL_ANALYZER_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Charts indisponibles (TechnicalAnalyzer non chargé)',
            'details': 'Module dashboard.technical_analysis manquant'
        }), 503
    
    try:
        period = request.args.get('period', '3mo')
        
        logger.info(f"📊 Chart request: {symbol} ({period})")
        
        # Créer l'analyseur (intervalle 1d pour charts)
        analyzer = TechnicalAnalyzer(symbol, period=period, interval='1d')
        
        # Récupérer les données brutes
        df = analyzer.df
        
        # Préparer les données OHLCV
        ohlcv_data = []
        for idx, row in df.iterrows():
            ohlcv_data.append({
                'date': idx.strftime('%Y-%m-%d'),
                'timestamp': int(idx.timestamp() * 1000),
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': int(row['Volume'])
            })
        
        # Calculer tous les indicateurs
        indicators = analyzer.get_all_indicators()
        signal = analyzer.generate_signal()
        
        # Calculer indicateurs pour chaque point (pour affichage sur chart)
        sma_20 = analyzer.calculate_sma(20)
        sma_50 = analyzer.calculate_sma(50)
        ema_20 = analyzer.calculate_ema(20)
        rsi = analyzer.calculate_rsi()
        macd_line, signal_line, histogram = analyzer.calculate_macd()
        upper, middle, lower = analyzer.calculate_bollinger_bands()
        
        # Ajouter indicateurs à chaque point OHLCV
        for i, data_point in enumerate(ohlcv_data):
            data_point['sma_20'] = float(sma_20.iloc[i]) if i < len(sma_20) and not pd.isna(sma_20.iloc[i]) else None
            data_point['sma_50'] = float(sma_50.iloc[i]) if i < len(sma_50) and not pd.isna(sma_50.iloc[i]) else None
            data_point['ema_20'] = float(ema_20.iloc[i]) if i < len(ema_20) and not pd.isna(ema_20.iloc[i]) else None
            data_point['rsi'] = float(rsi.iloc[i]) if i < len(rsi) and not pd.isna(rsi.iloc[i]) else None
            data_point['macd'] = float(macd_line.iloc[i]) if i < len(macd_line) and not pd.isna(macd_line.iloc[i]) else None
            data_point['macd_signal'] = float(signal_line.iloc[i]) if i < len(signal_line) and not pd.isna(signal_line.iloc[i]) else None
            data_point['macd_histogram'] = float(histogram.iloc[i]) if i < len(histogram) and not pd.isna(histogram.iloc[i]) else None
            data_point['bb_upper'] = float(upper.iloc[i]) if i < len(upper) and not pd.isna(upper.iloc[i]) else None
            data_point['bb_middle'] = float(middle.iloc[i]) if i < len(middle) and not pd.isna(middle.iloc[i]) else None
            data_point['bb_lower'] = float(lower.iloc[i]) if i < len(lower) and not pd.isna(lower.iloc[i]) else None
        
        return jsonify({
            'success': True,
            'symbol': symbol.upper(),
            'period': period,
            'data': ohlcv_data,
            'indicators': indicators,
            'signal': {
                'signal': signal.signal,
                'strength': signal.strength,
                'trend': signal.trend,
                'confidence': signal.confidence,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'reasons': signal.reasons
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Erreur chart {symbol}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/chart/<symbol>/support-resistance')
def api_chart_support_resistance(symbol):
    """
    🎯 Niveaux de support et résistance automatiques
    Utilisé par l'interface web pour afficher les zones clés
    
    Query params:
        period: '1mo', '3mo', '6mo', '1y', '2y' (défaut: 3mo)
    
    Returns:
        JSON avec niveaux de support et résistance
    """
    if not TECHNICAL_ANALYZER_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Support/Resistance indisponibles',
            'details': 'Module TechnicalAnalyzer manquant'
        }), 503
    
    try:
        period = request.args.get('period', '3mo')
        
        analyzer = TechnicalAnalyzer(symbol, period=period, interval='1d')
        df = analyzer.df
        
        # Détecter les pivots (méthode simplifiée)
        window = 10  # Fenetre pour détection pivots
        
        # Trouver les hauts et bas locaux
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(df) - window):
            # Résistance : prix plus haut que ses voisins
            if df['High'].iloc[i] == df['High'].iloc[i-window:i+window+1].max():
                resistance_levels.append(float(df['High'].iloc[i]))
            
            # Support : prix plus bas que ses voisins
            if df['Low'].iloc[i] == df['Low'].iloc[i-window:i+window+1].min():
                support_levels.append(float(df['Low'].iloc[i]))
        
        # Regrouper les niveaux proches (tolerance 1%)
        def cluster_levels(levels, tolerance=0.01):
            if not levels:
                return []
            
            levels_sorted = sorted(levels)
            clusters = []
            current_cluster = [levels_sorted[0]]
            
            for level in levels_sorted[1:]:
                if abs(level - current_cluster[-1]) / current_cluster[-1] <= tolerance:
                    current_cluster.append(level)
                else:
                    clusters.append(sum(current_cluster) / len(current_cluster))
                    current_cluster = [level]
            
            clusters.append(sum(current_cluster) / len(current_cluster))
            return clusters
        
        resistance_clusters = cluster_levels(resistance_levels)
        support_clusters = cluster_levels(support_levels)
        
        # Garder les 5 niveaux les plus pertinents (proches du prix actuel)
        current_price = float(df['Close'].iloc[-1])
        
        resistance_sorted = sorted(resistance_clusters, key=lambda x: abs(x - current_price))[:5]
        support_sorted = sorted(support_clusters, key=lambda x: abs(x - current_price))[:5]
        
        # Ajouter force (nombre de touches)
        def calculate_strength(level, all_levels):
            tolerance = level * 0.01
            touches = sum(1 for l in all_levels if abs(l - level) <= tolerance)
            return touches
        
        resistance_data = [
            {
                'level': r,
                'strength': calculate_strength(r, resistance_levels),
                'distance_pct': ((r - current_price) / current_price * 100)
            }
            for r in resistance_sorted if r > current_price
        ]
        
        support_data = [
            {
                'level': s,
                'strength': calculate_strength(s, support_levels),
                'distance_pct': ((current_price - s) / current_price * 100)
            }
            for s in support_sorted if s < current_price
        ]
        
        return jsonify({
            'success': True,
            'symbol': symbol.upper(),
            'current_price': current_price,
            'resistance': sorted(resistance_data, key=lambda x: x['level']),
            'support': sorted(support_data, key=lambda x: x['level'], reverse=True)
        })
        
    except Exception as e:
        logger.error(f"❌ Erreur support/resistance {symbol}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    import os
    host = os.getenv('DASHBOARD_HOST', '0.0.0.0')
    port = int(os.getenv('DASHBOARD_PORT', 5000))
    print("\n" + "="*70)
    print("🌐 PLOUTOS - V8 ORACLE + LIVE TRADING + WATCHLISTS + SIGNALS + CHARTS")
    print("="*70)
    print(f"\n🚀 http://{host}:{port}")
    print(f"🔥 Live Trading: http://{host}:{port}/live")
    print(f"🚦 Trading Signals: http://{host}:{port}/signals")
    print(f"📊 Advanced Charts: http://{host}:{port}/chart")
    if LIVE_WATCHLISTS_AVAILABLE:
        print(f"📊 9 Watchlists prédéfinies disponibles")
    if TECHNICAL_ANALYZER_AVAILABLE:
        print(f"✅ Technical Analysis: Endpoints /api/chart/* actifs")
    print("\n" + "="*70 + "\n")
    app.run(host=host, port=port, debug=False)
