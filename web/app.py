#!/usr/bin/env python3
"""
🌐 PLOUTOS WEB DASHBOARD - V8 ORACLE + TRADER PRO + 5 TOOLS + CHART PRO + PRO ANALYSIS + WATCHLISTS + LIVE TRADING + SIGNALS + SCALPER
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

# 🎯 PRO ANALYSIS BLUEPRINT
try:
    from web.routes.pro_analysis import pro_analysis_bp
    PRO_ANALYSIS_BP_AVAILABLE = True
    logger.info("✅ Pro Analysis blueprint chargé")
except Exception as e:
    PRO_ANALYSIS_BP_AVAILABLE = False
    logger.error(f"❌ Pro Analysis blueprint non disponible: {e}")

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

if WATCHLISTS_AVAILABLE:
    app.register_blueprint(watchlists_bp)
    logger.info("✅ Watchlists blueprint enregistré")

if LIVE_TRADING_AVAILABLE:
    app.register_blueprint(live_bp)
    logger.info("✅ Live Trading blueprint enregistré")

if LIVE_WATCHLISTS_AVAILABLE:
    app.register_blueprint(live_watchlists_bp)
    logger.info("✅ Live Watchlists blueprint enregistré")

if PRO_ANALYSIS_BP_AVAILABLE:
    app.register_blueprint(pro_analysis_bp)
    logger.info("✅ Pro Analysis blueprint enregistré")

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

@app.route('/live')
def live_page():
    return render_template('live.html')

@app.route('/signals')
def signals_page():
    return render_template('trading_signals.html')

@app.route('/scalper')
def scalper_page():
    return render_template('scalper.html')


@app.route('/api/health')
def api_health():
    return jsonify({
        'status': 'healthy',
        'modules': {
            'live_trading': LIVE_TRADING_AVAILABLE,
            'live_watchlists': LIVE_WATCHLISTS_AVAILABLE,
            'alpaca': alpaca_client is not None,
            'technical_analyzer': TECHNICAL_ANALYZER_AVAILABLE,
            'chart_data': TECHNICAL_ANALYZER_AVAILABLE,
            'mtf_analyzer': TRADER_PRO and mtf_analyzer is not None,
            'pro_analysis': PRO_ANALYSIS_BP_AVAILABLE
        }
    }), 200


@app.route('/api/chart/<symbol>')
def api_chart_data(symbol):
    """📊 Données OHLCV + tous indicateurs techniques pour affichage chart"""
    if not TECHNICAL_ANALYZER_AVAILABLE:
        return jsonify({'success': False, 'error': 'Charts indisponibles'}), 503
    
    try:
        period = request.args.get('period', '3mo')
        analyzer = TechnicalAnalyzer(symbol, period=period, interval='1d')
        df = analyzer.df
        
        if len(df) < 2:
            return jsonify({'success': False, 'error': 'Données insuffisantes'}), 400
        
        current_price = float(df['Close'].iloc[-1])
        previous_close = float(df['Close'].iloc[-2])
        price_change_24h = current_price - previous_close
        price_change_pct = (price_change_24h / previous_close * 100) if previous_close > 0 else 0
        volume_24h = int(df['Volume'].iloc[-1])
        high_24h = float(df['High'].iloc[-1])
        low_24h = float(df['Low'].iloc[-1])
        
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
        
        indicators = analyzer.get_all_indicators()
        signal = analyzer.generate_signal()
        
        sma_20 = analyzer.calculate_sma(20)
        sma_50 = analyzer.calculate_sma(50)
        sma_200 = analyzer.calculate_sma(200)
        ema_12 = analyzer.calculate_ema(12)
        ema_20 = analyzer.calculate_ema(20)
        ema_26 = analyzer.calculate_ema(26)
        rsi = analyzer.calculate_rsi()
        macd_line, signal_line, histogram = analyzer.calculate_macd()
        upper, middle, lower = analyzer.calculate_bollinger_bands()
        
        # Calcul sécurisé des indicateurs ta-lib
        import ta as ta_lib
        
        def safe_indicator_array(calc_func, default=[]):
            """Wrapper sécurisé pour indicateurs ta-lib"""
            try:
                result = calc_func()
                return [float(v) if not pd.isna(v) else None for v in result]
            except (IndexError, ValueError, Exception) as e:
                logger.warning(f"⚠️  Indicateur ignoré (données insuffisantes): {e}")
                return default
        
        stoch = ta_lib.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        adx_calc = ta_lib.trend.ADXIndicator(df['High'], df['Low'], df['Close'])
        atr_calc = ta_lib.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'])
        obv_calc = ta_lib.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume'])
        
        indicators_arrays = {
            'sma_20': [float(v) if not pd.isna(v) else None for v in sma_20],
            'sma_50': [float(v) if not pd.isna(v) else None for v in sma_50],
            'sma_200': [float(v) if not pd.isna(v) else None for v in sma_200] if len(sma_200) > 0 else [],
            'ema_12': [float(v) if not pd.isna(v) else None for v in ema_12],
            'ema_20': [float(v) if not pd.isna(v) else None for v in ema_20],
            'ema_26': [float(v) if not pd.isna(v) else None for v in ema_26],
            'rsi': [float(v) if not pd.isna(v) else None for v in rsi],
            'macd': [float(v) if not pd.isna(v) else None for v in macd_line],
            'macd_signal': [float(v) if not pd.isna(v) else None for v in signal_line],
            'macd_hist': [float(v) if not pd.isna(v) else None for v in histogram],
            'bb_upper': [float(v) if not pd.isna(v) else None for v in upper],
            'bb_middle': [float(v) if not pd.isna(v) else None for v in middle],
            'bb_lower': [float(v) if not pd.isna(v) else None for v in lower],
            'stoch_k': safe_indicator_array(stoch.stoch),
            'stoch_d': safe_indicator_array(stoch.stoch_signal),
            'adx': safe_indicator_array(adx_calc.adx),
            'atr': safe_indicator_array(atr_calc.average_true_range),
            'obv': safe_indicator_array(obv_calc.on_balance_volume)
        }
        
        rsi_current = float(rsi.iloc[-1]) if len(rsi) > 0 and not pd.isna(rsi.iloc[-1]) else 50.0
        
        if signal.signal == 'BUY':
            recommendation = 'STRONG BUY' if signal.strength > 70 else 'BUY'
        elif signal.signal == 'SELL':
            recommendation = 'STRONG SELL' if signal.strength > 70 else 'SELL'
        else:
            recommendation = 'HOLD'
        
        quick_stats = {
            'price': current_price,
            'change': price_change_24h,
            'change_pct': price_change_pct,
            'volume': volume_24h,
            'rsi': rsi_current,
            'adx': indicators.get('adx', 0),
            'recommendation': recommendation,
            'confidence': signal.confidence
        }
        
        return jsonify({
            'success': True,
            'symbol': symbol.upper(),
            'ticker': symbol.upper(),
            'period': period,
            'current_price': current_price,
            'price_change_24h': price_change_24h,
            'price_change_pct': price_change_pct,
            'volume_24h': volume_24h,
            'high_24h': high_24h,
            'low_24h': low_24h,
            'stats': {'price': current_price, 'change': price_change_24h, 'change_pct': price_change_pct, 'volume': volume_24h, 'high': high_24h, 'low': low_24h, 'open': float(df['Open'].iloc[-1])},
            'quick_stats': quick_stats,
            'dates': [d['date'] for d in ohlcv_data],
            'open': [d['open'] for d in ohlcv_data],
            'high': [d['high'] for d in ohlcv_data],
            'low': [d['low'] for d in ohlcv_data],
            'close': [d['close'] for d in ohlcv_data],
            'data': ohlcv_data,
            'indicators': indicators_arrays,
            'signal': {'signal': signal.signal, 'strength': signal.strength, 'trend': signal.trend, 'confidence': signal.confidence, 'entry_price': signal.entry_price, 'stop_loss': signal.stop_loss, 'take_profit': signal.take_profit, 'reasons': signal.reasons},
            'signals': {}
        })
    except Exception as e:
        logger.error(f"❌ Erreur chart {symbol}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/chart/<symbol>/support-resistance')
def api_chart_support_resistance(symbol):
    """🎯 Niveaux de support et résistance automatiques"""
    if not TECHNICAL_ANALYZER_AVAILABLE:
        return jsonify({'success': False, 'error': 'S/R indisponibles'}), 503
    try:
        period = request.args.get('period', '3mo')
        analyzer = TechnicalAnalyzer(symbol, period=period, interval='1d')
        df = analyzer.df
        window = 10
        resistance_levels = []
        support_levels = []
        for i in range(window, len(df) - window):
            if df['High'].iloc[i] == df['High'].iloc[i-window:i+window+1].max():
                resistance_levels.append(float(df['High'].iloc[i]))
            if df['Low'].iloc[i] == df['Low'].iloc[i-window:i+window+1].min():
                support_levels.append(float(df['Low'].iloc[i]))
        def cluster_levels(levels, tolerance=0.01):
            if not levels: return []
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
        current_price = float(df['Close'].iloc[-1])
        resistance_sorted = sorted(resistance_clusters, key=lambda x: abs(x - current_price))[:5]
        support_sorted = sorted(support_clusters, key=lambda x: abs(x - current_price))[:5]
        def calculate_strength(level, all_levels):
            tolerance = level * 0.01
            return sum(1 for l in all_levels if abs(l - level) <= tolerance)
        resistance_data = [{'level': r, 'strength': calculate_strength(r, resistance_levels), 'distance_pct': ((r - current_price) / current_price * 100)} for r in resistance_sorted if r > current_price]
        support_data = [{'level': s, 'strength': calculate_strength(s, support_levels), 'distance_pct': ((current_price - s) / current_price * 100)} for s in support_sorted if s < current_price]
        return jsonify({'success': True, 'symbol': symbol.upper(), 'current_price': current_price, 'resistance': sorted(resistance_data, key=lambda x: x['level']), 'support': sorted(support_data, key=lambda x: x['level'], reverse=True)})
    except Exception as e:
        logger.error(f"❌ S/R error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/chart/<symbol>/fibonacci')
def api_chart_fibonacci(symbol):
    """📐 Niveaux de retracement Fibonacci automatiques"""
    if not TECHNICAL_ANALYZER_AVAILABLE:
        return jsonify({'success': False, 'error': 'Fibonacci indisponible'}), 503
    
    try:
        period = request.args.get('period', '3mo')
        analyzer = TechnicalAnalyzer(symbol, period=period, interval='1d')
        df = analyzer.df
        swing_high = float(df['High'].max())
        swing_low = float(df['Low'].min())
        current_price = float(df['Close'].iloc[-1])
        fib_levels = {
            '0.0%': swing_high,
            '23.6%': swing_high - (swing_high - swing_low) * 0.236,
            '38.2%': swing_high - (swing_high - swing_low) * 0.382,
            '50.0%': swing_high - (swing_high - swing_low) * 0.5,
            '61.8%': swing_high - (swing_high - swing_low) * 0.618,
            '78.6%': swing_high - (swing_high - swing_low) * 0.786,
            '100.0%': swing_low
        }
        fib_extensions = {
            '161.8%': swing_low - (swing_high - swing_low) * 0.618,
            '261.8%': swing_low - (swing_high - swing_low) * 1.618
        }
        return jsonify({
            'success': True,
            'symbol': symbol.upper(),
            'current_price': current_price,
            'swing_high': swing_high,
            'swing_low': swing_low,
            'range': swing_high - swing_low,
            'retracements': fib_levels,
            'extensions': fib_extensions
        })
    except Exception as e:
        logger.error(f"❌ Fibonacci error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/chart/<symbol>/volume-profile')
def api_chart_volume_profile(symbol):
    """📊 Volume Profile (distribution du volume par niveau de prix)"""
    if not TECHNICAL_ANALYZER_AVAILABLE:
        return jsonify({'success': False, 'error': 'Volume Profile indisponible'}), 503
    
    try:
        period = request.args.get('period', '3mo')
        analyzer = TechnicalAnalyzer(symbol, period=period, interval='1d')
        df = analyzer.df
        num_bins = 50
        price_min = float(df['Low'].min())
        price_max = float(df['High'].max())
        bin_size = (price_max - price_min) / num_bins
        volume_profile = {}
        for i in range(num_bins):
            bin_start = price_min + (i * bin_size)
            bin_end = bin_start + bin_size
            bin_center = (bin_start + bin_end) / 2
            vol_in_bin = df[((df['High'] >= bin_start) & (df['Low'] <= bin_end))]['Volume'].sum()
            if vol_in_bin > 0:
                volume_profile[f"{bin_center:.2f}"] = int(vol_in_bin)
        if volume_profile:
            poc_price = max(volume_profile, key=volume_profile.get)
            poc_volume = volume_profile[poc_price]
        else:
            poc_price = None
            poc_volume = 0
        total_volume = sum(volume_profile.values())
        sorted_bins = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
        value_area_volume = 0
        value_area_prices = []
        for price, vol in sorted_bins:
            value_area_volume += vol
            value_area_prices.append(float(price))
            if value_area_volume >= total_volume * 0.7:
                break
        vah = max(value_area_prices) if value_area_prices else None
        val = min(value_area_prices) if value_area_prices else None
        return jsonify({
            'success': True,
            'symbol': symbol.upper(),
            'profile': volume_profile,
            'poc': {'price': float(poc_price) if poc_price else None, 'volume': poc_volume},
            'value_area': {'high': vah, 'low': val},
            'total_volume': total_volume
        })
    except Exception as e:
        logger.error(f"❌ Volume Profile error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/mtf/<symbol>')
def api_mtf_analysis(symbol):
    """🔍 Multi-TimeFrame Analysis"""
    if TRADER_PRO and mtf_analyzer:
        try:
            result = mtf_analyzer.analyze(symbol)
            return jsonify({'success': True, 'data': clean_for_json(result)})
        except Exception as e:
            logger.error(f"❌ MTF Analyzer error: {e}")
    if not TECHNICAL_ANALYZER_AVAILABLE:
        return jsonify({'success': False, 'error': 'MTF indisponible'}), 503
    try:
        timeframes = [
            {'period': '1mo', 'interval': '1h', 'name': '1H'},
            {'period': '3mo', 'interval': '4h', 'name': '4H'},
            {'period': '6mo', 'interval': '1d', 'name': '1D'},
            {'period': '1y', 'interval': '1wk', 'name': '1W'}
        ]
        results = {}
        signals_count = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        for tf in timeframes:
            try:
                analyzer = TechnicalAnalyzer(symbol, period=tf['period'], interval=tf['interval'])
                signal = analyzer.generate_signal()
                results[tf['name']] = {'signal': signal.signal, 'strength': signal.strength, 'trend': signal.trend, 'confidence': signal.confidence}
                signals_count[signal.signal] += 1
            except Exception as e:
                logger.warning(f"⚠️ MTF {tf['name']} error: {e}")
                results[tf['name']] = {'error': str(e)}
        if signals_count['BUY'] > signals_count['SELL']:
            consensus = 'BUY'
            consensus_strength = (signals_count['BUY'] / len(timeframes)) * 100
        elif signals_count['SELL'] > signals_count['BUY']:
            consensus = 'SELL'
            consensus_strength = (signals_count['SELL'] / len(timeframes)) * 100
        else:
            consensus = 'HOLD'
            consensus_strength = (signals_count['HOLD'] / len(timeframes)) * 100
        return jsonify({'success': True, 'symbol': symbol.upper(), 'timeframes': results, 'consensus': {'signal': consensus, 'strength': round(consensus_strength, 1), 'buy_count': signals_count['BUY'], 'sell_count': signals_count['SELL'], 'hold_count': signals_count['HOLD']}})
    except Exception as e:
        logger.error(f"❌ MTF error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    import os
    host = os.getenv('DASHBOARD_HOST', '0.0.0.0')
    port = int(os.getenv('DASHBOARD_PORT', 5000))
    print("\n" + "="*70)
    print("🌐 PLOUTOS - V8 ORACLE + LIVE TRADING + WATCHLISTS + SIGNALS + CHARTS + SCALPER")
    print("="*70)
    print(f"\n🚀 http://{host}:{port}")
    print(f"🔥 Live Trading: http://{host}:{port}/live")
    print(f"🚦 Trading Signals: http://{host}:{port}/signals")
    print(f"📊 Advanced Charts: http://{host}:{port}/chart")
    print(f"⚡ Scalper Pro: http://{host}:{port}/scalper")
    if LIVE_WATCHLISTS_AVAILABLE:
        print(f"📊 9 Watchlists prédéfinies disponibles")
    if TECHNICAL_ANALYZER_AVAILABLE:
        print(f"✅ Technical Analysis: /api/chart/* + /api/mtf/* actifs")
    if PRO_ANALYSIS_BP_AVAILABLE:
        print(f"✅ Pro Analysis: /api/pro-analysis/* actif")
    print("\n" + "="*70 + "\n")
    app.run(host=host, port=port, debug=False)
