#!/usr/bin/env python3
"""
🌐 PLOUTOS WEB DASHBOARD - V8 ORACLE EDITION
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

# Import modules Ploutos
try:
    from trading.alpaca_client import AlpacaClient
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

try:
    from core.self_improvement import SelfImprovementEngine
    SELF_IMPROVEMENT_AVAILABLE = True
except ImportError:
    SELF_IMPROVEMENT_AVAILABLE = False

try:
    from src.models.v8_oracle_ensemble import V8OracleEnsemble
    V8_ORACLE_AVAILABLE = True
except ImportError:
    V8_ORACLE_AVAILABLE = False

try:
    from src.models.v7_predictor import V7Predictor
    V7_AVAILABLE = True
except ImportError:
    V7_AVAILABLE = False


def clean_for_json(obj):
    """
    Nettoie les valeurs pour JSON (remplace NaN/Infinity par None)
    """
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


def convert_to_native_python(obj):
    """Alias pour rétro-compatibilité"""
    return clean_for_json(obj)

# Setup
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation clients
alpaca_client = None
if ALPACA_AVAILABLE:
    try:
        alpaca_client = AlpacaClient(paper_trading=True)
    except Exception as e:
        logger.warning(f"⚠️  Alpaca non disponible: {e}")

v8_oracle = None
if V8_ORACLE_AVAILABLE:
    try:
        v8_oracle = V8OracleEnsemble()
        if v8_oracle.load_models():
            logger.info("✅ V8 Oracle Ensemble chargé")
        else:
            v8_oracle = None
    except Exception as e:
        logger.warning(f"⚠️  Erreur V8: {e}")
        v8_oracle = None

v7_fallback = None
if not v8_oracle and V7_AVAILABLE:
    try:
        v7_fallback = V7Predictor()
        if v7_fallback.load("momentum"):
            logger.info("✅ V7 Fallback chargé")
        else:
            v7_fallback = None
    except Exception as e:
        logger.warning(f"⚠️  Erreur V7: {e}")
        v7_fallback = None

cache = {
    'account': None,
    'positions': None,
    'trades': None,
    'improvement_report': None,
    'v8_predictions_cache': None,
    'last_update': None,
    'v8_last_update': None,
    'chart_cache': {}
}


# ========== ROUTES HTML ==========

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chart')
def chart_page():
    return render_template('advanced_chart.html')


# ========== CHART DATA ENDPOINTS ==========

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
            return jsonify({'error': 'Aucune donnée trouvée'}), 404
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        indicators = calculate_all_indicators(df)
        analysis = perform_technical_analysis(df, indicators)
        
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
            'analysis': clean_for_json(analysis),
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


def calculate_all_indicators(df: pd.DataFrame) -> dict:
    indicators = {}
    
    # SMA
    indicators['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20).tolist()
    indicators['sma_50'] = ta.trend.sma_indicator(df['Close'], window=50).tolist()
    indicators['sma_200'] = ta.trend.sma_indicator(df['Close'], window=200).tolist()
    
    # EMA
    indicators['ema_12'] = ta.trend.ema_indicator(df['Close'], window=12).tolist()
    indicators['ema_26'] = ta.trend.ema_indicator(df['Close'], window=26).tolist()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Close'])
    indicators['bb_upper'] = bb.bollinger_hband().tolist()
    indicators['bb_middle'] = bb.bollinger_mavg().tolist()
    indicators['bb_lower'] = bb.bollinger_lband().tolist()
    
    # RSI
    indicators['rsi'] = ta.momentum.rsi(df['Close'], window=14).tolist()
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    indicators['macd'] = macd.macd().tolist()
    indicators['macd_signal'] = macd.macd_signal().tolist()
    indicators['macd_hist'] = macd.macd_diff().tolist()
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
    indicators['stoch_k'] = stoch.stoch().tolist()
    indicators['stoch_d'] = stoch.stoch_signal().tolist()
    
    # ATR
    indicators['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close']).tolist()
    
    # Volume
    indicators['volume_sma'] = df['Volume'].rolling(window=20).mean().tolist()
    
    return indicators


def perform_technical_analysis(df: pd.DataFrame, indicators: dict) -> dict:
    last_close = df['Close'].iloc[-1]
    
    # Safe get with NaN check
    def safe_get(arr, default=50):
        val = arr[-1]
        return float(val) if not (np.isnan(val) or np.isinf(val)) else default
    
    last_rsi = safe_get(indicators['rsi'])
    last_macd = safe_get(indicators['macd'], 0)
    last_macd_signal = safe_get(indicators['macd_signal'], 0)
    sma_20 = safe_get(indicators['sma_20'], last_close)
    sma_50 = safe_get(indicators['sma_50'], last_close)
    
    # Tendance
    if last_close > sma_20 and last_close > sma_50:
        trend = 'BULLISH'
    elif last_close < sma_20 and last_close < sma_50:
        trend = 'BEARISH'
    else:
        trend = 'NEUTRAL'
    
    # MACD
    macd_signal = 'BULLISH' if last_macd > last_macd_signal else 'BEARISH'
    macd_crossover = abs(last_macd - last_macd_signal) < 0.5
    
    # Volatilité
    bb_upper = safe_get(indicators['bb_upper'], last_close * 1.02)
    bb_lower = safe_get(indicators['bb_lower'], last_close * 0.98)
    bb_width = ((bb_upper - bb_lower) / last_close) * 100 if last_close > 0 else 0
    
    if bb_width > 5:
        volatility = 'HIGH'
    elif bb_width < 2:
        volatility = 'LOW'
    else:
        volatility = 'MEDIUM'
    
    # Signal global
    bullish_signals = 0
    bearish_signals = 0
    
    if trend == 'BULLISH': bullish_signals += 1
    if trend == 'BEARISH': bearish_signals += 1
    if last_rsi < 30: bullish_signals += 1
    if last_rsi > 70: bearish_signals += 1
    if macd_signal == 'BULLISH': bullish_signals += 1
    if macd_signal == 'BEARISH': bearish_signals += 1
    
    total_signals = bullish_signals + bearish_signals
    
    if bullish_signals > bearish_signals:
        overall_signal = 'BUY'
        confidence = (bullish_signals / total_signals) * 100 if total_signals > 0 else 50
    elif bearish_signals > bullish_signals:
        overall_signal = 'SELL'
        confidence = (bearish_signals / total_signals) * 100 if total_signals > 0 else 50
    else:
        overall_signal = 'HOLD'
        confidence = 50
    
    return {
        'trend': trend,
        'rsi': float(last_rsi),
        'macd_signal': macd_signal,
        'macd_crossover': bool(macd_crossover),
        'volatility': volatility,
        'bb_width': float(bb_width),
        'overall_signal': overall_signal,
        'confidence': float(confidence),
        'bullish_signals': int(bullish_signals),
        'bearish_signals': int(bearish_signals)
    }


@app.route('/api/ai-chat', methods=['POST'])
def api_ai_chat():
    data = request.json
    message = data.get('message', '').lower()
    ticker = data.get('ticker', '')
    context = data.get('context', {})
    
    response = generate_ai_response(message, ticker, context)
    
    return jsonify({
        'response': response,
        'timestamp': datetime.now().isoformat()
    })


def generate_ai_response(message: str, ticker: str, context: dict) -> str:
    if 'rsi' in message:
        rsi = context.get('rsi', 50)
        if rsi > 70:
            return f"📈 Le RSI de {ticker} est à {rsi:.1f}, en zone de **sur-achat**. Cela signifie que le titre a beaucoup monté récemment et pourrait connaître une correction à court terme. Attention à ne pas acheter au plus haut !"
        elif rsi < 30:
            return f"📉 Le RSI de {ticker} est à {rsi:.1f}, en zone de **sur-vente**. Le titre a beaucoup baissé et pourrait rebondir bientôt. C'est potentiellement une opportunité d'achat si la tendance globale est positive."
        else:
            return f"🟡 Le RSI de {ticker} est à {rsi:.1f}, en zone **neutre**. Le momentum n'est ni extrêmement haussier ni baissier. Attendez un signal plus clair avant d'agir."
    
    elif 'macd' in message:
        signal = context.get('macd_signal', 'NEUTRAL')
        crossover = context.get('macd_crossover', False)
        
        if crossover:
            return f"⚡ Un **croisement MACD** vient d'être détecté sur {ticker} ! Signal {signal}. Les croisements MACD sont des signaux forts de changement de tendance."
        else:
            return f"📊 Le MACD de {ticker} indique un signal {signal}. Pas de croisement récent détecté."
    
    elif 'tendance' in message or 'trend' in message:
        trend = context.get('trend', 'NEUTRAL')
        
        if trend == 'BULLISH':
            return f"🟢 {ticker} est en **tendance haussière** ! Le prix est au-dessus des moyennes mobiles. Les acheteurs contrôlent le marché."
        elif trend == 'BEARISH':
            return f"🔴 {ticker} est en **tendance baissière**. Le prix est sous les moyennes mobiles. Les vendeurs dominent."
        else:
            return f"🟡 {ticker} est en **consolidation**. Pas de direction claire."
    
    elif 'acheter' in message or 'buy' in message:
        signal = context.get('overall_signal', 'HOLD')
        confidence = context.get('confidence', 50)
        
        if signal == 'BUY':
            return f"✅ Oui, les indicateurs suggèrent un **signal d'achat** sur {ticker} avec {confidence:.0f}% de confiance. Utilisez toujours un stop-loss !"
        else:
            return f"⚠️ Les indicateurs ne montrent pas un signal d'achat clair pour {ticker}. Signal actuel: {signal}."
    
    elif 'vendre' in message or 'sell' in message:
        signal = context.get('overall_signal', 'HOLD')
        
        if signal == 'SELL':
            return f"🚨 Les indicateurs suggèrent de **vendre** ou de réduire l'exposition sur {ticker}."
        else:
            return f"🛡️ Pas de signal de vente clair pour {ticker}. Signal actuel: {signal}."
    
    elif 'volatilité' in message or 'volatility' in message:
        volatility = context.get('volatility', 'MEDIUM')
        bb_width = context.get('bb_width', 0)
        
        return f"🌊 La volatilité de {ticker} est **{volatility}** (BB width: {bb_width:.2f}%)."
    
    else:
        return f"🤖 Je peux vous aider à analyser {ticker} ! Posez-moi des questions sur:\n" + \
               "- Le **RSI** (sur-achat/sur-vente)\n" + \
               "- Le **MACD** (momentum)\n" + \
               "- La **tendance** (direction du marché)\n" + \
               "- La **volatilité**\n" + \
               "- Conseils d'**achat/vente**"


# ========== API V8 ORACLE ==========

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
                'model': 'V8 Oracle Ensemble',
                'predictions': clean_for_json(result['predictions']),
                'ensemble': clean_for_json(result.get('ensemble', {}))
            })
        else:
            return jsonify({'error': result['error']}), 400
            
    except Exception as e:
        logger.error(f"Erreur V8: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v8/recommend/<ticker>')
def api_v8_recommend(ticker):
    if not v8_oracle:
        return jsonify({'error': 'V8 Oracle non disponible'}), 503
    
    risk = request.args.get('risk', 'medium')
    
    try:
        rec = v8_oracle.get_recommendation(ticker.upper(), risk_tolerance=risk)
        
        if 'error' not in rec:
            return jsonify(clean_for_json(rec))
        else:
            return jsonify({'error': rec['error']}), 400
            
    except Exception as e:
        logger.error(f"Erreur V8: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v8/batch')
def api_v8_batch():
    if not v8_oracle:
        return jsonify({'error': 'V8 Oracle non disponible'}), 503
    
    if cache['v8_predictions_cache'] and cache['v8_last_update']:
        if (datetime.now() - cache['v8_last_update']).seconds < 300:
            return jsonify(cache['v8_predictions_cache'])
    
    tickers = request.args.get('tickers', 'NVDA,MSFT,AAPL,GOOGL,AMZN,SPY,QQQ').split(',')
    tickers = [t.strip().upper() for t in tickers]
    
    try:
        result = v8_oracle.batch_predict(tickers)
        
        response = {
            'timestamp': result['timestamp'],
            'model': 'V8 Oracle Ensemble',
            'summary': clean_for_json(result['summary']),
            'tickers': clean_for_json(result['tickers'])
        }
        
        cache['v8_predictions_cache'] = response
        cache['v8_last_update'] = datetime.now()
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Erreur V8 batch: {e}")
        return jsonify({'error': str(e)}), 500


# ========== STANDARD ENDPOINTS ==========

@app.route('/api/account')
def api_account():
    if not alpaca_client:
        return jsonify({'error': 'Alpaca non disponible'}), 503
    
    if cache['account'] and cache['last_update']:
        if (datetime.now() - cache['last_update']).seconds < 30:
            return jsonify(cache['account'])
    
    account = alpaca_client.get_account()
    if account:
        cache['account'] = account
        cache['last_update'] = datetime.now()
        return jsonify(account)
    
    return jsonify({'error': 'Impossible de récupérer le compte'}), 500


@app.route('/api/positions')
def api_positions():
    if not alpaca_client:
        return jsonify({'error': 'Alpaca non disponible'}), 503
    
    positions = alpaca_client.get_positions()
    return jsonify(positions)


@app.route('/api/trades')
def api_trades():
    days = request.args.get('days', 7, type=int)
    
    trades_dir = Path('logs/trades')
    all_trades = []
    
    for i in range(days):
        date = datetime.now() - timedelta(days=i)
        filename = trades_dir / f"trades_{date.strftime('%Y-%m-%d')}.json"
        
        if filename.exists():
            try:
                with open(filename, 'r') as f:
                    daily_trades = json.load(f)
                    all_trades.extend(daily_trades)
            except:
                pass
    
    all_trades.sort(key=lambda t: t.get('timestamp', ''), reverse=True)
    
    return jsonify(all_trades)


@app.route('/api/health')
def api_health():
    return jsonify({'status': 'healthy'}), 200


@app.route('/api/status')
def api_status():
    return jsonify({
        'status': 'online',
        'timestamp': datetime.now().isoformat(),
        'v8_oracle_available': v8_oracle is not None,
        'alpaca_connected': alpaca_client is not None
    })


if __name__ == '__main__':
    import os
    
    host = os.getenv('DASHBOARD_HOST', '0.0.0.0')
    port = int(os.getenv('DASHBOARD_PORT', 5000))
    debug = os.getenv('DASHBOARD_DEBUG', 'false').lower() == 'true'
    
    print("\n" + "="*70)
    print("🌐 PLOUTOS WEB DASHBOARD - V8 ORACLE EDITION")
    print("="*70)
    print(f"\n🚀 Démarrage sur http://{host}:{port}")
    
    if v8_oracle:
        print(f"⭐ V8 Oracle: Actif ({len(v8_oracle.models)} modèles)")
    
    print("\n✅ Pages:")
    print("   - /       (Dashboard)")
    print("   - /chart  (Graphiques + IA)")
    print("\n" + "="*70 + "\n")
    
    app.run(host=host, port=port, debug=debug)
