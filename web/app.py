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
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

# Import bibliothèque complète d'indicateurs
try:
    from web.utils.all_indicators import calculate_complete_indicators, get_indicator_signals
    from web.utils.advanced_ai import AdvancedAIAnalyzer
    COMPLETE_INDICATORS = True
except:
    COMPLETE_INDICATORS = False
    import ta

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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Initialiser l'IA avancée
ai_analyzer = AdvancedAIAnalyzer() if COMPLETE_INDICATORS else None

cache = {
    'account': None,
    'positions': None,
    'last_update': None,
    'chart_cache': {}
}


# ========== ROUTES HTML ==========

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chart')
def chart_page():
    return render_template('advanced_chart.html')


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
        
        # 🚀 NOUVEAU : Générer analyse IA complète
        ai_analysis = None
        if ai_analyzer:
            try:
                # Récupérer prédictions V8 si disponible
                v8_predictions = None
                if v8_oracle:
                    try:
                        v8_result = v8_oracle.predict_multi_horizon(ticker)
                        if 'error' not in v8_result:
                            v8_predictions = v8_result
                    except:
                        pass
                
                # Générer l'analyse complète
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
                ai_analysis = "Analyse IA temporairement indisponible"
        
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
            'ai_analysis': ai_analysis,  # ✨ NOUVEAU
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


def calculate_basic_indicators(df: pd.DataFrame) -> dict:
    """Fallback si bibliothèque complète non dispo"""
    import ta
    indicators = {}
    
    # Basiques
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
    """Génère les stats rapides pour le panneau"""
    
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
    
    # Volume
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
    """Chat IA avec réponses contextuelles"""
    data = request.json
    message = data.get('message', '').lower()
    ticker = data.get('ticker', '')
    context = data.get('context', {})
    
    # Réponses intelligentes
    response = generate_smart_ai_response(message, ticker, context)
    
    return jsonify({
        'response': response,
        'timestamp': datetime.now().isoformat()
    })


def generate_smart_ai_response(message: str, ticker: str, context: dict) -> str:
    """Génère réponse IA intelligente"""
    
    # Analyse complète
    if 'analyse' in message or 'complet' in message or 'détail' in message:
        if ai_analyzer and 'signals' in context:
            try:
                return ai_analyzer.generate_complete_analysis(ticker, context, None)
            except:
                pass
    
    # RSI
    if 'rsi' in message:
        rsi = context.get('rsi', 50)
        if rsi > 70:
            return f"📈 Le RSI de {ticker} est à {rsi:.1f}, en **sur-achat**. \n\n" \
                   f"Cela signifie que le titre a beaucoup monté récemment et pourrait connaître une correction à court terme. \n\n" \
                   f"⚠️ **Conseil**: Évitez d'acheter au plus haut. Attendez une baisse pour entrer en position."
        elif rsi < 30:
            return f"📉 Le RSI de {ticker} est à {rsi:.1f}, en **sur-vente**. \n\n" \
                   f"Le titre a beaucoup baissé et pourrait rebondir bientôt. \n\n" \
                   f"💡 **Opportunité**: Si la tendance globale est positive, c'est un bon point d'entrée."
        else:
            return f"🟡 Le RSI de {ticker} est à {rsi:.1f}, en zone **neutre**. \n\n" \
                   f"Le momentum n'est ni extrêmement haussier ni baissier. Attendez un signal plus clair."
    
    # MACD
    elif 'macd' in message:
        return f"📊 Le **MACD** (Moving Average Convergence Divergence) est un indicateur de momentum. \n\n" \
               f"🔍 **Comment l'utiliser**: \n" \
               f"- Croisement MACD > Signal = **Signal d'achat**\n" \
               f"- Croisement MACD < Signal = **Signal de vente**\n" \
               f"- Histogram positif = Momentum haussier\n" \
               f"- Histogram négatif = Momentum baissier"
    
    # Tendance
    elif 'tendance' in message or 'trend' in message:
        return f"📈 Pour évaluer la **tendance** de {ticker}: \n\n" \
               f"1. **SMA 20/50/200**: Si le prix est au-dessus → Tendance haussière\n" \
               f"2. **ADX > 25**: Tendance forte (peu importe la direction)\n" \
               f"3. **MACD**: Confirme la direction du momentum\n\n" \
               f"💡 Combinez plusieurs indicateurs pour plus de fiabilité !"
    
    # Acheter
    elif 'acheter' in message or 'buy' in message or 'achat' in message:
        rec = context.get('recommendation', 'HOLD')
        conf = context.get('confidence', 50)
        
        if 'BUY' in rec:
            return f"✅ **Signal d'achat détecté** sur {ticker} avec {conf:.0f}% de confiance !\n\n" \
                   f"📋 **Plan d'action**: \n" \
                   f"1. Entrez en position progressive (25-50% d'abord)\n" \
                   f"2. Placez un stop-loss à -4% du prix d'entrée\n" \
                   f"3. Objectif +8 à +12%\n" \
                   f"4. Surveillez le volume pour confirmation\n\n" \
                   f"⚠️ Toujours utiliser un stop-loss !"
        else:
            return f"⚠️ Les indicateurs ne montrent **pas de signal d'achat clair** pour {ticker}.\n\n" \
                   f"Signal actuel: **{rec}** ({conf:.0f}% confiance)\n\n" \
                   f"💡 **Conseil**: Attendez une meilleure opportunité. La patience paie en bourse !"
    
    # Vendre
    elif 'vendre' in message or 'sell' in message or 'vente' in message:
        rec = context.get('recommendation', 'HOLD')
        
        if 'SELL' in rec:
            return f"🚨 **Signal de vente détecté** sur {ticker} !\n\n" \
                   f"📋 **Actions recommandées**: \n" \
                   f"1. Sortez de vos positions longues\n" \
                   f"2. Prenez vos profits si vous êtes en gain\n" \
                   f"3. Coupez vos pertes si vous êtes en perte (stop-loss)\n\n" \
                   f"💡 Mieux vaut sortir trop tôt que trop tard !"
        else:
            return f"🛡️ Pas de signal de vente clair pour {ticker}.\n\n" \
                   f"Signal actuel: **{rec}**\n\n" \
                   f"Gardez vos positions si vous êtes satisfait de votre point d'entrée."
    
    # Volatilité
    elif 'volatilité' in message or 'volatility' in message or 'risque' in message:
        return f"🌊 La **volatilité** mesure l'amplitude des mouvements de prix.\n\n" \
               f"📊 **Indicateurs de volatilité**:\n" \
               f"- **ATR** (Average True Range): Amplitude moyenne\n" \
               f"- **Bollinger Bands**: Bandes de volatilité\n" \
               f"- **Bollinger Width**: Largeur des bandes\n\n" \
               f"💡 Forte volatilité = Plus de risque ET plus d'opportunités"
    
    # Niveau / Support / Résistance
    elif 'niveau' in message or 'support' in message or 'résistance' in message:
        price = context.get('price', 0)
        high_52w = context.get('high_52w', price)
        low_52w = context.get('low_52w', price)
        
        return f"🎯 **Niveaux clés pour {ticker}**:\n\n" \
               f"📈 **Résistances**:\n" \
               f"- Plus haut 52s: **{high_52w:.2f}$**\n" \
               f"- Prix actuel + 5%: **{price * 1.05:.2f}$**\n\n" \
               f"📉 **Supports**:\n" \
               f"- Prix actuel - 5%: **{price * 0.95:.2f}$**\n" \
               f"- Plus bas 52s: **{low_52w:.2f}$**\n\n" \
               f"💡 Surveillez les cassures de ces niveaux avec volume !"
    
    # Stratégie
    elif 'stratégie' in message or 'comment' in message or 'conseil' in message:
        return f"📚 **Guide de trading pour débutants**:\n\n" \
               f"1️⃣ **Toujours** utiliser un stop-loss (-3 à -5%)\n" \
               f"2️⃣ Ne risquez jamais plus de 2% de votre capital par trade\n" \
               f"3️⃣ Attendez la confluence de plusieurs signaux\n" \
               f"4️⃣ Suivez la tendance ("The trend is your friend")\n" \
               f"5️⃣ Prenez vos profits progressivement\n\n" \
               f"⚠️ **Ne tradez JAMAIS sous le coup de l'émotion !**"
    
    # Default
    else:
        return f"🤖 Je suis l'**assistant IA Ploutos V8** ! Je peux vous aider avec:\n\n" \
               f"📊 **Indicateurs**: RSI, MACD, tendance, volatilité\n" \
               f"💡 **Conseils**: acheter, vendre, stratégie\n" \
               f"🎯 **Niveaux**: support, résistance\n" \
               f"📈 **Analyse complète**: tapez "analyse complète"\n\n" \
               f"Posez-moi une question sur {ticker} !"


# ========== V8 ORACLE ==========

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


@app.route('/api/v8/recommend/<ticker>')
def api_v8_recommend(ticker):
    if not v8_oracle:
        return jsonify({'error': 'V8 Oracle non disponible'}), 503
    
    risk = request.args.get('risk', 'medium')
    
    try:
        rec = v8_oracle.get_recommendation(ticker.upper(), risk_tolerance=risk)
        return jsonify(clean_for_json(rec)) if 'error' not in rec else (jsonify({'error': rec['error']}), 400)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ========== STANDARD ==========

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
    
    return jsonify({'error': 'Erreur compte'}), 500


@app.route('/api/positions')
def api_positions():
    if not alpaca_client:
        return jsonify({'error': 'Alpaca non disponible'}), 503
    return jsonify(alpaca_client.get_positions())


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
                with open(filename) as f:
                    all_trades.extend(json.load(f))
            except:
                pass
    
    all_trades.sort(key=lambda t: t.get('timestamp', ''), reverse=True)
    return jsonify(all_trades)


@app.route('/api/health')
def api_health():
    return jsonify({'status': 'healthy'}), 200


if __name__ == '__main__':
    import os
    
    host = os.getenv('DASHBOARD_HOST', '0.0.0.0')
    port = int(os.getenv('DASHBOARD_PORT', 5000))
    
    print("\n" + "="*70)
    print("🌐 PLOUTOS WEB DASHBOARD - V8 ORACLE")
    print("="*70)
    print(f"\n🚀 http://{host}:{port}")
    
    if COMPLETE_INDICATORS:
        print("✅ 50+ indicateurs professionnels")
    
    if ai_analyzer:
        print("🤖 IA avancée activée (analyse multi-facteurs)")
    
    if v8_oracle:
        print(f"⭐ V8 Oracle: {len(v8_oracle.models)} modèles")
    
    print("\n✅ Pages: / et /chart")
    print("\n" + "="*70 + "\n")
    
    app.run(host=host, port=port, debug=False)
