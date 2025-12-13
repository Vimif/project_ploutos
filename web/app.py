#!/usr/bin/env python3
"""
🌐 PLOUTOS WEB DASHBOARD - V8 ORACLE EDITION

Dashboard Web moderne avec prédictions multi-horizon V8 Oracle

Features:
- Vue temps réel du portfolio
- Graphiques de performances
- ★ Prédictions V8 Oracle multi-horizon (65-75% accuracy)
- Recommandations BUY/SELL/HOLD intelligentes
- Analyse de confiance multi-facteurs
- Historique des trades
- Compatibilité rétroactive V7

Auteur: Ploutos AI Team
Date: Dec 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
import pandas as pd
import numpy as np
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

# ★ IMPORT V8 ORACLE ENSEMBLE
try:
    from src.models.v8_oracle_ensemble import V8OracleEnsemble
    V8_ORACLE_AVAILABLE = True
except ImportError:
    V8_ORACLE_AVAILABLE = False

# Fallback V7 pour compatibilité
try:
    from src.models.v7_predictor import V7Predictor
    V7_AVAILABLE = True
except ImportError:
    V7_AVAILABLE = False


def convert_to_native_python(obj):
    """Convertit les types numpy en types Python natifs pour JSON"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_native_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_python(item) for item in obj]
    return obj

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

# ★ INITIALISER V8 ORACLE
v8_oracle = None
if V8_ORACLE_AVAILABLE:
    try:
        v8_oracle = V8OracleEnsemble()
        if v8_oracle.load_models():
            logger.info("✅ V8 Oracle Ensemble chargé (multi-horizon 65-75% accuracy)")
        else:
            logger.warning("⚠️  V8 Oracle non chargé")
            v8_oracle = None
    except Exception as e:
        logger.warning(f"⚠️  Erreur chargement V8 Oracle: {e}")
        v8_oracle = None

# Fallback V7
v7_fallback = None
if not v8_oracle and V7_AVAILABLE:
    try:
        v7_fallback = V7Predictor()
        if v7_fallback.load("momentum"):
            logger.info("✅ V7 Fallback chargé (68.35% accuracy)")
        else:
            v7_fallback = None
    except Exception as e:
        logger.warning(f"⚠️  Erreur chargement V7: {e}")
        v7_fallback = None

# Cache simple
cache = {
    'account': None,
    'positions': None,
    'trades': None,
    'improvement_report': None,
    'v8_predictions_cache': None,
    'last_update': None,
    'v8_last_update': None
}


# ========== ROUTES HTML ==========

@app.route('/')
def index():
    """Page principale"""
    return render_template('index.html')


# ========== API ENDPOINTS ==========

@app.route('/api/status')
def api_status():
    """Status général du système"""
    predictor_status = 'none'
    predictor_info = 'Aucun modèle'
    
    if v8_oracle:
        predictor_status = 'v8_oracle'
        predictor_info = f'V8 Oracle ({len(v8_oracle.models)} modèles)'
    elif v7_fallback:
        predictor_status = 'v7_fallback'
        predictor_info = 'V7 Momentum (fallback)'
    
    return jsonify({
        'status': 'online',
        'timestamp': datetime.now().isoformat(),
        'alpaca_connected': alpaca_client is not None,
        'self_improvement_available': SELF_IMPROVEMENT_AVAILABLE,
        'predictor_status': predictor_status,
        'predictor_info': predictor_info,
        'v8_oracle_available': v8_oracle is not None,
        'v7_available': v7_fallback is not None
    })


# ★ V8 ORACLE ENDPOINTS (NOUVEAUX)

@app.route('/api/v8/predict/<ticker>')
def api_v8_predict_single(ticker):
    """
    Prédiction V8 Oracle multi-horizon pour un ticker
    """
    if not v8_oracle:
        if v7_fallback:
            return api_v7_fallback(ticker)
        return jsonify({'error': 'V8 Oracle non disponible'}), 503
    
    try:
        result = v8_oracle.predict_multi_horizon(ticker.upper())
        
        if 'error' not in result:
            return jsonify({
                'ticker': ticker.upper(),
                'timestamp': result['timestamp'],
                'model': 'V8 Oracle Ensemble',
                'predictions': convert_to_native_python(result['predictions']),
                'ensemble': convert_to_native_python(result.get('ensemble', {}))
            })
        else:
            return jsonify({'error': result['error']}), 400
            
    except Exception as e:
        logger.error(f"Erreur V8 prediction {ticker}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v8/recommend/<ticker>')
def api_v8_recommend(ticker):
    """
    Recommandation de trading V8 Oracle
    """
    if not v8_oracle:
        return jsonify({'error': 'V8 Oracle non disponible'}), 503
    
    risk = request.args.get('risk', 'medium')
    
    try:
        rec = v8_oracle.get_recommendation(ticker.upper(), risk_tolerance=risk)
        
        if 'error' not in rec:
            return jsonify(convert_to_native_python(rec))
        else:
            return jsonify({'error': rec['error']}), 400
            
    except Exception as e:
        logger.error(f"Erreur V8 recommendation {ticker}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v8/batch')
def api_v8_batch():
    """
    Prédictions V8 Oracle pour plusieurs tickers
    Cache de 5 minutes
    """
    if not v8_oracle:
        return jsonify({'error': 'V8 Oracle non disponible'}), 503
    
    # Check cache
    if cache['v8_predictions_cache'] and cache['v8_last_update']:
        if (datetime.now() - cache['v8_last_update']).seconds < 300:
            return jsonify(cache['v8_predictions_cache'])
    
    tickers = request.args.get('tickers', 'NVDA,MSFT,AAPL,GOOGL,AMZN,SPY,QQQ').split(',')
    tickers = [t.strip().upper() for t in tickers]
    
    try:
        logger.info(f"🔮 Génération prédictions V8 pour {len(tickers)} tickers...")
        
        result = v8_oracle.batch_predict(tickers)
        
        response = {
            'timestamp': result['timestamp'],
            'model': 'V8 Oracle Ensemble',
            'summary': convert_to_native_python(result['summary']),
            'tickers': convert_to_native_python(result['tickers'])
        }
        
        cache['v8_predictions_cache'] = response
        cache['v8_last_update'] = datetime.now()
        
        logger.info(f"✅ Prédictions V8: {len(tickers)} tickers")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Erreur prédictions V8: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v8/heatmap')
def api_v8_heatmap():
    """
    Heatmap de confiance pour tous les tickers
    """
    if not v8_oracle:
        return jsonify({'error': 'V8 Oracle non disponible'}), 503
    
    tickers = request.args.get('tickers', 'NVDA,MSFT,AAPL,GOOGL,AMZN,META,TSLA').split(',')
    tickers = [t.strip().upper() for t in tickers]
    
    try:
        heatmap_data = []
        
        for ticker in tickers:
            result = v8_oracle.predict_multi_horizon(ticker)
            
            if 'error' not in result:
                row = {'ticker': ticker}
                
                for horizon, pred in result.get('predictions', {}).items():
                    if 'error' not in pred:
                        row[horizon] = pred.get('confidence', 0)
                
                heatmap_data.append(row)
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'data': convert_to_native_python(heatmap_data)
        })
        
    except Exception as e:
        logger.error(f"Erreur heatmap: {e}")
        return jsonify({'error': str(e)}), 500


# ★ ENDPOINTS COMPATIBILITÉ V7

@app.route('/api/v7/enhanced/predict/<ticker>')
@app.route('/api/v7/analysis')
def api_v7_compatibility(ticker=None):
    """
    Compatibilité V7 - redirige vers V8 ou fallback V7
    """
    if not ticker:
        ticker = request.args.get('ticker', '').upper()
    
    if not ticker:
        return jsonify({'error': 'Ticker requis'}), 400
    
    # Utiliser V8 si disponible
    if v8_oracle:
        try:
            result = v8_oracle.predict_multi_horizon(ticker)
            
            if 'error' not in result and 'ensemble' in result:
                ensemble = result['ensemble']
                
                signal = 'BUY' if ensemble['prediction'] == 'UP' else 'SELL'
                strength = ensemble.get('agreement', 'WEAK')
                
                return jsonify({
                    'ticker': ticker,
                    'signal': signal,
                    'strength': strength,
                    'confidence': ensemble['confidence'],
                    'model': 'V8 Oracle',
                    'timestamp': result['timestamp'],
                    'note': 'Using V8 Oracle multi-horizon (65-75% accuracy)'
                })
        except Exception as e:
            logger.error(f"Erreur V8: {e}")
    
    # Fallback V7
    if v7_fallback:
        return api_v7_fallback(ticker)
    
    return jsonify({'error': 'Aucun modèle disponible'}), 503


def api_v7_fallback(ticker):
    """Fallback V7 si V8 non disponible"""
    try:
        result = v7_fallback.predict(ticker, period="3mo")
        
        if "error" not in result:
            signal = 'BUY' if result['prediction'] == 'UP' else 'SELL'
            strength = 'STRONG' if result['confidence'] > 0.65 else 'WEAK'
            
            return jsonify({
                'ticker': ticker,
                'signal': signal,
                'strength': strength,
                'confidence': result['confidence'] * 100,
                'model': 'V7 Momentum (fallback)',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': result['error']}), 400
            
    except Exception as e:
        logger.error(f"Erreur V7 fallback: {e}")
        return jsonify({'error': str(e)}), 500


# ========== ENDPOINTS STANDARD ==========

@app.route('/api/account')
def api_account():
    """Informations du compte"""
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
    """Positions actuelles"""
    if not alpaca_client:
        return jsonify({'error': 'Alpaca non disponible'}), 503
    
    positions = alpaca_client.get_positions()
    cache['positions'] = positions
    
    return jsonify(positions)


@app.route('/api/trades')
def api_trades():
    """Historique des trades"""
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


@app.route('/api/performance')
def api_performance():
    """Statistiques de performance"""
    days = request.args.get('days', 7, type=int)
    
    trades_dir = Path('logs/trades')
    trades = []
    
    for i in range(days):
        date = datetime.now() - timedelta(days=i)
        filename = trades_dir / f"trades_{date.strftime('%Y-%m-%d')}.json"
        
        if filename.exists():
            try:
                with open(filename, 'r') as f:
                    daily_trades = json.load(f)
                    trades.extend(daily_trades)
            except:
                pass
    
    buys = [t for t in trades if t['action'] == 'BUY']
    sells = [t for t in trades if t['action'] == 'SELL']
    
    total_invested = sum(t['amount'] for t in buys)
    total_proceeds = sum(t['amount'] for t in sells)
    
    return jsonify({
        'total_trades': len(trades),
        'buy_count': len(buys),
        'sell_count': len(sells),
        'total_invested': total_invested,
        'total_proceeds': total_proceeds,
        'net_pnl': total_proceeds - total_invested,
        'days_analyzed': days
    })


@app.route('/api/health')
def api_health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200


# ========== GESTION ERREURS ==========

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ========== MAIN ==========

if __name__ == '__main__':
    import os
    
    host = os.getenv('DASHBOARD_HOST', '0.0.0.0')
    port = int(os.getenv('DASHBOARD_PORT', 5000))
    debug = os.getenv('DASHBOARD_DEBUG', 'false').lower() == 'true'
    
    print("\n" + "="*70)
    print("🌐 PLOUTOS WEB DASHBOARD - V8 ORACLE EDITION")
    print("="*70)
    print(f"\n🚀 Démarrage sur http://{host}:{port}")
    print(f"🔧 Mode debug: {debug}")
    print(f"📊 Alpaca: {'Actif' if alpaca_client else 'Inactif'}")
    print(f"🧠 Self-Improvement: {'Actif' if SELF_IMPROVEMENT_AVAILABLE else 'Inactif'}")
    
    if v8_oracle:
        print(f"⭐ V8 Oracle: Actif ({len(v8_oracle.models)} modèles, 65-75% accuracy)")
        print(f"   Modèles chargés: {', '.join(v8_oracle.models.keys())}")
    elif v7_fallback:
        print(f"⚠️  V7 Fallback: Actif (68.35% accuracy)")
    else:
        print("❌ Aucun modèle prédictif chargé")
    
    print("\n✅ Endpoints V8 Oracle:")
    print("   - /api/v8/predict/<ticker>     (prédiction multi-horizon)")
    print("   - /api/v8/recommend/<ticker>   (recommandation BUY/SELL/HOLD)")
    print("   - /api/v8/batch               (analyse batch)")
    print("   - /api/v8/heatmap             (heatmap de confiance)")
    print("\n🔄 Compatibilité V7:")
    print("   - /api/v7/analysis            (ancien format)")
    print("   - /api/v7/enhanced/predict/<ticker>")
    print("\n" + "="*70 + "\n")
    
    app.run(host=host, port=port, debug=debug)
