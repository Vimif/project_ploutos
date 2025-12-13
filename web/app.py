#!/usr/bin/env python3
"""
🌐 PLOUTOS WEB DASHBOARD + V7 ENHANCED (AVEC COMPATIBILITÉ)

Dashboard Web moderne pour monitorer le bot de trading

Features:
- Vue temps réel du portfolio
- Graphiques de performances
- ★ Prédictions V7 Enhanced Momentum (68.35% accuracy)
- Health Score et auto-amélioration
- Historique des trades
- Alertes et suggestions
- Compatibilité avec anciens endpoints V7 (redirection automatique)

NOTE: Les anciens endpoints /api/v7/analysis et /api/v7/batch redirigent
      automatiquement vers le nouveau système V7 Enhanced pour éviter
      de casser le front-end existant.

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

# ★ IMPORT V7 ENHANCED PREDICTOR
try:
    from src.models.v7_predictor import V7Predictor
    V7_ENHANCED_AVAILABLE = True
except ImportError:
    V7_ENHANCED_AVAILABLE = False


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

# ★ INITIALISER V7 ENHANCED PREDICTOR UNIQUEMENT
v7_enhanced = None
if V7_ENHANCED_AVAILABLE:
    try:
        v7_enhanced = V7Predictor()
        if v7_enhanced.load("momentum"):
            logger.info("✅ V7 Enhanced Predictor chargé (68.35% accuracy)")
        else:
            logger.warning("⚠️  V7 Enhanced non chargé")
            v7_enhanced = None
    except Exception as e:
        logger.warning(f"⚠️  Erreur chargement V7 Enhanced: {e}")
        v7_enhanced = None

# Cache simple
cache = {
    'account': None,
    'positions': None,
    'trades': None,
    'improvement_report': None,
    'v7_enhanced_cache': None,
    'last_update': None,
    'v7_enhanced_last_update': None
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
    return jsonify({
        'status': 'online',
        'timestamp': datetime.now().isoformat(),
        'alpaca_connected': alpaca_client is not None,
        'self_improvement_available': SELF_IMPROVEMENT_AVAILABLE,
        'v7_enhanced_available': v7_enhanced is not None,
    })


# ★ V7 ENHANCED ENDPOINTS (NOUVEAUX)

@app.route('/api/v7/enhanced/predict/<ticker>')
def api_v7_enhanced_single(ticker):
    """Prédiction V7 Enhanced pour un ticker"""
    if not v7_enhanced:
        return jsonify({'error': 'V7 Enhanced non disponible'}), 503
    
    try:
        result = v7_enhanced.predict(ticker.upper(), period="3mo")
        
        if "error" not in result:
            return jsonify({
                'ticker': ticker.upper(),
                'timestamp': datetime.now().isoformat(),
                'model': 'V7 Enhanced Momentum',
                'accuracy': '68.35%',
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'signal_strength': result['signal_strength'],
                'probabilities': result['probabilities']
            })
        else:
            return jsonify({'error': result['error']}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v7/enhanced/batch')
def api_v7_enhanced_batch():
    """
    Prédictions V7 Enhanced pour plusieurs tickers
    Cache de 5 minutes
    """
    if not v7_enhanced:
        return jsonify({'error': 'V7 Enhanced non disponible'}), 503
    
    if cache['v7_enhanced_cache'] and cache['v7_enhanced_last_update']:
        if (datetime.now() - cache['v7_enhanced_last_update']).seconds < 300:
            return jsonify(cache['v7_enhanced_cache'])
    
    tickers = request.args.get('tickers', 'NVDA,MSFT,AAPL,GOOGL,AMZN,SPY,QQQ').split(',')
    
    try:
        logger.info(f"🔮 Génération prédictions V7 Enhanced pour {len(tickers)} tickers...")
        
        predictions = {}
        for ticker in tickers:
            ticker = ticker.strip().upper()
            try:
                result = v7_enhanced.predict(ticker, period="3mo")
                
                if "error" not in result:
                    predictions[ticker] = {
                        'prediction': result['prediction'],
                        'confidence': round(result['confidence'], 4),
                        'signal_strength': round(result['signal_strength'], 4),
                        'probabilities': {
                            'up': round(result['probabilities']['up'], 4),
                            'down': round(result['probabilities']['down'], 4)
                        }
                    }
                else:
                    predictions[ticker] = {'error': result['error']}
                    
            except Exception as e:
                predictions[ticker] = {'error': str(e)}
        
        response = {
            'timestamp': datetime.now().isoformat(),
            'model': 'V7 Enhanced Momentum',
            'accuracy': '68.35%',
            'predictions': predictions
        }
        
        cache['v7_enhanced_cache'] = response
        cache['v7_enhanced_last_update'] = datetime.now()
        
        logger.info(f"✅ Prédictions V7 Enhanced: {len(predictions)} tickers")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Erreur prédictions V7 Enhanced: {e}")
        return jsonify({'error': str(e)}), 500


# ★ V7 COMPATIBILITY ENDPOINTS (ANCIENS - REDIRECTION)

@app.route('/api/v7/analysis')
def api_v7_analysis():
    """
    COMPATIBILITÉ: Ancien endpoint /api/v7/analysis
    Redirige vers V7 Enhanced avec format compatible
    """
    ticker = request.args.get('ticker', '').upper()
    
    if not ticker:
        return jsonify({'error': 'Ticker requis'}), 400
    
    if not v7_enhanced:
        return jsonify({'error': 'V7 Enhanced non disponible'}), 503
    
    try:
        result = v7_enhanced.predict(ticker, period="3mo")
        
        if "error" not in result:
            # Format compatible avec l'ancien système V7 Ensemble
            signal = 'BUY' if result['prediction'] == 'UP' else 'SELL'
            strength = 'STRONG' if result['confidence'] > 0.65 else 'WEAK'
            
            return jsonify({
                'ticker': ticker,
                'signal': signal,
                'strength': strength,
                'experts': {
                    'momentum': {
                        'prediction': result['prediction'],
                        'confidence': result['confidence'] * 100
                    }
                },
                'timestamp': datetime.now().isoformat(),
                'note': 'Using V7 Enhanced (68.35% accuracy)'
            })
        else:
            return jsonify({'error': result['error']}), 400
            
    except Exception as e:
        logger.error(f"Erreur V7 analysis: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v7/batch')
def api_v7_batch():
    """
    COMPATIBILITÉ: Ancien endpoint /api/v7/batch
    Redirige vers V7 Enhanced avec format compatible
    """
    tickers = request.args.get('tickers', 'NVDA,AAPL,MSFT').split(',')
    
    if not v7_enhanced:
        return jsonify({'error': 'V7 Enhanced non disponible'}), 503
    
    results = []
    
    for ticker in tickers:
        ticker = ticker.strip().upper()
        try:
            result = v7_enhanced.predict(ticker, period="3mo")
            
            if "error" not in result:
                signal = 'BUY' if result['prediction'] == 'UP' else 'SELL'
                strength = 'STRONG' if result['confidence'] > 0.65 else 'WEAK'
                
                results.append({
                    'ticker': ticker,
                    'signal': signal,
                    'strength': strength,
                    'momentum_conf': float(result['confidence'] * 100)
                })
        except Exception as e:
            logger.error(f"Erreur V7 batch {ticker}: {e}")
            pass
    
    return jsonify({
        'results': convert_to_native_python(results),
        'timestamp': datetime.now().isoformat(),
        'note': 'Using V7 Enhanced (68.35% accuracy)'
    })


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
    """Historique des trades (depuis JSON)"""
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


@app.route('/api/improvement')
def api_improvement():
    """Rapport d'auto-amélioration"""
    if not SELF_IMPROVEMENT_AVAILABLE:
        return jsonify({'error': 'Self-Improvement non disponible'}), 503
    
    if cache['improvement_report']:
        report_time = datetime.fromisoformat(cache['improvement_report']['timestamp'])
        if (datetime.now() - report_time).seconds < 300:
            return jsonify(cache['improvement_report'])
    
    report_file = Path('logs/self_improvement_report.json')
    
    if report_file.exists():
        try:
            with open(report_file, 'r') as f:
                report = json.load(f)
                cache['improvement_report'] = report
                return jsonify(report)
        except:
            pass
    
    try:
        engine = SelfImprovementEngine()
        result = engine.analyze_recent_performance(days=7)
        
        if result['status'] == 'analyzed':
            report = engine.export_report()
            cache['improvement_report'] = report
            return jsonify(report)
    except Exception as e:
        logger.error(f"Erreur analyse: {e}")
    
    return jsonify({'error': 'Impossible de générer le rapport'}), 500


@app.route('/api/chart/portfolio')
def api_chart_portfolio():
    """Données pour graphique portfolio"""
    days = request.args.get('days', 30, type=int)
    
    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') 
             for i in range(days-1, -1, -1)]
    
    return jsonify({
        'dates': dates,
        'values': [100000] * days
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
    
    print("\n" + "="*60)
    print("🌐 PLOUTOS WEB DASHBOARD + V7 ENHANCED (AVEC COMPATIBILITÉ)")
    print("="*60)
    print(f"\n🚀 Démarrage sur http://{host}:{port}")
    print(f"🔧 Mode debug: {debug}")
    print(f"📊 Alpaca: {'Actif' if alpaca_client else 'Inactif'}")
    print(f"🧠 Self-Improvement: {'Actif' if SELF_IMPROVEMENT_AVAILABLE else 'Inactif'}")
    print(f"⭐ V7 Enhanced: {'Actif (68.35% accuracy)' if v7_enhanced else 'Inactif'}")
    print("\n✅ Endpoints V7:")
    print("   - /api/v7/enhanced/predict/<ticker> (nouveau)")
    print("   - /api/v7/enhanced/batch (nouveau)")
    print("   - /api/v7/analysis (ancien, redirigé vers V7 Enhanced)")
    print("   - /api/v7/batch (ancien, redirigé vers V7 Enhanced)")
    print("\n" + "="*60 + "\n")
    
    app.run(host=host, port=port, debug=debug)
