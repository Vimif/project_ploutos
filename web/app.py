#!/usr/bin/env python3
"""
🌐 PLOUTOS WEB DASHBOARD

Dashboard Web moderne pour monitorer le bot de trading

Features:
- Vue temps réel du portfolio
- Graphiques de performances
- Health Score et auto-amélioration
- Historique des trades
- Alertes et suggestions

Auteur: Ploutos AI Team
Date: Dec 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
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

# Cache simple
cache = {
    'account': None,
    'positions': None,
    'trades': None,
    'improvement_report': None,
    'last_update': None
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
        'self_improvement_available': SELF_IMPROVEMENT_AVAILABLE
    })


@app.route('/api/account')
def api_account():
    """Informations du compte"""
    if not alpaca_client:
        return jsonify({'error': 'Alpaca non disponible'}), 503
    
    # Cache 30 secondes
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
    
    # Trier par date décroissante
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
    
    # Calculer stats basiques
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
    
    # Cache 5 minutes
    if cache['improvement_report']:
        report_time = datetime.fromisoformat(cache['improvement_report']['timestamp'])
        if (datetime.now() - report_time).seconds < 300:
            return jsonify(cache['improvement_report'])
    
    # Charger dernier rapport
    report_file = Path('logs/self_improvement_report.json')
    
    if report_file.exists():
        try:
            with open(report_file, 'r') as f:
                report = json.load(f)
                cache['improvement_report'] = report
                return jsonify(report)
        except:
            pass
    
    # Sinon, générer nouveau rapport
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
    
    # TODO: Implémenter historique portfolio depuis logs
    # Pour l'instant, retourner données factices
    
    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') 
             for i in range(days-1, -1, -1)]
    
    return jsonify({
        'dates': dates,
        'values': [100000] * days  # Placeholder
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
    print("🌐 PLOUTOS WEB DASHBOARD")
    print("="*60)
    print(f"\n🚀 Démarrage sur http://{host}:{port}")
    print(f"🔧 Mode debug: {debug}")
    print(f"📊 Alpaca: {'Actif' if alpaca_client else 'Inactif'}")
    print(f"🧠 Self-Improvement: {'Actif' if SELF_IMPROVEMENT_AVAILABLE else 'Inactif'}")
    print("\n" + "="*60 + "\n")
    
    app.run(host=host, port=port, debug=debug)
