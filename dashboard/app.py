# dashboard/app.py
"""Dashboard Flask pour le bot de trading - VERSION COMPLÈTE AVEC BDD"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import traceback

from trading.alpaca_client import AlpacaClient
from core.utils import setup_logging

logger = setup_logging(__name__, 'dashboard.log')

# Configuration Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'ploutos-trading-bot-secret-2025'
CORS(app)

# SocketIO avec gevent
socketio = SocketIO(
    app, 
    cors_allowed_origins="*",
    async_mode='gevent',
    logger=False,
    engineio_logger=False
)

# Client Alpaca global
alpaca_client = None

# ========== INTÉGRATION BASE DE DONNÉES ==========
try:
    from database.db import (
        get_trade_history, get_trade_statistics,
        get_portfolio_evolution, get_daily_summary,
        get_top_symbols, get_win_loss_ratio
    )
    DB_AVAILABLE = True
    logger.info("✅ Module database disponible")
except ImportError:
    DB_AVAILABLE = False
    logger.warning("⚠️  Module database non disponible")

def init_alpaca():
    """Initialiser le client Alpaca"""
    global alpaca_client
    try:
        logger.info("🔄 Initialisation du client Alpaca...")
        alpaca_client = AlpacaClient(paper_trading=True)
        logger.info("✅ Client Alpaca initialisé")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur init Alpaca: {e}")
        logger.error(traceback.format_exc())
        return False

@app.route('/')
def index():
    """Page principale du dashboard"""
    return render_template('index.html')

@app.route('/api/account')
def get_account():
    """Obtenir les infos du compte"""
    try:
        if not alpaca_client:
            if not init_alpaca():
                return jsonify({'success': False, 'error': 'Client non initialisé'}), 500
        
        account = alpaca_client.get_account()
        
        return jsonify({
            'success': True,
            'data': {
                'portfolio_value': float(account['portfolio_value']),
                'cash': float(account['cash']),
                'buying_power': float(account['buying_power']),
                'equity': float(account['equity']),
                'last_equity': float(account.get('last_equity', account['equity']))
            }
        })
    except Exception as e:
        logger.error(f"❌ Erreur /api/account: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/positions')
def get_positions():
    """Obtenir toutes les positions"""
    try:
        if not alpaca_client:
            if not init_alpaca():
                return jsonify({'success': False, 'error': 'Client non initialisé'}), 500
        
        positions = alpaca_client.get_positions()
        
        positions_data = [{
            'symbol': p['symbol'],
            'qty': float(p['qty']),
            'avg_entry_price': float(p['avg_entry_price']),
            'current_price': float(p['current_price']),
            'market_value': float(p['market_value']),
            'unrealized_pl': float(p['unrealized_pl']),
            'unrealized_plpc': float(p['unrealized_plpc']) * 100
        } for p in positions]
        
        return jsonify({'success': True, 'data': positions_data})
    except Exception as e:
        logger.error(f"❌ Erreur /api/positions: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/orders')
def get_orders():
    """Obtenir les ordres récents"""
    try:
        if not alpaca_client:
            if not init_alpaca():
                return jsonify({'success': False, 'error': 'Client non initialisé'}), 500
        
        orders = alpaca_client.get_orders(status='closed', limit=50)
        
        orders_data = [{
            'symbol': o.get('symbol', ''),
            'qty': float(o.get('qty', 0)),
            'side': o.get('side', ''),
            'status': o.get('status', ''),
            'filled_avg_price': float(o.get('filled_avg_price', 0)) if o.get('filled_avg_price') else 0,
            'filled_at': o.get('filled_at', '')
        } for o in orders]
        
        return jsonify({'success': True, 'data': orders_data})
    except Exception as e:
        logger.error(f"❌ Erreur /api/orders: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/performance')
def get_performance():
    """Calculer les performances"""
    try:
        if not alpaca_client:
            if not init_alpaca():
                return jsonify({'success': False, 'error': 'Client non initialisé'}), 500
        
        account = alpaca_client.get_account()
        positions = alpaca_client.get_positions()
        
        total_pl = sum(float(p['unrealized_pl']) for p in positions)
        winning = [p for p in positions if float(p['unrealized_pl']) > 0]
        losing = [p for p in positions if float(p['unrealized_pl']) < 0]
        win_rate = (len(winning) / len(positions) * 100) if positions else 0
        
        equity = float(account['equity'])
        
        return jsonify({
            'success': True,
            'data': {
                'total_unrealized_pl': total_pl,
                'total_unrealized_plpc': (total_pl / equity * 100) if equity > 0 else 0,
                'total_positions': len(positions),
                'winning_positions': len(winning),
                'losing_positions': len(losing),
                'win_rate': win_rate
            }
        })
    except Exception as e:
        logger.error(f"❌ Erreur /api/performance: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/close_position/<symbol>', methods=['POST'])
def close_position(symbol):
    """Fermer une position manuellement"""
    try:
        if not alpaca_client:
            if not init_alpaca():
                return jsonify({'success': False, 'error': 'Client non initialisé'}), 500
        
        result = alpaca_client.close_position(symbol, reason='Fermeture manuelle dashboard')
        
        if result:
            logger.info(f"✅ Position {symbol} fermée manuellement")
            return jsonify({'success': True, 'message': f'Position {symbol} fermée'})
        else:
            return jsonify({'success': False, 'error': 'Échec de fermeture'}), 400
    except Exception as e:
        logger.error(f"❌ Erreur close_position: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ========== ROUTES AVEC BASE DE DONNÉES ==========

@app.route('/api/db/trades')
def api_db_trades():
    """Historique trades depuis BDD"""
    if not DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'BDD non configurée'}), 503
    
    try:
        days = int(request.args.get('days', 30))
        symbol = request.args.get('symbol', None)
        
        trades = get_trade_history(days=days, symbol=symbol)
        
        # Convertir les dates en string pour JSON
        for trade in trades:
            if 'timestamp' in trade and trade['timestamp']:
                trade['timestamp'] = str(trade['timestamp'])
        
        return jsonify({
            'success': True,
            'data': trades,
            'count': len(trades)
        })
    except Exception as e:
        logger.error(f"❌ Erreur /api/db/trades: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/db/statistics')
def api_db_statistics():
    """Statistiques depuis BDD"""
    if not DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'BDD non configurée'}), 503
    
    try:
        days = int(request.args.get('days', 30))
        
        stats = get_trade_statistics(days=days)
        top_symbols = get_top_symbols(days=days, limit=10)
        win_loss = get_win_loss_ratio(days=days)
        
        # Convertir Decimal en float
        for key in stats:
            if stats[key] is not None:
                stats[key] = float(stats[key])
        
        for symbol_data in top_symbols:
            for key in symbol_data:
                if symbol_data[key] is not None and key != 'symbol':
                    symbol_data[key] = float(symbol_data[key])
        
        return jsonify({
            'success': True,
            'data': {
                'statistics': stats,
                'top_symbols': top_symbols,
                'win_loss': win_loss
            }
        })
    except Exception as e:
        logger.error(f"❌ Erreur /api/db/statistics: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/db/evolution')
def api_db_evolution():
    """Évolution portfolio depuis BDD"""
    if not DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'BDD non configurée'}), 503
    
    try:
        days = int(request.args.get('days', 30))
        evolution = get_portfolio_evolution(days=days)
        
        # Convertir dates et Decimal
        for point in evolution:
            if 'date' in point and point['date']:
                point['date'] = str(point['date'])
            for key in ['portfolio_value', 'total_pl', 'cash']:
                if key in point and point[key] is not None:
                    point[key] = float(point[key])
        
        return jsonify({
            'success': True,
            'data': evolution
        })
    except Exception as e:
        logger.error(f"❌ Erreur /api/db/evolution: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/db/summary')
def api_db_summary():
    """Résumés quotidiens depuis BDD"""
    if not DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'BDD non configurée'}), 503
    
    try:
        days = int(request.args.get('days', 30))
        summaries = get_daily_summary(days=days)
        
        # Convertir dates et Decimal
        for summary in summaries:
            if 'date' in summary and summary['date']:
                summary['date'] = str(summary['date'])
            for key in ['portfolio_value', 'cash', 'buying_power', 'total_pl']:
                if key in summary and summary[key] is not None:
                    summary[key] = float(summary[key])
        
        return jsonify({
            'success': True,
            'data': summaries
        })
    except Exception as e:
        logger.error(f"❌ Erreur /api/db/summary: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ========== WEBSOCKET ==========

@socketio.on('connect')
def handle_connect():
    logger.info("🔌 Client WebSocket connecté")
    emit('status', {'message': 'Connecté au serveur'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("🔌 Client WebSocket déconnecté")

if __name__ == '__main__':
    logger.info("="*70)
    logger.info("🚀 DÉMARRAGE DU DASHBOARD PLOUTOS")
    logger.info("="*70)
    
    if init_alpaca():
        logger.info("✅ Dashboard prêt sur http://0.0.0.0:5000")
        logger.info(f"📊 Base de données: {'✅ Disponible' if DB_AVAILABLE else '❌ Non configurée'}")
        logger.info("="*70)
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    else:
        logger.error("❌ Échec démarrage dashboard")