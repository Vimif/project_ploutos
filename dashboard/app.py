# dashboard/app.py
"""Dashboard Flask pour le bot de trading"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import json
from datetime import datetime, timedelta
import threading
import time

from trading.alpaca_client import AlpacaClient
from core.utils import setup_logging

logger = setup_logging(__name__, 'dashboard.log')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Client Alpaca global
alpaca_client = None

def init_alpaca():
    """Initialiser le client Alpaca"""
    global alpaca_client
    try:
        alpaca_client = AlpacaClient(paper_trading=True)
        logger.info("✅ Client Alpaca initialisé pour dashboard")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur init Alpaca: {e}")
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
            init_alpaca()
        
        account = alpaca_client.get_account()
        
        return jsonify({
            'success': True,
            'data': {
                'portfolio_value': float(account['portfolio_value']),
                'cash': float(account['cash']),
                'buying_power': float(account['buying_power']),
                'equity': float(account['equity']),
                'last_equity': float(account.get('last_equity', account['equity'])),
                'daytrade_count': account.get('daytrade_count', 0),
                'pattern_day_trader': account.get('pattern_day_trader', False)
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
            init_alpaca()
        
        positions = alpaca_client.get_positions()
        
        positions_data = []
        for pos in positions:
            positions_data.append({
                'symbol': pos['symbol'],
                'qty': float(pos['qty']),
                'avg_entry_price': float(pos['avg_entry_price']),
                'current_price': float(pos['current_price']),
                'market_value': float(pos['market_value']),
                'cost_basis': float(pos.get('cost_basis', pos['qty'] * pos['avg_entry_price'])),
                'unrealized_pl': float(pos['unrealized_pl']),
                'unrealized_plpc': float(pos['unrealized_plpc']) * 100,
                'side': pos.get('side', 'long')
            })
        
        return jsonify({
            'success': True,
            'data': positions_data
        })
    except Exception as e:
        logger.error(f"❌ Erreur /api/positions: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/orders')
def get_orders():
    """Obtenir les ordres récents"""
    try:
        if not alpaca_client:
            init_alpaca()
        
        # Récupérer ordres fermés des dernières 24h
        orders = alpaca_client.get_orders(status='closed', limit=50)
        
        orders_data = []
        for order in orders:
            orders_data.append({
                'id': order.get('id', ''),
                'symbol': order.get('symbol', ''),
                'qty': float(order.get('qty', 0)),
                'side': order.get('side', ''),
                'type': order.get('type', ''),
                'status': order.get('status', ''),
                'filled_avg_price': float(order.get('filled_avg_price', 0)) if order.get('filled_avg_price') else 0,
                'filled_at': order.get('filled_at', ''),
                'created_at': order.get('created_at', '')
            })
        
        return jsonify({
            'success': True,
            'data': orders_data
        })
    except Exception as e:
        logger.error(f"❌ Erreur /api/orders: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/performance')
def get_performance():
    """Calculer les performances"""
    try:
        if not alpaca_client:
            init_alpaca()
        
        account = alpaca_client.get_account()
        positions = alpaca_client.get_positions()
        
        # Calculer métriques
        portfolio_value = float(account['portfolio_value'])
        equity = float(account['equity'])
        last_equity = float(account.get('last_equity', equity))
        
        daily_pl = equity - last_equity
        daily_pl_pct = (daily_pl / last_equity * 100) if last_equity > 0 else 0
        
        # Total P&L des positions ouvertes
        total_unrealized_pl = sum(float(pos['unrealized_pl']) for pos in positions)
        total_unrealized_plpc = (total_unrealized_pl / (equity - total_unrealized_pl) * 100) if (equity - total_unrealized_pl) > 0 else 0
        
        # Positions gagnantes vs perdantes
        winning_positions = [p for p in positions if float(p['unrealized_pl']) > 0]
        losing_positions = [p for p in positions if float(p['unrealized_pl']) < 0]
        
        win_rate = (len(winning_positions) / len(positions) * 100) if positions else 0
        
        return jsonify({
            'success': True,
            'data': {
                'portfolio_value': portfolio_value,
                'daily_pl': daily_pl,
                'daily_pl_pct': daily_pl_pct,
                'total_unrealized_pl': total_unrealized_pl,
                'total_unrealized_plpc': total_unrealized_plpc,
                'total_positions': len(positions),
                'winning_positions': len(winning_positions),
                'losing_positions': len(losing_positions),
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
            init_alpaca()
        
        result = alpaca_client.close_position(symbol)
        
        if result:
            return jsonify({
                'success': True,
                'message': f'Position {symbol} fermée'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Échec de fermeture'
            }), 400
    except Exception as e:
        logger.error(f"❌ Erreur close_position: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# WebSocket pour mises à jour temps réel
def background_updates():
    """Envoyer des mises à jour toutes les 5 secondes"""
    while True:
        try:
            if alpaca_client:
                account = alpaca_client.get_account()
                positions = alpaca_client.get_positions()
                
                socketio.emit('account_update', {
                    'portfolio_value': float(account['portfolio_value']),
                    'cash': float(account['cash']),
                    'buying_power': float(account['buying_power'])
                })
                
                socketio.emit('positions_update', {
                    'count': len(positions),
                    'total_value': sum(float(p['market_value']) for p in positions)
                })
        except Exception as e:
            logger.error(f"❌ Erreur background_updates: {e}")
        
        time.sleep(5)

@socketio.on('connect')
def handle_connect():
    logger.info("🔌 Client connecté au WebSocket")
    emit('status', {'message': 'Connecté au serveur'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("🔌 Client déconnecté du WebSocket")

if __name__ == '__main__':
    logger.info("🚀 Démarrage du dashboard...")
    
    # Initialiser Alpaca
    if init_alpaca():
        logger.info("✅ Dashboard prêt")
        
        # Lancer thread de mise à jour en background
        update_thread = threading.Thread(target=background_updates, daemon=True)
        update_thread.start()
        
        # Démarrer le serveur
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    else:
        logger.error("❌ Impossible de démarrer le dashboard")