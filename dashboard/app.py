"""Dashboard Flask pour le bot de trading - VERSION PRODUCTION"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, jsonify
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
        
        result = alpaca_client.close_position(symbol)
        
        if result:
            logger.info(f"✅ Position {symbol} fermée manuellement")
            return jsonify({'success': True, 'message': f'Position {symbol} fermée'})
        else:
            return jsonify({'success': False, 'error': 'Échec de fermeture'}), 400
    except Exception as e:
        logger.error(f"❌ Erreur close_position: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

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
        logger.info("="*70)
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    else:
        logger.error("❌ Échec démarrage dashboard")