# dashboard/app.py
"""Dashboard Flask pour le bot de trading - VERSION JSON (Sans PostgreSQL)"""

import sys
import os
import secrets
from pathlib import Path
from functools import wraps
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, jsonify, request, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import traceback
import json
from datetime import datetime, timedelta
from collections import defaultdict

from trading.alpaca_client import AlpacaClient
from core.utils import setup_logging

logger = setup_logging(__name__, 'dashboard.log')

# Configuration Flask
app = Flask(__name__)
# SECURE: Use environment variable or generate random key
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(24))
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

# Dossier logs trades
TRADES_LOG_DIR = Path('logs/trades')

# ========== AUTHENTICATION ==========

def check_auth(username, password):
    """Check if a username/password combination is valid."""
    correct_username = os.environ.get('DASHBOARD_USERNAME', 'admin')
    correct_password = os.environ.get('DASHBOARD_PASSWORD', 'ploutos')
    return secrets.compare_digest(username, correct_username) and \
           secrets.compare_digest(password, correct_password)

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return Response(
                'Could not verify your access level for that URL.\n'
                'You have to login with proper credentials', 401,
                {'WWW-Authenticate': 'Basic realm="Login Required"'})
        return f(*args, **kwargs)
    return decorated

# ====================================

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

def load_trades_from_json(days=30):
    """
    Charger les trades depuis les fichiers JSON
    
    Args:
        days: Nombre de jours à charger
    
    Returns:
        list: Liste des trades
    """
    trades = []
    
    try:
        if not TRADES_LOG_DIR.exists():
            logger.warning(f"⚠️  Dossier {TRADES_LOG_DIR} n'existe pas")
            return []
        
        # Charger tous les fichiers trades_*.json
        for json_file in sorted(TRADES_LOG_DIR.glob('trades_*.json'), reverse=True):
            try:
                with open(json_file, 'r') as f:
                    file_trades = json.load(f)
                    trades.extend(file_trades)
            except Exception as e:
                logger.error(f"❌ Erreur lecture {json_file}: {e}")
        
        # Filtrer par date si nécessaire
        if days and trades:
            cutoff = datetime.now() - timedelta(days=days)
            trades = [
                t for t in trades 
                if datetime.fromisoformat(t['timestamp']) > cutoff
            ]
        
        # Trier par date décroissante
        trades.sort(key=lambda t: t['timestamp'], reverse=True)
        
        logger.debug(f"✅ {len(trades)} trades chargés depuis JSON")
        return trades
        
    except Exception as e:
        logger.error(f"❌ Erreur load_trades_from_json: {e}")
        return []

def calculate_statistics_from_trades(trades):
    """
    Calculer statistiques depuis les trades JSON
    
    Args:
        trades: Liste des trades
    
    Returns:
        dict: Statistiques
    """
    if not trades:
        return {
            'total_trades': 0,
            'buy_count': 0,
            'sell_count': 0,
            'total_volume': 0,
            'avg_trade_size': 0
        }
    
    buy_trades = [t for t in trades if t['action'] == 'BUY']
    sell_trades = [t for t in trades if t['action'] == 'SELL']
    
    total_volume = sum(t['amount'] for t in trades)
    
    return {
        'total_trades': len(trades),
        'buy_count': len(buy_trades),
        'sell_count': len(sell_trades),
        'total_volume': total_volume,
        'avg_trade_size': total_volume / len(trades) if trades else 0
    }

def get_top_symbols_from_trades(trades, limit=10):
    """
    Obtenir les symboles les plus tradés
    
    Args:
        trades: Liste des trades
        limit: Nombre de symboles à retourner
    
    Returns:
        list: Top symboles
    """
    symbol_stats = defaultdict(lambda: {'count': 0, 'volume': 0})
    
    for trade in trades:
        symbol = trade['symbol']
        symbol_stats[symbol]['count'] += 1
        symbol_stats[symbol]['volume'] += trade['amount']
    
    # Convertir en liste et trier
    top_symbols = [
        {
            'symbol': symbol,
            'trade_count': stats['count'],
            'total_volume': stats['volume']
        }
        for symbol, stats in symbol_stats.items()
    ]
    
    top_symbols.sort(key=lambda x: x['trade_count'], reverse=True)
    
    return top_symbols[:limit]

@app.route('/')
@requires_auth
def index():
    """Page principale du dashboard"""
    return render_template('index.html')

@app.route('/api/account')
@requires_auth
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
@requires_auth
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
@requires_auth
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
@requires_auth
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
@requires_auth
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

# ========== ROUTES LECTURE JSON (REMPLACEMENT BDD) ==========

@app.route('/api/db/trades')
@requires_auth
def api_db_trades():
    """Historique trades depuis JSON"""
    try:
        days = int(request.args.get('days', 30))
        symbol = request.args.get('symbol', None)
        
        trades = load_trades_from_json(days=days)
        
        # Filtrer par symbole si demandé
        if symbol:
            trades = [t for t in trades if t['symbol'] == symbol]
        
        return jsonify({
            'success': True,
            'data': trades,
            'count': len(trades),
            'source': 'json'
        })
    except Exception as e:
        logger.error(f"❌ Erreur /api/db/trades: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/db/statistics')
@requires_auth
def api_db_statistics():
    """Statistiques depuis JSON"""
    try:
        days = int(request.args.get('days', 30))
        
        trades = load_trades_from_json(days=days)
        stats = calculate_statistics_from_trades(trades)
        top_symbols = get_top_symbols_from_trades(trades, limit=10)
        
        # Win/loss basique
        buy_count = stats['buy_count']
        sell_count = stats['sell_count']
        
        return jsonify({
            'success': True,
            'data': {
                'statistics': stats,
                'top_symbols': top_symbols,
                'win_loss': {
                    'buy_count': buy_count,
                    'sell_count': sell_count
                }
            },
            'source': 'json'
        })
    except Exception as e:
        logger.error(f"❌ Erreur /api/db/statistics: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/db/evolution')
@requires_auth
def api_db_evolution():
    """Évolution portfolio depuis JSON"""
    try:
        days = int(request.args.get('days', 30))
        trades = load_trades_from_json(days=days)
        
        # Grouper par jour
        daily_data = defaultdict(lambda: {'trades': 0, 'volume': 0, 'portfolio_value': None})
        
        for trade in trades:
            date = trade['timestamp'][:10]  # YYYY-MM-DD
            daily_data[date]['trades'] += 1
            daily_data[date]['volume'] += trade['amount']
            if trade.get('portfolio_value'):
                daily_data[date]['portfolio_value'] = trade['portfolio_value']
        
        # Convertir en liste
        evolution = [
            {
                'date': date,
                'trades': data['trades'],
                'volume': data['volume'],
                'portfolio_value': data['portfolio_value']
            }
            for date, data in sorted(daily_data.items())
        ]
        
        return jsonify({
            'success': True,
            'data': evolution,
            'source': 'json'
        })
    except Exception as e:
        logger.error(f"❌ Erreur /api/db/evolution: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/db/summary')
@requires_auth
def api_db_summary():
    """Résumés quotidiens depuis JSON"""
    try:
        days = int(request.args.get('days', 30))
        trades = load_trades_from_json(days=days)
        
        # Grouper par jour
        daily_summary = defaultdict(lambda: {
            'date': None,
            'trade_count': 0,
            'buy_count': 0,
            'sell_count': 0,
            'total_volume': 0,
            'portfolio_value': None
        })
        
        for trade in trades:
            date = trade['timestamp'][:10]
            daily_summary[date]['date'] = date
            daily_summary[date]['trade_count'] += 1
            daily_summary[date]['total_volume'] += trade['amount']
            
            if trade['action'] == 'BUY':
                daily_summary[date]['buy_count'] += 1
            else:
                daily_summary[date]['sell_count'] += 1
            
            if trade.get('portfolio_value'):
                daily_summary[date]['portfolio_value'] = trade['portfolio_value']
        
        summaries = list(daily_summary.values())
        summaries.sort(key=lambda x: x['date'], reverse=True)
        
        return jsonify({
            'success': True,
            'data': summaries,
            'source': 'json'
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
    logger.info("🚀 DÉMARRAGE DU DASHBOARD PLOUTOS (JSON MODE)")
    logger.info("="*70)
    
    if init_alpaca():
        logger.info("✅ Dashboard prêt sur http://0.0.0.0:5000")
        logger.info("📝 Source données: Fichiers JSON (logs/trades/)")
        logger.info("="*70)
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    else:
        logger.error("❌ Échec démarrage dashboard")