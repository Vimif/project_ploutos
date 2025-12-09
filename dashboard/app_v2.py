# dashboard/app_v2.py
"""Dashboard Flask AMÉLIORÉ pour Ploutos - Version 2.0

Nouveautés:
- Connexion PostgreSQL native avec fallback JSON
- Métriques financières avancées (Sharpe, Sortino, Calmar)
- Analyse drawdown et risque
- Comparaison benchmark
- Analytics par symbole
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import traceback
import json
from datetime import datetime, timedelta
from collections import defaultdict

from trading.alpaca_client import AlpacaClient
from database.db import (
    get_trade_history, get_daily_summary, get_trade_statistics,
    get_top_symbols, get_portfolio_evolution, get_win_loss_ratio,
    get_connection
)
from dashboard.analytics import PortfolioAnalytics
from core.utils import setup_logging

logger = setup_logging(__name__, 'dashboard_v2.log')

# Configuration Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'ploutos-trading-bot-v2-secret-2025'
CORS(app)

# SocketIO
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='gevent',
    logger=False,
    engineio_logger=False
)

# Client Alpaca global
alpaca_client = None

# Flag: PostgreSQL disponible ?
PG_AVAILABLE = False

# Dossier logs trades (fallback JSON)
TRADES_LOG_DIR = Path('logs/trades')


def check_postgres():
    """Vérifier si PostgreSQL est accessible"""
    global PG_AVAILABLE
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute('SELECT 1')
            PG_AVAILABLE = True
            logger.info("✅ PostgreSQL disponible")
            return True
    except Exception as e:
        PG_AVAILABLE = False
        logger.warning(f"⚠️  PostgreSQL non disponible: {e}")
        logger.info("📝 Fallback sur fichiers JSON")
        return False


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
    """Charger trades depuis JSON (fallback)"""
    trades = []
    
    try:
        if not TRADES_LOG_DIR.exists():
            logger.warning(f"⚠️  Dossier {TRADES_LOG_DIR} n'existe pas")
            return []
        
        for json_file in sorted(TRADES_LOG_DIR.glob('trades_*.json'), reverse=True):
            try:
                with open(json_file, 'r') as f:
                    file_trades = json.load(f)
                    trades.extend(file_trades)
            except Exception as e:
                logger.error(f"❌ Erreur lecture {json_file}: {e}")
        
        if days and trades:
            cutoff = datetime.now() - timedelta(days=days)
            trades = [
                t for t in trades
                if datetime.fromisoformat(t['timestamp']) > cutoff
            ]
        
        trades.sort(key=lambda t: t['timestamp'], reverse=True)
        return trades
        
    except Exception as e:
        logger.error(f"❌ Erreur load_trades_from_json: {e}")
        return []


def get_trades_data(days=30, symbol=None):
    """Obtenir les trades (PostgreSQL ou JSON)"""
    if PG_AVAILABLE:
        try:
            trades = get_trade_history(days=days, symbol=symbol)
            logger.debug(f"✅ {len(trades)} trades depuis PostgreSQL")
            return trades
        except Exception as e:
            logger.error(f"❌ Erreur PostgreSQL get_trades: {e}")
            # Fallback JSON
            return load_trades_from_json(days=days)
    else:
        trades = load_trades_from_json(days=days)
        if symbol:
            trades = [t for t in trades if t['symbol'] == symbol]
        return trades


def get_daily_data(days=30):
    """Obtenir résumés quotidiens (PostgreSQL ou JSON)"""
    if PG_AVAILABLE:
        try:
            return get_daily_summary(days=days)
        except Exception as e:
            logger.error(f"❌ Erreur PostgreSQL get_daily: {e}")
            return []
    else:
        # Pas de résumés quotidiens en JSON
        return []


# ========== ROUTES PAGES HTML ==========

@app.route('/')
def index():
    """Page principale"""
    return render_template('index.html')


@app.route('/trades')
def trades_page():
    """Page des trades"""
    return render_template('trades.html')


@app.route('/metrics')
def metrics_page():
    """Page des métriques"""
    return render_template('metrics.html')


# ========== ROUTES API ALPACA ==========

@app.route('/api/account')
def get_account():
    """Infos compte Alpaca"""
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
    """Positions actuelles"""
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
    """Ordres récents"""
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
    """Performances basiques"""
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
            logger.info(f"✅ Position {symbol} fermée")
            return jsonify({'success': True, 'message': f'Position {symbol} fermée'})
        else:
            return jsonify({'success': False, 'error': 'Échec fermeture'}), 400
    except Exception as e:
        logger.error(f"❌ Erreur close_position: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ========== ENDPOINTS ANALYTICS AVANCÉS ==========

@app.route('/api/analytics/advanced')
def api_analytics_advanced():
    """Métriques financières avancées (Sharpe, Sortino, etc.)"""
    try:
        days = int(request.args.get('days', 30))
        
        # Charger les données
        trades = get_trades_data(days=days)
        daily_data = get_daily_data(days=days)
        
        # Calculer les métriques
        analytics = PortfolioAnalytics(trades, daily_data)
        metrics = analytics.get_all_metrics()
        
        return jsonify({
            'success': True,
            'data': metrics,
            'source': 'postgresql' if PG_AVAILABLE else 'json'
        })
    except Exception as e:
        logger.error(f"❌ Erreur /api/analytics/advanced: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/analytics/symbol/<symbol>')
def api_analytics_symbol(symbol):
    """Analytics pour un symbole spécifique"""
    try:
        days = int(request.args.get('days', 30))
        
        trades = get_trades_data(days=days, symbol=symbol)
        
        if not trades:
            return jsonify({
                'success': True,
                'data': {'message': f'Aucun trade pour {symbol}'},
                'count': 0
            })
        
        # Stats basiques
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']
        
        total_volume = sum(float(t['amount']) for t in trades)
        avg_price = sum(float(t['price']) for t in trades) / len(trades)
        
        return jsonify({
            'success': True,
            'data': {
                'symbol': symbol,
                'total_trades': len(trades),
                'buy_count': len(buy_trades),
                'sell_count': len(sell_trades),
                'total_volume': total_volume,
                'avg_price': avg_price,
                'trades': trades[:20]  # 20 plus récents
            }
        })
    except Exception as e:
        logger.error(f"❌ Erreur /api/analytics/symbol: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/db/trades')
def api_db_trades():
    """Historique trades"""
    try:
        days = int(request.args.get('days', 30))
        symbol = request.args.get('symbol', None)
        
        trades = get_trades_data(days=days, symbol=symbol)
        
        return jsonify({
            'success': True,
            'data': trades,
            'count': len(trades),
            'source': 'postgresql' if PG_AVAILABLE else 'json'
        })
    except Exception as e:
        logger.error(f"❌ Erreur /api/db/trades: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/db/statistics')
def api_db_statistics():
    """Statistiques globales"""
    try:
        days = int(request.args.get('days', 30))
        
        if PG_AVAILABLE:
            stats = get_trade_statistics(days=days)
            top_symbols = get_top_symbols(days=days, limit=10)
            win_loss = get_win_loss_ratio(days=days)
        else:
            trades = load_trades_from_json(days=days)
            
            buy_count = len([t for t in trades if t['action'] == 'BUY'])
            sell_count = len([t for t in trades if t['action'] == 'SELL'])
            total_volume = sum(float(t['amount']) for t in trades)
            
            stats = {
                'total_trades': len(trades),
                'buy_count': buy_count,
                'sell_count': sell_count,
                'total_volume': total_volume,
                'avg_trade_size': total_volume / len(trades) if trades else 0
            }
            
            # Top symbols
            symbol_counts = defaultdict(lambda: {'count': 0, 'volume': 0})
            for t in trades:
                symbol_counts[t['symbol']]['count'] += 1
                symbol_counts[t['symbol']]['volume'] += float(t['amount'])
            
            top_symbols = [
                {'symbol': s, 'trade_count': d['count'], 'total_volume': d['volume']}
                for s, d in sorted(symbol_counts.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
            ]
            
            win_loss = {'wins': 0, 'losses': 0, 'total': 0, 'win_rate': 0}
        
        return jsonify({
            'success': True,
            'data': {
                'statistics': stats,
                'top_symbols': top_symbols,
                'win_loss': win_loss
            },
            'source': 'postgresql' if PG_AVAILABLE else 'json'
        })
    except Exception as e:
        logger.error(f"❌ Erreur /api/db/statistics: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/db/evolution')
def api_db_evolution():
    """Évolution portfolio"""
    try:
        days = int(request.args.get('days', 30))
        
        if PG_AVAILABLE:
            evolution = get_portfolio_evolution(days=days)
        else:
            trades = load_trades_from_json(days=days)
            daily_data = defaultdict(lambda: {'trades': 0, 'volume': 0, 'portfolio_value': None})
            
            for trade in trades:
                date = trade['timestamp'][:10]
                daily_data[date]['trades'] += 1
                daily_data[date]['volume'] += float(trade['amount'])
                if trade.get('portfolio_value'):
                    daily_data[date]['portfolio_value'] = trade['portfolio_value']
            
            evolution = [
                {'date': date, **data}
                for date, data in sorted(daily_data.items())
            ]
        
        return jsonify({
            'success': True,
            'data': evolution,
            'source': 'postgresql' if PG_AVAILABLE else 'json'
        })
    except Exception as e:
        logger.error(f"❌ Erreur /api/db/evolution: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/health')
def health_check():
    """Vérifier l'état du système"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'version': '2.0',
        'postgres_available': PG_AVAILABLE,
        'alpaca_connected': alpaca_client is not None
    })


# ========== WEBSOCKET ==========

@socketio.on('connect')
def handle_connect():
    logger.info("🔌 Client WebSocket connecté")
    emit('status', {'message': 'Connecté au serveur v2.0'})


@socketio.on('disconnect')
def handle_disconnect():
    logger.info("🔌 Client WebSocket déconnecté")


if __name__ == '__main__':
    logger.info("="*70)
    logger.info("🚀 DÉMARRAGE DU DASHBOARD PLOUTOS V2.0")
    logger.info("="*70)
    
    # Vérifier PostgreSQL
    check_postgres()
    
    # Init Alpaca
    if init_alpaca():
        logger.info("✅ Dashboard v2.0 prêt sur http://0.0.0.0:5000")
        logger.info(f"📊 Source données: {'PostgreSQL' if PG_AVAILABLE else 'JSON (fallback)'}")
        logger.info("🔥 Nouvelles fonctionnalités:")
        logger.info("   - Métriques avancées (Sharpe/Sortino/Calmar)")
        logger.info("   - Analyse drawdown et risque")
        logger.info("   - Analytics par symbole")
        logger.info("   - Pages /trades et /metrics complètes")
        logger.info("="*70)
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    else:
        logger.error("❌ Échec démarrage dashboard")
