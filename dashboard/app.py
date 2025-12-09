#!/usr/bin/env python3
"""
🏛️ PLOUTOS DASHBOARD V4 - Interface Web Moderne

Dashboard Flask pour visualiser:
- Portfolio en temps réel
- Trades (historique + actifs)
- Métriques performance
- Prédictions modèle
- Graphiques interactifs

Port: 5000 (local) ou 8080 (production)
API: /api/status, /api/trades, /api/metrics

Auteur: Ploutos AI Team
Date: 9 Dec 2025
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

app = Flask(__name__)
CORS(app)  # Enable CORS

# ============================================================================
# CONFIG
# ============================================================================

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'ploutos'),
    'user': os.getenv('DB_USER', 'ploutos'),
    'password': os.getenv('DB_PASSWORD', 'your_password')  # Change in prod
}

# ============================================================================
# DATABASE HELPERS
# ============================================================================

def get_db_connection():
    """Connexion PostgreSQL"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"❌ Erreur connexion DB: {e}")
        return None

def execute_query(query, params=None, fetch=True):
    """Exécute requête SQL"""
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(query, params)
        
        if fetch:
            result = cur.fetchall()
        else:
            conn.commit()
            result = True
        
        cur.close()
        conn.close()
        return result
    except Exception as e:
        print(f"❌ Erreur query: {e}")
        if conn:
            conn.close()
        return None

# ============================================================================
# ROUTES WEB
# ============================================================================

@app.route('/')
def index():
    """Page d'accueil dashboard"""
    return render_template('index.html')

@app.route('/trades')
def trades_page():
    """Page historique trades"""
    return render_template('trades.html')

@app.route('/metrics')
def metrics_page():
    """Page métriques performance"""
    return render_template('metrics.html')

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/api/status')
def api_status():
    """Status bot + portfolio actuel"""
    
    # Portfolio actuel
    portfolio_query = """
        SELECT 
            balance,
            portfolio_value,
            (portfolio_value - 100000) / 100000 * 100 as return_pct,
            timestamp
        FROM daily_summary
        ORDER BY timestamp DESC
        LIMIT 1
    """
    portfolio = execute_query(portfolio_query)
    
    # Trades aujourd'hui
    today_trades_query = """
        SELECT COUNT(*) as count
        FROM trades
        WHERE DATE(entry_time) = CURRENT_DATE
    """
    today_trades = execute_query(today_trades_query)
    
    # Positions ouvertes
    positions_query = """
        SELECT 
            ticker,
            shares,
            entry_price,
            current_price,
            (current_price - entry_price) / entry_price * 100 as pnl_pct
        FROM trades
        WHERE exit_time IS NULL
        ORDER BY entry_time DESC
    """
    positions = execute_query(positions_query)
    
    # Dernières prédictions
    predictions_query = """
        SELECT 
            ticker,
            action,
            confidence,
            timestamp
        FROM predictions
        ORDER BY timestamp DESC
        LIMIT 10
    """
    predictions = execute_query(predictions_query)
    
    return jsonify({
        'success': True,
        'timestamp': datetime.now().isoformat(),
        'portfolio': portfolio[0] if portfolio else None,
        'trades_today': today_trades[0]['count'] if today_trades else 0,
        'positions': positions or [],
        'recent_predictions': predictions or []
    })

@app.route('/api/trades')
def api_trades():
    """Historique trades avec filtres"""
    
    # Params
    days = request.args.get('days', 30, type=int)
    ticker = request.args.get('ticker', None)
    limit = request.args.get('limit', 100, type=int)
    
    # Query
    query = """
        SELECT 
            id,
            ticker,
            action,
            shares,
            entry_price,
            exit_price,
            entry_time,
            exit_time,
            pnl,
            pnl_pct,
            commission
        FROM trades
        WHERE entry_time >= NOW() - INTERVAL '%s days'
    """
    params = [days]
    
    if ticker:
        query += " AND ticker = %s"
        params.append(ticker)
    
    query += " ORDER BY entry_time DESC LIMIT %s"
    params.append(limit)
    
    trades = execute_query(query, params)
    
    # Stats
    stats_query = """
        SELECT 
            COUNT(*) as total_trades,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
            AVG(pnl_pct) as avg_pnl_pct,
            MAX(pnl_pct) as max_win,
            MIN(pnl_pct) as max_loss,
            SUM(pnl) as total_pnl
        FROM trades
        WHERE entry_time >= NOW() - INTERVAL '%s days'
    """
    stats_params = [days]
    
    if ticker:
        stats_query += " AND ticker = %s"
        stats_params.append(ticker)
    
    stats = execute_query(stats_query, stats_params)
    
    return jsonify({
        'success': True,
        'trades': trades or [],
        'stats': stats[0] if stats else {},
        'filters': {
            'days': days,
            'ticker': ticker,
            'limit': limit
        }
    })

@app.route('/api/metrics')
def api_metrics():
    """Métriques performance"""
    
    days = request.args.get('days', 90, type=int)
    
    # Performance quotidienne
    daily_query = """
        SELECT 
            DATE(timestamp) as date,
            portfolio_value,
            balance,
            daily_return,
            n_trades
        FROM daily_summary
        WHERE timestamp >= NOW() - INTERVAL '%s days'
        ORDER BY date ASC
    """
    daily = execute_query(daily_query, [days])
    
    # Calcul métriques
    if daily and len(daily) > 0:
        returns = [d['daily_return'] for d in daily if d['daily_return'] is not None]
        
        import numpy as np
        
        total_return = (daily[-1]['portfolio_value'] - 100000) / 100000 * 100
        
        # Sharpe
        if len(returns) > 0:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Max Drawdown
        portfolio_values = [d['portfolio_value'] for d in daily]
        peak = portfolio_values[0]
        max_dd = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (value - peak) / peak
            if dd < max_dd:
                max_dd = dd
        
        max_drawdown = abs(max_dd) * 100
        
        metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'current_value': daily[-1]['portfolio_value'],
            'total_trades': sum(d['n_trades'] or 0 for d in daily),
            'avg_trades_per_day': np.mean([d['n_trades'] or 0 for d in daily])
        }
    else:
        metrics = {}
    
    return jsonify({
        'success': True,
        'metrics': metrics,
        'daily_data': daily or [],
        'period_days': days
    })

@app.route('/api/predictions')
def api_predictions():
    """Prédictions récentes modèle"""
    
    limit = request.args.get('limit', 50, type=int)
    
    query = """
        SELECT 
            ticker,
            action,
            confidence,
            timestamp,
            features
        FROM predictions
        ORDER BY timestamp DESC
        LIMIT %s
    """
    predictions = execute_query(query, [limit])
    
    # Distribution actions
    dist_query = """
        SELECT 
            action,
            COUNT(*) as count
        FROM predictions
        WHERE timestamp >= NOW() - INTERVAL '24 hours'
        GROUP BY action
    """
    distribution = execute_query(dist_query)
    
    return jsonify({
        'success': True,
        'predictions': predictions or [],
        'distribution_24h': distribution or []
    })

@app.route('/api/tickers')
def api_tickers():
    """Liste tickers surveillés"""
    
    query = """
        SELECT DISTINCT ticker
        FROM trades
        ORDER BY ticker
    """
    tickers = execute_query(query)
    
    return jsonify({
        'success': True,
        'tickers': [t['ticker'] for t in tickers] if tickers else []
    })

@app.route('/api/health')
def api_health():
    """Healthcheck"""
    
    # Test DB
    db_ok = get_db_connection() is not None
    
    return jsonify({
        'success': True,
        'status': 'healthy' if db_ok else 'degraded',
        'database': 'ok' if db_ok else 'error',
        'timestamp': datetime.now().isoformat()
    })

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Vérifier DB
    print("⚡ Vérification connexion DB...")
    conn = get_db_connection()
    if conn:
        print("✅ DB connectée")
        conn.close()
    else:
        print("❌ DB non accessible (dashboard fonctionne en mode dégradé)")
    
    # Run
    print("\n🚀 Démarrage dashboard...")
    print("   URL: http://localhost:5000")
    print("   API: http://localhost:5000/api/status")
    
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('DEBUG', 'True').lower() == 'true'
    )
