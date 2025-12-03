# core/metrics.py
"""Métriques Prometheus pour monitoring"""

from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Métriques
trades_total = Counter('ploutos_trades_total', 'Nombre total de trades', ['action', 'symbol'])
portfolio_value = Gauge('ploutos_portfolio_value', 'Valeur du portfolio')
positions_count = Gauge('ploutos_positions_count', 'Nombre de positions')
trade_pl = Histogram('ploutos_trade_pl', 'P&L des trades')

def start_metrics_server(port=9090):
    """Démarrer le serveur de métriques"""
    start_http_server(port)

def record_trade(symbol, action):
    trades_total.labels(action=action, symbol=symbol).inc()

def update_portfolio(value):
    portfolio_value.set(value)

def update_positions(count):
    positions_count.set(count)

def record_pl(pl):
    trade_pl.observe(pl)