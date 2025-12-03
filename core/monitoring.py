# core/monitoring.py
"""Monitoring et métriques Prometheus pour Ploutos Trading"""

import sys
from pathlib import Path
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import time
from datetime import datetime
from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    start_http_server, CollectorRegistry, REGISTRY
)
from core.utils import setup_logging

logger = setup_logging(__name__)

class TradingMetrics:
    """Métriques Prometheus pour le trading"""
    
    def __init__(self, port: int = 9090):
        self.port = port
        self.started = False
        
        # ========== COMPTEURS ==========
        self.trades_total = Counter(
            'ploutos_trades_total',
            'Nombre total de trades',
            ['action', 'symbol', 'result']
        )
        
        self.predictions_total = Counter(
            'ploutos_predictions_total',
            'Nombre total de prédictions',
            ['action', 'sector']
        )
        
        self.alerts_total = Counter(
            'ploutos_alerts_total',
            'Nombre total d\'alertes',
            ['priority', 'type']
        )
        
        self.errors_total = Counter(
            'ploutos_errors_total',
            'Nombre total d\'erreurs',
            ['component', 'error_type']
        )
        
        # ========== GAUGES (valeurs instantanées) ==========
        self.portfolio_value = Gauge(
            'ploutos_portfolio_value_usd',
            'Valeur totale du portfolio en USD'
        )
        
        self.cash_available = Gauge(
            'ploutos_cash_available_usd',
            'Cash disponible en USD'
        )
        
        self.buying_power = Gauge(
            'ploutos_buying_power_usd',
            'Buying power disponible en USD'
        )
        
        self.positions_count = Gauge(
            'ploutos_positions_count',
            'Nombre de positions ouvertes'
        )
        
        self.positions_value = Gauge(
            'ploutos_positions_value_usd',
            'Valeur totale des positions en USD'
        )
        
        self.daily_pnl = Gauge(
            'ploutos_daily_pnl_usd',
            'Profit & Loss quotidien en USD'
        )
        
        self.daily_pnl_percent = Gauge(
            'ploutos_daily_pnl_percent',
            'Profit & Loss quotidien en pourcentage'
        )
        
        self.total_pnl = Gauge(
            'ploutos_total_pnl_usd',
            'Profit & Loss total depuis le début'
        )
        
        self.unrealized_pnl = Gauge(
            'ploutos_unrealized_pnl_usd',
            'Profit & Loss non réalisé total'
        )
        
        self.win_rate = Gauge(
            'ploutos_win_rate_percent',
            'Taux de réussite des trades en %'
        )
        
        self.exposure = Gauge(
            'ploutos_exposure_percent',
            'Exposition du portfolio en %'
        )
        
        # Position individuelle
        self.position_pnl = Gauge(
            'ploutos_position_pnl_usd',
            'P&L d\'une position spécifique',
            ['symbol']
        )
        
        self.position_size = Gauge(
            'ploutos_position_size_usd',
            'Taille d\'une position spécifique',
            ['symbol']
        )
        
        # Risk Management
        self.circuit_breaker = Gauge(
            'ploutos_circuit_breaker_active',
            'État du circuit breaker (0=off, 1=on)'
        )
        
        self.risky_positions_count = Gauge(
            'ploutos_risky_positions_count',
            'Nombre de positions à risque élevé'
        )
        
        self.max_drawdown = Gauge(
            'ploutos_max_drawdown_percent',
            'Maximum drawdown en %'
        )
        
        self.sharpe_ratio = Gauge(
            'ploutos_sharpe_ratio',
            'Sharpe ratio du portfolio'
        )
        
        # ========== HISTOGRAMMES (distributions) ==========
        self.trade_amount = Histogram(
            'ploutos_trade_amount_usd',
            'Distribution des montants de trades',
            buckets=[100, 500, 1000, 2500, 5000, 10000, 25000, 50000]
        )
        
        self.trade_latency = Histogram(
            'ploutos_trade_latency_seconds',
            'Latence d\'exécution des trades',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        self.position_hold_time = Histogram(
            'ploutos_position_hold_time_hours',
            'Durée de détention des positions',
            buckets=[1, 6, 12, 24, 48, 72, 168, 336]  # jusqu'à 2 semaines
        )
        
        # ========== SUMMARIES ==========
        self.api_request_duration = Summary(
            'ploutos_api_request_duration_seconds',
            'Durée des requêtes API',
            ['endpoint', 'method']
        )
        
        logger.info("📊 Métriques Prometheus initialisées")
    
    def start_server(self):
        """Démarrer le serveur Prometheus"""
        if self.started:
            logger.warning("⚠️  Serveur Prometheus déjà démarré")
            return
        
        try:
            start_http_server(self.port)
            self.started = True
            logger.info(f"✅ Serveur Prometheus démarré sur port {self.port}")
            logger.info(f"📊 Métriques disponibles: http://localhost:{self.port}/metrics")
        except Exception as e:
            logger.error(f"❌ Erreur démarrage Prometheus: {e}")
    
    # ========== MÉTHODES D'ENREGISTREMENT ==========
    
    def record_trade(self, symbol: str, action: str, amount: float, 
                    execution_time: float = 0.0, result: str = 'pending'):
        """
        Enregistrer un trade
        
        Args:
            symbol: Symbole tradé
            action: BUY ou SELL
            amount: Montant en USD
            execution_time: Temps d'exécution en secondes
            result: success, failed, pending
        """
        self.trades_total.labels(action=action, symbol=symbol, result=result).inc()
        self.trade_amount.observe(amount)
        
        if execution_time > 0:
            self.trade_latency.observe(execution_time)
        
        logger.debug(f"📊 Trade enregistré: {action} {symbol} ${amount:,.2f}")
    
    def record_prediction(self, action: str, sector: str):
        """Enregistrer une prédiction du modèle"""
        self.predictions_total.labels(action=action, sector=sector).inc()
    
    def record_alert(self, priority: str, alert_type: str):
        """Enregistrer une alerte envoyée"""
        self.alerts_total.labels(priority=priority, type=alert_type).inc()
    
    def record_error(self, component: str, error_type: str):
        """Enregistrer une erreur"""
        self.errors_total.labels(component=component, error_type=error_type).inc()
    
    def update_portfolio_metrics(self, account: dict, positions: list):
        """
        Mettre à jour toutes les métriques du portfolio
        
        Args:
            account: Dict avec infos compte Alpaca
            positions: Liste des positions ouvertes
        """
        # Métriques compte
        self.portfolio_value.set(float(account.get('portfolio_value', 0)))
        self.cash_available.set(float(account.get('cash', 0)))
        self.buying_power.set(float(account.get('buying_power', 0)))
        
        # Métriques positions
        self.positions_count.set(len(positions))
        
        total_positions_value = sum(float(p.get('market_value', 0)) for p in positions)
        self.positions_value.set(total_positions_value)
        
        # P&L non réalisé
        total_unrealized_pl = sum(float(p.get('unrealized_pl', 0)) for p in positions)
        self.unrealized_pnl.set(total_unrealized_pl)
        
        # Exposition
        if account.get('portfolio_value', 0) > 0:
            exposure_pct = (total_positions_value / float(account['portfolio_value'])) * 100
            self.exposure.set(exposure_pct)
        
        # Positions individuelles
        for pos in positions:
            symbol = pos.get('symbol')
            if symbol:
                self.position_pnl.labels(symbol=symbol).set(float(pos.get('unrealized_pl', 0)))
                self.position_size.labels(symbol=symbol).set(float(pos.get('market_value', 0)))
        
        logger.debug(f"📊 Métriques portfolio mises à jour: ${account.get('portfolio_value', 0):,.2f}")
    
    def update_daily_metrics(self, initial_value: float, current_value: float):
        """Mettre à jour les métriques quotidiennes"""
        daily_pl = current_value - initial_value
        daily_pl_pct = (daily_pl / initial_value * 100) if initial_value > 0 else 0
        
        self.daily_pnl.set(daily_pl)
        self.daily_pnl_percent.set(daily_pl_pct)
    
    def update_performance_metrics(self, win_rate: float, sharpe: float = 0.0, 
                                   max_dd: float = 0.0):
        """Mettre à jour les métriques de performance"""
        self.win_rate.set(win_rate)
        
        if sharpe != 0:
            self.sharpe_ratio.set(sharpe)
        
        if max_dd != 0:
            self.max_drawdown.set(abs(max_dd) * 100)  # En pourcentage positif
    
    def update_risk_metrics(self, circuit_breaker_active: bool, 
                           risky_positions: int):
        """Mettre à jour les métriques de risque"""
        self.circuit_breaker.set(1 if circuit_breaker_active else 0)
        self.risky_positions_count.set(risky_positions)
    
    def record_api_call(self, endpoint: str, method: str, duration: float):
        """Enregistrer un appel API"""
        self.api_request_duration.labels(endpoint=endpoint, method=method).observe(duration)

# Instance globale
_metrics = None

def get_metrics(port: int = 9090) -> TradingMetrics:
    """Obtenir l'instance des métriques (singleton)"""
    global _metrics
    if _metrics is None:
        _metrics = TradingMetrics(port=port)
    return _metrics

def start_monitoring(port: int = 9090):
    """Démarrer le monitoring"""
    metrics = get_metrics(port)
    metrics.start_server()
    return metrics