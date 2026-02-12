#!/usr/bin/env python3
"""
📅 MARKET STATUS CHECKER

Vérifie si le marché US (NYSE/NASDAQ) est ouvert
Utilise l'API Alpaca pour obtenir les horaires exacts

Auteur: Ploutos AI Team
Date: Dec 2025
"""

from datetime import datetime, timedelta
import logging
import pytz

logger = logging.getLogger(__name__)

class MarketStatus:
    """
    Vérifie l'état du marché (ouvert/fermé)
    Cache les résultats pour éviter trop de requêtes API
    """
    
    def __init__(self, alpaca_client=None):
        """
        Args:
            alpaca_client: Instance de AlpacaClient (optionnel)
        """
        self.client = alpaca_client
        self._cache = None
        self._cache_time = None
        self._cache_duration = 60  # Cache pendant 60 secondes
    
    def is_market_open(self, use_cache=True):
        """
        Vérifie si le marché est actuellement ouvert
        
        Args:
            use_cache: Utiliser le cache si disponible
        
        Returns:
            bool: True si ouvert, False sinon
        """
        status = self.get_market_status(use_cache=use_cache)
        return status.get('is_open', False) if status else False
    
    def get_market_status(self, use_cache=True):
        """
        Obtenir le statut complet du marché
        
        Args:
            use_cache: Utiliser le cache si disponible
        
        Returns:
            dict: {
                'is_open': bool,
                'next_open': datetime,
                'next_close': datetime,
                'time_until_open': str (ex: "2h 30min"),
                'time_until_close': str,
                'timestamp': datetime
            }
        """
        # Vérifier cache
        if use_cache and self._cache and self._cache_time:
            age = (datetime.now() - self._cache_time).total_seconds()
            if age < self._cache_duration:
                return self._cache
        
        # Pas de cache valide, interroger API
        status = self._fetch_market_status()
        
        # Mettre en cache
        if status:
            self._cache = status
            self._cache_time = datetime.now()
        
        return status
    
    def _fetch_market_status(self):
        """
        Interroger l'API Alpaca pour le statut
        
        Returns:
            dict ou None
        """
        if not self.client:
            logger.warning("⚠️  Pas de client Alpaca, impossible de vérifier le marché")
            return self._fallback_market_check()
        
        try:
            # Récupérer clock Alpaca
            clock = self.client.trading_client.get_clock()
            
            # Convertir en timezone-aware datetimes
            next_open = clock.next_open.replace(tzinfo=pytz.UTC)
            next_close = clock.next_close.replace(tzinfo=pytz.UTC)
            now = datetime.now(pytz.UTC)
            
            # Calculer temps restant
            if clock.is_open:
                delta = next_close - now
                time_str = self._format_timedelta(delta)
                time_until_close = time_str
                time_until_open = None
            else:
                delta = next_open - now
                time_str = self._format_timedelta(delta)
                time_until_open = time_str
                time_until_close = None
            
            return {
                'is_open': clock.is_open,
                'next_open': next_open,
                'next_close': next_close,
                'time_until_open': time_until_open,
                'time_until_close': time_until_close,
                'timestamp': now
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération statut marché: {e}")
            return self._fallback_market_check()
    
    def _fallback_market_check(self):
        """
        Vérification de secours si API non disponible
        Utilise les horaires standard NYSE (approximatif)
        
        Returns:
            dict
        """
        now = datetime.now(pytz.timezone('America/New_York'))
        
        # Marché fermé le weekend
        if now.weekday() >= 5:  # Samedi (5) ou Dimanche (6)
            # Calculer lundi prochain 9:30
            days_until_monday = (7 - now.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 1
            
            next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            next_open += timedelta(days=days_until_monday)
            
            delta = next_open - now
            
            return {
                'is_open': False,
                'next_open': next_open,
                'next_close': None,
                'time_until_open': self._format_timedelta(delta),
                'time_until_close': None,
                'timestamp': now
            }
        
        # En semaine, vérifier horaires 9:30-16:00 EST
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        is_open = market_open <= now < market_close
        
        if is_open:
            delta = market_close - now
            return {
                'is_open': True,
                'next_open': None,
                'next_close': market_close,
                'time_until_open': None,
                'time_until_close': self._format_timedelta(delta),
                'timestamp': now
            }
        else:
            # Marché fermé, calculer prochaine ouverture
            if now < market_open:
                next_open = market_open
            else:
                # Après 16:00, prochaine ouverture demain
                next_open = market_open + timedelta(days=1)
            
            delta = next_open - now
            
            return {
                'is_open': False,
                'next_open': next_open,
                'next_close': None,
                'time_until_open': self._format_timedelta(delta),
                'time_until_close': None,
                'timestamp': now
            }
    
    def _format_timedelta(self, delta):
        """
        Formater un timedelta en chaîne lisible
        
        Args:
            delta: timedelta
        
        Returns:
            str: Ex: "2h 30min", "45min", "3 jours"
        """
        total_seconds = int(delta.total_seconds())
        
        if total_seconds < 0:
            return "0min"
        
        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        minutes = (total_seconds % 3600) // 60
        
        if days > 0:
            if hours > 0:
                return f"{days}j {hours}h"
            return f"{days} jour{'s' if days > 1 else ''}"
        
        if hours > 0:
            if minutes > 0:
                return f"{hours}h {minutes}min"
            return f"{hours}h"
        
        return f"{minutes}min"
    
    def wait_for_market_open(self, check_interval=60):
        """
        Attendre que le marché ouvre (bloquant)
        
        Args:
            check_interval: Intervalle de vérification en secondes
        
        Returns:
            bool: True quand le marché ouvre
        """
        import time
        
        while not self.is_market_open(use_cache=False):
            status = self.get_market_status(use_cache=False)
            
            if status and status.get('time_until_open'):
                logger.info(f"⏳ Marché fermé. Ouverture dans: {status['time_until_open']}")
            
            time.sleep(check_interval)
        
        logger.info("✅ Marché ouvert !")
        return True

# Helper fonction pour utilisation simple
def is_market_open(alpaca_client=None):
    """
    Vérification rapide du statut marché
    
    Args:
        alpaca_client: Instance AlpacaClient (optionnel)
    
    Returns:
        bool: True si ouvert
    """
    checker = MarketStatus(alpaca_client)
    return checker.is_market_open()
