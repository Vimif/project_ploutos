"""
🌍 YAHOO FINANCE STREAM - Alternative pour actions européennes

Problème résolu :
- Alpaca ne supporte QUE les actions US
- Yahoo Finance supporte TOUTES les bourses mondiales (Euronext, LSE, etc.)

Limitations :
- Pas de WebSocket temps réel (polling HTTP toutes les 1-5 min)
- Délai de 15-20 minutes sur les données
- Rate limiting (max ~2000 requêtes/heure)

Usage :
    stream = YahooStreamManager()
    stream.subscribe('DSY.PA', my_callback, interval=60)  # Poll toutes les 60s
    stream.start()
"""

import asyncio
import logging
from typing import Dict, List, Callable, Optional
from collections import defaultdict
from datetime import datetime
import threading
import time

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

logger = logging.getLogger(__name__)


class YahooBar:
    """
    Objet de type Bar compatible avec l'API Alpaca.
    Permet de réutiliser le code existant sans modification.
    """
    def __init__(self, symbol: str, data: dict):
        self.symbol = symbol
        self.timestamp = datetime.now()
        self.open = data.get('open', 0.0)
        self.high = data.get('high', 0.0)
        self.low = data.get('low', 0.0)
        self.close = data.get('close', 0.0)
        self.volume = data.get('volume', 0)
        self.trade_count = None  # Yahoo ne fournit pas cette info
        self.vwap = None  # Yahoo ne fournit pas cette info
    
    def __repr__(self):
        return f"YahooBar({self.symbol}, close={self.close}, time={self.timestamp})"


class YahooStreamManager:
    """
    Gestionnaire de flux Yahoo Finance avec polling HTTP.
    Simule un WebSocket en interrogeant Yahoo Finance périodiquement.
    
    Usage:
        manager = YahooStreamManager()
        manager.subscribe('DSY.PA', my_callback, interval=60)
        manager.start()
    """
    
    _instance: Optional['YahooStreamManager'] = None
    _lock = threading.Lock()
    
    def __init__(self, default_interval: int = 60):
        """
        Args:
            default_interval: Intervalle par défaut entre les polls (secondes)
        """
        if not YFINANCE_AVAILABLE:
            raise ImportError("❌ yfinance non installé. Installer avec: pip install yfinance")
        
        # Callbacks par ticker : {"DSY.PA": [(callback, interval), ...]}
        self.subscriptions: Dict[str, List[tuple]] = defaultdict(list)
        
        # Dernière valeur connue par ticker (pour détecter les changements)
        self.last_prices: Dict[str, float] = {}
        
        # État
        self.is_running = False
        self.default_interval = default_interval
        self._polling_thread: Optional[threading.Thread] = None
        
        logger.info(f"✅ YahooStreamManager initialisé (poll interval: {default_interval}s)")
    
    @classmethod
    def get_instance(cls, default_interval: int = 60) -> 'YahooStreamManager':
        """
        Singleton pattern (optionnel, pour compatibilité avec WebSocketManager).
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(default_interval)
        return cls._instance
    
    def subscribe(self, ticker: str, callback: Callable, interval: Optional[int] = None):
        """
        Abonne un callback à un ticker.
        
        Args:
            ticker: Symbole boursier (ex: "DSY.PA", "MC.PA", "AI.PA")
            callback: Fonction appelée à chaque nouvelle donnée
            interval: Intervalle de polling en secondes (None = utilise default_interval)
        """
        ticker = ticker.upper()
        interval = interval or self.default_interval
        
        # Ajouter le callback
        if (callback, interval) not in self.subscriptions[ticker]:
            self.subscriptions[ticker].append((callback, interval))
            logger.info(f"📡 Callback ajouté pour {ticker} (poll: {interval}s, total: {len(self.subscriptions[ticker])})")
    
    def unsubscribe(self, ticker: str, callback: Optional[Callable] = None):
        """
        Désabonne un callback d'un ticker.
        
        Args:
            ticker: Symbole boursier
            callback: Callback spécifique à retirer (None = tous)
        """
        ticker = ticker.upper()
        
        if ticker not in self.subscriptions:
            return
        
        if callback is None:
            # Retirer tous les callbacks
            self.subscriptions[ticker].clear()
            del self.subscriptions[ticker]
            logger.info(f"🚫 Tous les callbacks retirés pour {ticker}")
        else:
            # Retirer un callback spécifique
            self.subscriptions[ticker] = [(cb, interval) for cb, interval in self.subscriptions[ticker] if cb != callback]
            if not self.subscriptions[ticker]:
                del self.subscriptions[ticker]
            logger.info(f"🚫 Callback retiré pour {ticker}")
    
    def _poll_ticker(self, ticker: str):
        """
        Interroge Yahoo Finance pour un ticker donné.
        Appelle les callbacks si de nouvelles données sont disponibles.
        """
        try:
            # Télécharger les données
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extraire le prix actuel
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            
            if current_price is None:
                logger.warning(f"⚠️ Pas de prix disponible pour {ticker}")
                return
            
            # Vérifier si le prix a changé (pour éviter les appels inutiles)
            if ticker in self.last_prices and self.last_prices[ticker] == current_price:
                # Pas de changement, skip
                return
            
            # Créer un objet Bar
            bar_data = {
                'open': info.get('regularMarketOpen', current_price),
                'high': info.get('dayHigh', current_price),
                'low': info.get('dayLow', current_price),
                'close': current_price,
                'volume': info.get('volume', 0)
            }
            
            bar = YahooBar(ticker, bar_data)
            
            # Mettre à jour le dernier prix
            self.last_prices[ticker] = current_price
            
            # Appeler tous les callbacks
            for callback, _ in self.subscriptions.get(ticker, []):
                try:
                    if asyncio.iscoroutinefunction(callback):
                        # Callback async
                        asyncio.run(callback(bar))
                    else:
                        # Callback sync
                        callback(bar)
                except Exception as e:
                    logger.error(f"❌ Erreur callback {ticker}: {e}", exc_info=True)
            
            logger.debug(f"📊 {ticker}: {current_price:.2f} (volume: {bar_data['volume']:,})")
            
        except Exception as e:
            logger.error(f"❌ Erreur polling {ticker}: {e}", exc_info=True)
    
    def _polling_loop(self):
        """
        Boucle principale de polling.
        Interroge chaque ticker à l'intervalle défini.
        """
        logger.info("🚀 Boucle de polling Yahoo Finance démarrée")
        
        # Dernière exécution par ticker
        last_poll_time: Dict[str, float] = {}
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Pour chaque ticker abonné
                for ticker, callbacks in list(self.subscriptions.items()):
                    # Trouver l'intervalle minimum demandé pour ce ticker
                    min_interval = min(interval for _, interval in callbacks)
                    
                    # Vérifier si c'est le moment de poller
                    last_time = last_poll_time.get(ticker, 0)
                    if current_time - last_time >= min_interval:
                        self._poll_ticker(ticker)
                        last_poll_time[ticker] = current_time
                
                # Attendre 1 seconde avant la prochaine itération
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Erreur boucle polling: {e}", exc_info=True)
                time.sleep(5)  # Attendre avant de réessayer
        
        logger.info("🛑 Boucle de polling arrêtée")
    
    def start(self):
        """
        Démarre le polling dans un thread séparé.
        """
        if self.is_running:
            logger.warning("⚠️ Polling déjà actif")
            return
        
        if not self.subscriptions:
            logger.warning("⚠️ Aucun ticker abonné, rien à poller")
            return
        
        self.is_running = True
        self._polling_thread = threading.Thread(
            target=self._polling_loop,
            daemon=True,
            name="YahooPolling"
        )
        self._polling_thread.start()
        logger.info(f"✅ Polling démarré pour {len(self.subscriptions)} ticker(s)")
    
    def stop(self):
        """
        Arrête le polling.
        """
        if not self.is_running:
            logger.warning("⚠️ Polling déjà arrêté")
            return
        
        self.is_running = False
        if self._polling_thread:
            self._polling_thread.join(timeout=5)
        
        logger.info("🛑 Polling Yahoo Finance arrêté")
    
    def get_stats(self) -> dict:
        """
        Retourne les statistiques.
        """
        return {
            'is_running': self.is_running,
            'subscribed_tickers': list(self.subscriptions.keys()),
            'total_callbacks': sum(len(cbs) for cbs in self.subscriptions.values()),
            'last_prices': self.last_prices,
            'default_interval': self.default_interval
        }


# === EXEMPLE D'UTILISATION ===
if __name__ == "__main__":
    # Créer le manager
    manager = YahooStreamManager(default_interval=10)  # Poll toutes les 10s
    
    # Définir un callback
    def on_bar(bar: YahooBar):
        print(f"📊 {bar.symbol}: {bar.close:.2f}€ (volume: {bar.volume:,})")
    
    # S'abonner à des actions européennes
    manager.subscribe('DSY.PA', on_bar)      # Dassault Systèmes
    manager.subscribe('MC.PA', on_bar)       # LVMH
    manager.subscribe('AI.PA', on_bar)       # Air Liquide
    manager.subscribe('OR.PA', on_bar)       # L'Oréal
    
    # Démarrer
    manager.start()
    
    # Laisser tourner
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Arrêt...")
        manager.stop()
