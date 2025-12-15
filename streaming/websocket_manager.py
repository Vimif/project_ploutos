"""
🔗 WEBSOCKET MANAGER - Singleton pour gérer UNE SEULE connexion Alpaca WebSocket

Problème résolu :
- Alpaca limite à 1 connexion WebSocket par compte (Paper Trading)
- Chaque LiveAnalyzer créait sa propre connexion → ERREUR "connection limit exceeded"

Solution :
- Singleton pattern : 1 seule instance de connexion WebSocket partagée
- Multiplexage des callbacks : plusieurs consommateurs sur la même connexion
- Auto-reconnexion en cas de déconnexion
"""

import asyncio
import logging
from typing import Dict, List, Callable, Optional
from collections import defaultdict
import threading

try:
    from alpaca.data.live import StockDataStream
    from alpaca.data.models import Bar
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    StockDataStream = None
    Bar = None

from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Singleton pour gérer UNE SEULE connexion WebSocket Alpaca partagée.
    
    Usage:
        manager = WebSocketManager.get_instance()
        manager.subscribe('NVDA', my_callback)
        await manager.start()
    """
    
    _instance: Optional['WebSocketManager'] = None
    _lock = threading.Lock()
    
    def __init__(self):
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-py non installé")
        
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            raise ValueError("⚠️  Clés Alpaca non configurées (ALPACA_API_KEY, ALPACA_SECRET_KEY)")
        
        # Connexion WebSocket unique
        self.stream = StockDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        
        # Callbacks par ticker : {"NVDA": [callback1, callback2], ...}
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Tickers actuellement souscrits
        self.subscribed_tickers: set = set()
        
        # État
        self.is_running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
        logger.info("✅ WebSocketManager initialisé (singleton)")
    
    @classmethod
    def get_instance(cls) -> 'WebSocketManager':
        """
        Récupère l'instance unique (singleton pattern).
        Thread-safe avec double-checked locking.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """
        Détruit l'instance (utile pour tests ou redémarrage complet).
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance.stop()
                cls._instance = None
                logger.warning("♻️ WebSocketManager reset")
    
    def subscribe(self, ticker: str, callback: Callable[[Bar], None]):
        """
        Abonne un callback à un ticker.
        
        Args:
            ticker: Symbole boursier (ex: "NVDA")
            callback: Fonction appelée à chaque nouvelle barre reçue
        """
        ticker = ticker.upper()
        
        # Ajouter le callback
        if callback not in self.callbacks[ticker]:
            self.callbacks[ticker].append(callback)
            logger.info(f"📡 Callback ajouté pour {ticker} (total: {len(self.callbacks[ticker])})")
        
        # S'abonner au ticker si pas déjà fait
        if ticker not in self.subscribed_tickers:
            self.subscribed_tickers.add(ticker)
            
            # Définir le handler pour ce ticker
            async def bar_handler(bar: Bar):
                """Handler global qui dispatche vers tous les callbacks"""
                for cb in self.callbacks[ticker]:
                    try:
                        # Exécuter le callback (peut être sync ou async)
                        if asyncio.iscoroutinefunction(cb):
                            await cb(bar)
                        else:
                            cb(bar)
                    except Exception as e:
                        logger.error(f"❌ Erreur callback {ticker}: {e}")
            
            # Enregistrer le handler dans Alpaca
            self.stream.subscribe_bars(bar_handler, ticker)
            logger.info(f"✅ Abonné à {ticker} sur WebSocket Alpaca")
    
    def unsubscribe(self, ticker: str, callback: Optional[Callable] = None):
        """
        Désabonne un callback d'un ticker.
        
        Args:
            ticker: Symbole boursier
            callback: Callback spécifique à retirer (None = tous)
        """
        ticker = ticker.upper()
        
        if ticker not in self.callbacks:
            return
        
        if callback is None:
            # Retirer tous les callbacks pour ce ticker
            self.callbacks[ticker].clear()
            logger.info(f"🚫 Tous les callbacks retirés pour {ticker}")
        else:
            # Retirer un callback spécifique
            if callback in self.callbacks[ticker]:
                self.callbacks[ticker].remove(callback)
                logger.info(f"🚫 Callback retiré pour {ticker}")
        
        # Si plus aucun callback, se désabonner du ticker
        if len(self.callbacks[ticker]) == 0:
            self.stream.unsubscribe_bars(ticker)
            self.subscribed_tickers.discard(ticker)
            del self.callbacks[ticker]
            logger.info(f"📡 Désabonnement Alpaca de {ticker}")
    
    async def start(self):
        """
        Démarre la connexion WebSocket (bloquant).
        
        IMPORTANT: Utiliser dans un thread séparé ou avec asyncio.create_task()
        """
        if self.is_running:
            logger.warning("⚠️ WebSocket déjà en cours d'exécution")
            return
        
        self.is_running = True
        logger.info("🚀 Démarrage WebSocket Alpaca...")
        
        try:
            await self.stream.run()
        except Exception as e:
            logger.error(f"❌ Erreur WebSocket: {e}", exc_info=True)
            self.is_running = False
            
            # Auto-reconnexion
            if self.reconnect_attempts < self.max_reconnect_attempts:
                self.reconnect_attempts += 1
                wait_time = min(2 ** self.reconnect_attempts, 60)  # Exponential backoff
                logger.warning(f"🔄 Reconnexion dans {wait_time}s (tentative {self.reconnect_attempts}/{self.max_reconnect_attempts})")
                await asyncio.sleep(wait_time)
                await self.start()
            else:
                logger.error("❌ Nombre max de reconnexions atteint")
                raise
    
    def stop(self):
        """
        Arrête la connexion WebSocket.
        """
        if not self.is_running:
            logger.warning("⚠️ WebSocket déjà arrêté")
            return
        
        try:
            self.stream.stop()
            self.is_running = False
            self.reconnect_attempts = 0
            logger.info("🛑 WebSocket arrêté")
        except Exception as e:
            logger.error(f"❌ Erreur arrêt WebSocket: {e}")
    
    def get_stats(self) -> dict:
        """
        Retourne les statistiques de connexion.
        """
        return {
            'is_running': self.is_running,
            'subscribed_tickers': list(self.subscribed_tickers),
            'total_callbacks': sum(len(cbs) for cbs in self.callbacks.values()),
            'reconnect_attempts': self.reconnect_attempts,
            'callbacks_by_ticker': {ticker: len(cbs) for ticker, cbs in self.callbacks.items()}
        }


# === EXEMPLE D'UTILISATION ===
if __name__ == "__main__":
    async def main():
        # Récupérer l'instance unique
        manager = WebSocketManager.get_instance()
        
        # Définir un callback
        def on_nvda_bar(bar: Bar):
            print(f"NVDA: {bar.close} à {bar.timestamp}")
        
        # S'abonner
        manager.subscribe('NVDA', on_nvda_bar)
        manager.subscribe('AAPL', lambda bar: print(f"AAPL: {bar.close}"))
        
        # Démarrer (bloquant)
        await manager.start()
    
    asyncio.run(main())
