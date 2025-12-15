"""
🔥 LIVE ANALYZER - Analyse temps réel via Alpaca WebSocket

Connecte au flux Alpaca WebSocket, reçoit les données en temps réel,
et utilise SignalDetector pour générer des alertes BUY/SELL.
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import List, Dict, Callable
import json

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alpaca.data.live import StockDataStream
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed  # ✅ FIX: Import DataFeed enum
from streaming.signal_detector import SignalDetector
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY


class LiveAnalyzer:
    """Analyseur temps réel avec Alpaca WebSocket"""
    
    def __init__(self, tickers: List[str], timeframe_minutes: int = 1):
        """
        Args:
            tickers: Liste des tickers à surveiller
            timeframe_minutes: Intervalle des barres (1, 5, 15 min)
        """
        self.tickers = tickers
        self.timeframe_minutes = timeframe_minutes
        
        # WebSocket Alpaca
        self.stream = StockDataStream(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            feed=DataFeed.IEX  # ✅ FIX: Utiliser l'enum au lieu de string
        )
        
        # Créer un détecteur par ticker
        self.detectors = {
            ticker: SignalDetector(ticker, window_size=50)
            for ticker in tickers
        }
        
        # Callbacks pour les signaux
        self.signal_callbacks: List[Callable] = []
        
        # Statistiques
        self.total_bars_received = 0
        self.total_signals_generated = 0
        self.start_time = None
        
        print(f"✅ LiveAnalyzer initialisé pour {len(tickers)} tickers (timeframe: {timeframe_minutes}min)")
    
    
    async def on_bar(self, bar):
        """
        Callback appelé à chaque nouvelle barre reçue
        
        Args:
            bar: Objet Bar d'Alpaca contenant OHLCV
        """
        ticker = bar.symbol
        self.total_bars_received += 1
        
        # Vérifier que le ticker est surveillé
        if ticker not in self.detectors:
            return
        
        # Analyser la barre
        detector = self.detectors[ticker]
        signal = detector.add_bar(
            timestamp=bar.timestamp,
            open_price=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume
        )
        
        # Afficher info
        print(f"[{bar.timestamp.strftime('%H:%M:%S')}] {ticker}: ${bar.close:.2f} | Vol: {bar.volume:,}")
        
        # Si signal important (pas HOLD ou WAIT)
        if signal["signal"] in ["BUY", "SELL"] and signal["confidence"] >= 60:
            self.total_signals_generated += 1
            self._handle_signal(signal)
    
    
    def _handle_signal(self, signal: dict):
        """
        Traite un signal BUY/SELL détecté
        
        Args:
            signal: Dictionnaire du signal généré
        """
        # Affichage console avec couleurs
        color = "\033[92m" if signal["signal"] == "BUY" else "\033[91m"  # Vert/Rouge
        reset = "\033[0m"
        
        print(f"\n{'='*80}")
        print(f"{color}🚨 SIGNAL {signal['signal']} DÉTECTÉ ! {reset}")
        print(f"Ticker: {signal['ticker']}")
        print(f"Prix: ${signal['current_price']:.2f}")
        print(f"Confidence: {signal['confidence']}%")
        print(f"Raisons:")
        for reason in signal['reasons']:
            print(f"  - {reason}")
        print(f"Indicateurs:")
        for key, value in signal['indicators'].items():
            print(f"  {key}: {value}")
        print(f"{'='*80}\n")
        
        # Appeler les callbacks enregistrés (pour dashboard, notifications, etc.)
        for callback in self.signal_callbacks:
            try:
                callback(signal)
            except Exception as e:
                print(f"❌ Erreur callback: {e}")
    
    
    def add_signal_callback(self, callback: Callable):
        """
        Enregistre un callback appelé à chaque signal
        
        Args:
            callback: Fonction(signal: dict) -> None
        """
        self.signal_callbacks.append(callback)
        print(f"✅ Callback enregistré: {callback.__name__}")
    
    
    async def start(self):
        """
        Démarre le monitoring temps réel
        """
        self.start_time = datetime.now()
        
        print(f"\n🚀 Démarrage du Live Analyzer...")
        print(f"Tickers surveillés: {', '.join(self.tickers)}")
        print(f"Timeframe: {self.timeframe_minutes} minute(s)")
        print(f"{'='*80}\n")
        
        # S'abonner aux barres pour chaque ticker
        self.stream.subscribe_bars(self.on_bar, *self.tickers)
        
        # Lancer le stream (bloquant)
        try:
            await self.stream._run_forever()
        except KeyboardInterrupt:
            print("\n⚠️ Arrêt demandé par l'utilisateur")
            self.stop()
    
    
    def stop(self):
        """
        Arrête le monitoring et affiche les stats
        """
        if self.start_time:
            duration = datetime.now() - self.start_time
            print(f"\n{'='*80}")
            print(f"📊 STATISTIQUES DE SESSION")
            print(f"Durée: {duration}")
            print(f"Barres reçues: {self.total_bars_received}")
            print(f"Signaux générés: {self.total_signals_generated}")
            print(f"\nPar ticker:")
            for ticker, detector in self.detectors.items():
                stats = detector.get_stats()
                print(f"  {ticker}: {stats['signals_emitted']}")
            print(f"{'='*80}\n")
    
    
    def get_current_state(self) -> Dict:
        """
        Retourne l'état actuel de tous les tickers
        
        Returns:
            dict: État par ticker
        """
        state = {}
        for ticker, detector in self.detectors.items():
            state[ticker] = {
                "last_signal": detector.last_signal,
                "last_signal_time": detector.last_signal_time,
                "stats": detector.get_stats()
            }
        return state


# ========== EXEMPLE D'UTILISATION ==========

async def main():
    """
    Exemple d'utilisation du LiveAnalyzer
    """
    
    # Liste des tickers à surveiller
    WATCHLIST = ["NVDA", "AAPL", "MSFT", "GOOGL", "TSLA"]
    
    # Créer l'analyseur (barres de 1 minute)
    analyzer = LiveAnalyzer(WATCHLIST, timeframe_minutes=1)
    
    # Ajouter un callback personnalisé
    def on_signal(signal):
        """Callback appelé à chaque signal"""
        # Ici tu peux ajouter :
        # - Envoi d'email/Telegram
        # - Sauvegarde en BDD
        # - Exécution automatique d'ordre
        print(f"🔔 Callback: Signal {signal['signal']} pour {signal['ticker']}")
    
    analyzer.add_signal_callback(on_signal)
    
    # Démarrer le monitoring
    await analyzer.start()


if __name__ == "__main__":
    # Lancer le monitoring
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Au revoir !")
