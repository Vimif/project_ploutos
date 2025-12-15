"""
🎯 SIGNAL DETECTOR - Détection signaux BUY/SELL en temps réel

Analyse les données Alpaca WebSocket et génère des signaux de trading
basés sur plusieurs stratégies (RSI, EMA, MACD, Volume, Bollinger).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import talib as ta


class SignalDetector:
    """Détecte les signaux de trading en temps réel"""
    
    def __init__(self, ticker: str, window_size: int = 50):
        """
        Args:
            ticker: Symbole de l'action
            window_size: Nombre de barres à conserver en mémoire
        """
        self.ticker = ticker
        self.window_size = window_size
        
        # Buffer circulaire pour stocker les dernières barres
        self.prices = deque(maxlen=window_size)
        self.volumes = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
        # Dernier signal émis
        self.last_signal = None
        self.last_signal_time = None
        
        # Statistiques
        self.signals_count = {"BUY": 0, "SELL": 0, "HOLD": 0}
        
    
    def add_bar(self, timestamp, open_price, high, low, close, volume):
        """
        Ajoute une nouvelle barre et calcule les signaux
        
        Args:
            timestamp: Horodatage de la barre
            open_price, high, low, close: Prix OHLC
            volume: Volume échangé
            
        Returns:
            dict: Signal généré avec détails
        """
        self.timestamps.append(timestamp)
        self.prices.append(close)
        self.volumes.append(volume)
        
        # Attendre d'avoir assez de données
        if len(self.prices) < 26:  # Minimum pour MACD
            return {"signal": "WAIT", "reason": "Données insuffisantes", "confidence": 0}
        
        # Calculer les signaux
        signal = self._generate_signal()
        
        # Mettre à jour statistiques
        self.signals_count[signal["signal"]] += 1
        self.last_signal = signal["signal"]
        self.last_signal_time = timestamp
        
        return signal
    
    
    def _generate_signal(self) -> dict:
        """
        Génère un signal de trading basé sur 5 stratégies
        
        Returns:
            dict: {
                "signal": "BUY" | "SELL" | "HOLD",
                "confidence": 0-100,
                "reasons": [liste des raisons],
                "indicators": {valeurs des indicateurs}
            }
        """
        df = pd.DataFrame({
            'close': list(self.prices),
            'volume': list(self.volumes)
        })
        
        # === INDICATEURS TECHNIQUES ===
        
        # 1. RSI (14)
        rsi = ta.RSI(df['close'].values, timeperiod=14)[-1]
        
        # 2. EMA (9 et 21)
        ema_fast = ta.EMA(df['close'].values, timeperiod=9)[-1]
        ema_slow = ta.EMA(df['close'].values, timeperiod=21)[-1]
        
        # 3. MACD
        macd, signal_line, hist = ta.MACD(df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
        macd_current = macd[-1]
        signal_current = signal_line[-1]
        hist_current = hist[-1]
        
        # 4. Bollinger Bands
        upper, middle, lower = ta.BBANDS(df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2)
        bb_upper = upper[-1]
        bb_lower = lower[-1]
        current_price = df['close'].iloc[-1]
        
        # 5. Volume
        avg_volume = df['volume'].tail(20).mean()
        current_volume = df['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        
        # === DÉTECTION DES SIGNAUX ===
        
        buy_signals = []
        sell_signals = []
        confidence_score = 0
        
        # Signal 1: RSI Oversold/Overbought
        if rsi < 30:
            buy_signals.append(f"RSI survente ({rsi:.1f})")
            confidence_score += 25
        elif rsi > 70:
            sell_signals.append(f"RSI surachat ({rsi:.1f})")
            confidence_score += 25
        
        # Signal 2: EMA Crossover
        if ema_fast > ema_slow:
            buy_signals.append(f"EMA Golden Cross ({ema_fast:.2f} > {ema_slow:.2f})")
            confidence_score += 20
        elif ema_fast < ema_slow:
            sell_signals.append(f"EMA Death Cross ({ema_fast:.2f} < {ema_slow:.2f})")
            confidence_score += 20
        
        # Signal 3: MACD Crossover
        if macd_current > signal_current and hist_current > 0:
            buy_signals.append(f"MACD bullish ({hist_current:.3f})")
            confidence_score += 20
        elif macd_current < signal_current and hist_current < 0:
            sell_signals.append(f"MACD bearish ({hist_current:.3f})")
            confidence_score += 20
        
        # Signal 4: Bollinger Bands
        if current_price < bb_lower:
            buy_signals.append(f"Prix sous BB inférieur ({current_price:.2f} < {bb_lower:.2f})")
            confidence_score += 15
        elif current_price > bb_upper:
            sell_signals.append(f"Prix au-dessus BB supérieur ({current_price:.2f} > {bb_upper:.2f})")
            confidence_score += 15
        
        # Signal 5: Volume Spike
        if volume_ratio > 1.5:
            # Volume élevé renforce le signal existant
            confidence_score += 10
            if len(buy_signals) > len(sell_signals):
                buy_signals.append(f"Volume spike ({volume_ratio:.1f}x)")
            elif len(sell_signals) > len(buy_signals):
                sell_signals.append(f"Volume spike ({volume_ratio:.1f}x)")
        
        
        # === DÉCISION FINALE ===
        
        if len(buy_signals) >= 3:
            final_signal = "BUY"
            reasons = buy_signals
        elif len(sell_signals) >= 3:
            final_signal = "SELL"
            reasons = sell_signals
        else:
            final_signal = "HOLD"
            reasons = ["Signaux mixtes ou insuffisants"]
            confidence_score = max(0, 100 - confidence_score)
        
        
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamps[-1],
            "signal": final_signal,
            "confidence": min(confidence_score, 100),
            "reasons": reasons,
            "current_price": current_price,
            "indicators": {
                "RSI": round(rsi, 2),
                "EMA_Fast": round(ema_fast, 2),
                "EMA_Slow": round(ema_slow, 2),
                "MACD": round(macd_current, 3),
                "MACD_Signal": round(signal_current, 3),
                "MACD_Hist": round(hist_current, 3),
                "BB_Upper": round(bb_upper, 2),
                "BB_Lower": round(bb_lower, 2),
                "Volume_Ratio": round(volume_ratio, 2)
            },
            "buy_signals_count": len(buy_signals),
            "sell_signals_count": len(sell_signals)
        }
    
    
    def get_stats(self) -> dict:
        """Retourne les statistiques du détecteur"""
        return {
            "ticker": self.ticker,
            "bars_analyzed": len(self.prices),
            "signals_emitted": self.signals_count,
            "last_signal": self.last_signal,
            "last_signal_time": self.last_signal_time
        }


if __name__ == "__main__":
    # Test unitaire
    detector = SignalDetector("NVDA", window_size=50)
    
    # Simuler quelques barres
    import random
    base_price = 500.0
    
    for i in range(30):
        price = base_price + random.uniform(-5, 5)
        volume = random.randint(1000000, 5000000)
        
        signal = detector.add_bar(
            timestamp=datetime.now() - timedelta(minutes=30-i),
            open_price=price,
            high=price + random.uniform(0, 2),
            low=price - random.uniform(0, 2),
            close=price,
            volume=volume
        )
        
        if signal["signal"] != "WAIT":
            print(f"Signal: {signal['signal']} | Confidence: {signal['confidence']}% | Reasons: {signal['reasons']}")
    
    print(f"\nStats: {detector.get_stats()}")
