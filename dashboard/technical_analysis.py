"""Module d'analyse technique en temps réel pour le Trading Signals Dashboard

Calcule des indicateurs techniques pertinents pour déterminer les tendances
et les signaux d'achat/vente sur les actions en temps réel.

Auteur: Ploutos Team
Date: 2025-12-15
Version: 1.0.0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from core.utils import setup_logging

logger = setup_logging(__name__, "technical_analysis.log")


@dataclass
class TradingSignal:
    """Signal de trading structuré"""

    signal: str  # 'BUY', 'SELL', 'HOLD'
    strength: int  # 0-100
    trend: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    confidence: float  # 0.0-1.0
    reasons: list[str]
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None

    def to_dict(self) -> dict:
        """Convertir en dictionnaire pour JSON"""
        return asdict(self)


class TechnicalAnalyzer:
    """Analyseur technique en temps réel pour une action"""

    def __init__(self, symbol: str, period: str = "3mo", interval: str = "1h"):
        """
        Initialiser l'analyseur technique

        Args:
            symbol: Ticker de l'action (ex: 'NVDA')
            period: Période historique ('1mo', '3mo', '6mo', '1y', '2y')
            interval: Intervalle des données ('1m', '5m', '15m', '1h', '1d')
        """
        self.symbol = symbol.upper()
        self.period = period
        self.interval = interval
        self.df = None
        self._fetch_data()

    def _fetch_data(self) -> None:
        """Récupérer les données depuis Yahoo Finance"""
        try:
            logger.info(f"📥 Téléchargement données {self.symbol} ({self.period}, {self.interval})")
            ticker = yf.Ticker(self.symbol)
            self.df = ticker.history(period=self.period, interval=self.interval)

            if self.df.empty:
                logger.error(f"❌ Aucune donnée pour {self.symbol}")
                raise ValueError(f"Pas de données pour {self.symbol}")

            logger.info(f"✅ {len(self.df)} barres téléchargées pour {self.symbol}")

        except Exception as e:
            logger.error(f"❌ Erreur téléchargement {self.symbol}: {e}")
            raise

    def refresh_data(self) -> None:
        """Rafraîchir les données (pour mise à jour temps réel)"""
        self._fetch_data()

    # ========== INDICATEURS DE TENDANCE ==========

    def calculate_sma(self, period: int = 20) -> pd.Series:
        """Simple Moving Average (SMA)"""
        return self.df["Close"].rolling(window=period).mean()

    def calculate_ema(self, period: int = 20) -> pd.Series:
        """Exponential Moving Average (EMA) - Plus réactif que SMA"""
        return self.df["Close"].ewm(span=period, adjust=False).mean()

    def calculate_macd(self) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence)

        Returns:
            Tuple (macd_line, signal_line, histogram)
        """
        ema_12 = self.df["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = self.df["Close"].ewm(span=26, adjust=False).mean()

        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    # ========== INDICATEURS DE MOMENTUM ==========

    def calculate_rsi(self, period: int = 14) -> pd.Series:
        """
        RSI (Relative Strength Index)

        RSI > 70: Suracheté (potentiel SELL)
        RSI < 30: Survendu (potentiel BUY)
        """
        delta = self.df["Close"].diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_stochastic(
        self, k_period: int = 14, d_period: int = 3
    ) -> tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator

        %K > 80: Suracheté
        %K < 20: Survendu
        """
        low_min = self.df["Low"].rolling(window=k_period).min()
        high_max = self.df["High"].rolling(window=k_period).max()

        k_percent = 100 * ((self.df["Close"] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()

        return k_percent, d_percent

    # ========== INDICATEURS DE VOLATILITÉ ==========

    def calculate_bollinger_bands(
        self, period: int = 20, std_dev: float = 2.0
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands

        Prix > Upper Band: Potentiel suracheté
        Prix < Lower Band: Potentiel survendu
        """
        middle_band = self.calculate_sma(period)
        std = self.df["Close"].rolling(window=period).std()

        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)

        return upper_band, middle_band, lower_band

    def calculate_atr(self, period: int = 14) -> pd.Series:
        """ATR (Average True Range) - Mesure la volatilité"""
        high_low = self.df["High"] - self.df["Low"]
        high_close = np.abs(self.df["High"] - self.df["Close"].shift())
        low_close = np.abs(self.df["Low"] - self.df["Close"].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return atr

    # ========== INDICATEURS DE VOLUME ==========

    def calculate_obv(self) -> pd.Series:
        """OBV (On-Balance Volume) - Indicateur de flux de volume cumulatif"""
        obv = np.where(
            self.df["Close"] > self.df["Close"].shift(),
            self.df["Volume"],
            np.where(self.df["Close"] < self.df["Close"].shift(), -self.df["Volume"], 0),
        )
        return pd.Series(obv, index=self.df.index).cumsum()

    def calculate_vwap(self) -> pd.Series:
        """
        VWAP (Volume Weighted Average Price)

        Prix > VWAP: Tendance haussière
        Prix < VWAP: Tendance baissière
        """
        typical_price = (self.df["High"] + self.df["Low"] + self.df["Close"]) / 3
        vwap = (typical_price * self.df["Volume"]).cumsum() / self.df["Volume"].cumsum()

        return vwap

    # ========== ANALYSE GLOBALE ==========

    def detect_trend(self) -> str:
        """
        Détecter la tendance actuelle

        Returns:
            'BULLISH', 'BEARISH', ou 'NEUTRAL'
        """
        if self.df is None or len(self.df) < 50:
            return "NEUTRAL"

        # Moyennes mobiles
        sma_20 = self.calculate_sma(20).iloc[-1]
        sma_50 = self.calculate_sma(50).iloc[-1]
        current_price = self.df["Close"].iloc[-1]

        # MACD
        macd_line, signal_line, _ = self.calculate_macd()
        macd_current = macd_line.iloc[-1]
        signal_current = signal_line.iloc[-1]

        bullish_signals = 0
        bearish_signals = 0

        # Signal 1: Prix vs SMA
        if current_price > sma_20 > sma_50:
            bullish_signals += 2
        elif current_price < sma_20 < sma_50:
            bearish_signals += 2

        # Signal 2: MACD
        if macd_current > signal_current and macd_current > 0:
            bullish_signals += 1
        elif macd_current < signal_current and macd_current < 0:
            bearish_signals += 1

        # Signal 3: Tendance SMA 20
        sma_20_slope = (sma_20 - self.calculate_sma(20).iloc[-5]) / 5
        if sma_20_slope > 0:
            bullish_signals += 1
        elif sma_20_slope < 0:
            bearish_signals += 1

        if bullish_signals > bearish_signals:
            return "BULLISH"
        elif bearish_signals > bullish_signals:
            return "BEARISH"
        else:
            return "NEUTRAL"

    def generate_signal(self) -> TradingSignal:
        """
        Générer un signal de trading complet basé sur tous les indicateurs

        Returns:
            TradingSignal avec recommandation BUY/SELL/HOLD
        """
        if self.df is None or len(self.df) < 50:
            return TradingSignal(
                signal="HOLD",
                strength=0,
                trend="NEUTRAL",
                confidence=0.0,
                reasons=["Pas assez de données historiques"],
            )

        current_price = self.df["Close"].iloc[-1]
        reasons = []
        buy_score = 0
        sell_score = 0

        # 1. RSI
        rsi = self.calculate_rsi().iloc[-1]
        if rsi < 30:
            buy_score += 2
            reasons.append(f"RSI survendu ({rsi:.1f})")
        elif rsi > 70:
            sell_score += 2
            reasons.append(f"RSI suracheté ({rsi:.1f})")
        elif 40 <= rsi <= 60:
            reasons.append(f"RSI neutre ({rsi:.1f})")

        # 2. MACD
        macd_line, signal_line, histogram = self.calculate_macd()
        macd_current = macd_line.iloc[-1]
        signal_current = signal_line.iloc[-1]

        if macd_current > signal_current and histogram.iloc[-1] > histogram.iloc[-2]:
            buy_score += 2
            reasons.append("MACD croisement haussier")
        elif macd_current < signal_current and histogram.iloc[-1] < histogram.iloc[-2]:
            sell_score += 2
            reasons.append("MACD croisement baissier")

        # 3. Bollinger Bands
        upper, middle, lower = self.calculate_bollinger_bands()
        if current_price < lower.iloc[-1]:
            buy_score += 1
            reasons.append("Prix sous bande de Bollinger inférieure")
        elif current_price > upper.iloc[-1]:
            sell_score += 1
            reasons.append("Prix au-dessus bande de Bollinger supérieure")

        # 4. Moyennes mobiles
        sma_20 = self.calculate_sma(20).iloc[-1]
        sma_50 = self.calculate_sma(50).iloc[-1]

        if current_price > sma_20 > sma_50:
            buy_score += 1
            reasons.append("Prix au-dessus SMA 20 et 50 (tendance haussière)")
        elif current_price < sma_20 < sma_50:
            sell_score += 1
            reasons.append("Prix sous SMA 20 et 50 (tendance baissière)")

        # 5. Stochastic
        k_percent, d_percent = self.calculate_stochastic()
        k_current = k_percent.iloc[-1]

        if k_current < 20:
            buy_score += 1
            reasons.append(f"Stochastique survendu ({k_current:.1f})")
        elif k_current > 80:
            sell_score += 1
            reasons.append(f"Stochastique suracheté ({k_current:.1f})")

        # 6. Volume (OBV)
        obv = self.calculate_obv()
        obv_trend = (obv.iloc[-1] - obv.iloc[-5]) / 5

        if obv_trend > 0 and current_price > self.df["Close"].iloc[-5]:
            buy_score += 1
            reasons.append("Volume confirmant la hausse")
        elif obv_trend < 0 and current_price < self.df["Close"].iloc[-5]:
            sell_score += 1
            reasons.append("Volume confirmant la baisse")

        # Déterminer le signal final
        if buy_score > sell_score and buy_score >= 4:
            signal = "BUY"
            strength = min(100, int(buy_score / 7 * 100))
            confidence = min(1.0, buy_score / 7)
        elif sell_score > buy_score and sell_score >= 4:
            signal = "SELL"
            strength = min(100, int(sell_score / 7 * 100))
            confidence = min(1.0, sell_score / 7)
        else:
            signal = "HOLD"
            strength = 50
            confidence = 0.5
            reasons.append("Signaux mixtes, attendre confirmation")

        # Calculer stop-loss et take-profit
        atr = self.calculate_atr().iloc[-1]

        stop_loss = None
        take_profit = None

        if signal == "BUY":
            stop_loss = current_price - (2 * atr)
            take_profit = current_price + (3 * atr)
        elif signal == "SELL":
            stop_loss = current_price + (2 * atr)
            take_profit = current_price - (3 * atr)

        trend = self.detect_trend()

        return TradingSignal(
            signal=signal,
            strength=strength,
            trend=trend,
            confidence=confidence,
            reasons=reasons,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    def get_all_indicators(self) -> dict:
        """
        Obtenir tous les indicateurs techniques actuels

        Returns:
            Dict avec toutes les valeurs des indicateurs
        """
        if self.df is None or len(self.df) < 50:
            return {"error": "Pas assez de données"}

        current_price = self.df["Close"].iloc[-1]

        # Calculer tous les indicateurs
        sma_20 = self.calculate_sma(20).iloc[-1]
        sma_50 = self.calculate_sma(50).iloc[-1]
        ema_20 = self.calculate_ema(20).iloc[-1]

        macd_line, signal_line, histogram = self.calculate_macd()

        rsi = self.calculate_rsi().iloc[-1]
        k_percent, d_percent = self.calculate_stochastic()

        upper, middle, lower = self.calculate_bollinger_bands()
        atr = self.calculate_atr().iloc[-1]

        obv = self.calculate_obv().iloc[-1]
        vwap = self.calculate_vwap().iloc[-1]

        return {
            "price": {
                "current": float(current_price),
                "change_24h": (
                    float(
                        (current_price - self.df["Close"].iloc[-24])
                        / self.df["Close"].iloc[-24]
                        * 100
                    )
                    if len(self.df) >= 24
                    else 0
                ),
                "high_24h": (
                    float(self.df["High"].iloc[-24:].max())
                    if len(self.df) >= 24
                    else float(current_price)
                ),
                "low_24h": (
                    float(self.df["Low"].iloc[-24:].min())
                    if len(self.df) >= 24
                    else float(current_price)
                ),
            },
            "moving_averages": {
                "sma_20": float(sma_20),
                "sma_50": float(sma_50),
                "ema_20": float(ema_20),
            },
            "macd": {
                "macd_line": float(macd_line.iloc[-1]),
                "signal_line": float(signal_line.iloc[-1]),
                "histogram": float(histogram.iloc[-1]),
            },
            "momentum": {
                "rsi": float(rsi),
                "stochastic_k": float(k_percent.iloc[-1]),
                "stochastic_d": float(d_percent.iloc[-1]),
            },
            "volatility": {
                "bb_upper": float(upper.iloc[-1]),
                "bb_middle": float(middle.iloc[-1]),
                "bb_lower": float(lower.iloc[-1]),
                "atr": float(atr),
            },
            "volume": {
                "obv": float(obv),
                "vwap": float(vwap),
                "volume_24h": (
                    float(self.df["Volume"].iloc[-24:].sum()) if len(self.df) >= 24 else 0
                ),
            },
        }
