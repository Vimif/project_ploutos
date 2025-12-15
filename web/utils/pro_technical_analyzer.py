#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 PRO TECHNICAL ANALYZER
Module d'analyse technique professionnelle avec les 5 indicateurs clés :
1. Tendance : SMA 200, SMA 50, EMA 20
2. Momentum : RSI avec détection de divergences
3. Confirmation : MACD avec analyse de l'histogramme
4. Volatilité : Bollinger Bands avec détection du squeeze
5. Volume : OBV avec analyse de l'accumulation
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum


class TrendDirection(str, Enum):
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"


class Signal(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class TrendAnalysis:
    """Résultat analyse de tendance"""
    direction: TrendDirection
    strength: float  # 0-100
    price_vs_sma200: str  # "above" ou "below"
    golden_cross: bool
    death_cross: bool
    support_level: Optional[float]
    resistance_level: Optional[float]
    explanation: str


@dataclass
class MomentumAnalysis:
    """Résultat analyse momentum"""
    rsi_value: float
    zone: str  # "oversold", "neutral", "overbought"
    divergence_detected: bool
    divergence_type: Optional[str]  # "bullish" ou "bearish"
    signal: Signal
    explanation: str


@dataclass
class MACDAnalysis:
    """Résultat analyse MACD"""
    macd_value: float
    signal_value: float
    histogram_value: float
    crossover: Optional[str]  # "bullish" ou "bearish"
    histogram_direction: str  # "increasing", "decreasing", "flat"
    signal: Signal
    explanation: str


@dataclass
class VolatilityAnalysis:
    """Résultat analyse volatilité"""
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width: float
    squeeze_detected: bool
    price_position: str  # "upper", "middle", "lower", "outside"
    explanation: str


@dataclass
class VolumeAnalysis:
    """Résultat analyse volume"""
    obv_trend: str  # "rising", "falling", "flat"
    volume_confirmation: bool
    smart_money_accumulation: bool
    explanation: str


@dataclass
class ProTechnicalReport:
    """Rapport complet d'analyse technique pro"""
    ticker: str
    current_price: float
    timestamp: str
    trend: TrendAnalysis
    momentum: MomentumAnalysis
    macd: MACDAnalysis
    volatility: VolatilityAnalysis
    volume: VolumeAnalysis
    overall_signal: Signal
    confidence: float  # 0-100
    trading_plan: str
    risk_level: str  # "LOW", "MEDIUM", "HIGH"


class ProTechnicalAnalyzer:
    """
    Analyseur technique professionnel avec règles d'interprétation avancées
    """
    
    def __init__(self):
        self.lookback_divergence = 14
        self.squeeze_threshold = 0.02  # 2% de largeur de BB
    
    def analyze(self, df: pd.DataFrame, ticker: str = "") -> ProTechnicalReport:
        """
        Analyse technique complète d'un DataFrame OHLCV
        
        Args:
            df: DataFrame avec colonnes ['Open', 'High', 'Low', 'Close', 'Volume']
            ticker: Symbole de l'action
        
        Returns:
            ProTechnicalReport complet
        """
        # Normaliser les colonnes
        df = self._prepare_dataframe(df)
        
        # Calculer tous les indicateurs
        df = self._calculate_all_indicators(df)
        
        # Analyser chaque catégorie
        trend = self._analyze_trend(df)
        momentum = self._analyze_momentum(df)
        macd = self._analyze_macd(df)
        volatility = self._analyze_volatility(df)
        volume = self._analyze_volume(df)
        
        # Synthèse globale
        overall_signal, confidence = self._compute_overall_signal(
            trend, momentum, macd, volatility, volume
        )
        
        trading_plan = self._generate_trading_plan(
            df, trend, momentum, macd, volatility, volume, overall_signal
        )
        
        risk_level = self._assess_risk(
            momentum, volatility, volume
        )
        
        return ProTechnicalReport(
            ticker=ticker,
            current_price=float(df['Close'].iloc[-1]),
            timestamp=df.index[-1].isoformat() if hasattr(df.index[-1], 'isoformat') else str(df.index[-1]),
            trend=trend,
            momentum=momentum,
            macd=macd,
            volatility=volatility,
            volume=volume,
            overall_signal=overall_signal,
            confidence=confidence,
            trading_plan=trading_plan,
            risk_level=risk_level
        )
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise les colonnes du DataFrame"""
        df = df.copy()
        
        # Gérer les MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Normaliser les noms de colonnes
        column_mapping = {
            col: col.capitalize() for col in df.columns
        }
        df.rename(columns=column_mapping, inplace=True)
        
        return df
    
    def _calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule tous les indicateurs techniques"""
        df = df.copy()
        
        # 1. TENDANCE
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # 2. RSI
        df['RSI'] = self._calculate_rsi(df['Close'], period=14)
        
        # 3. MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # 4. BOLLINGER BANDS
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (2 * bb_std)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        
        # 5. OBV
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcul RSI avec méthode de Wilder"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _analyze_trend(self, df: pd.DataFrame) -> TrendAnalysis:
        """Analyse de la tendance"""
        current_price = float(df['Close'].iloc[-1])
        sma_200 = float(df['SMA_200'].iloc[-1]) if not pd.isna(df['SMA_200'].iloc[-1]) else None
        sma_50 = float(df['SMA_50'].iloc[-1]) if not pd.isna(df['SMA_50'].iloc[-1]) else None
        ema_20 = float(df['EMA_20'].iloc[-1]) if not pd.isna(df['EMA_20'].iloc[-1]) else None
        
        # Détection Golden/Death Cross
        golden_cross = False
        death_cross = False
        
        if sma_50 and sma_200 and len(df) >= 2:
            prev_sma_50 = float(df['SMA_50'].iloc[-2]) if not pd.isna(df['SMA_50'].iloc[-2]) else None
            prev_sma_200 = float(df['SMA_200'].iloc[-2]) if not pd.isna(df['SMA_200'].iloc[-2]) else None
            
            if prev_sma_50 and prev_sma_200:
                if sma_50 > sma_200 and prev_sma_50 <= prev_sma_200:
                    golden_cross = True
                elif sma_50 < sma_200 and prev_sma_50 >= prev_sma_200:
                    death_cross = True
        
        # Déterminer direction et force
        if sma_200:
            price_vs_sma200 = "above" if current_price > sma_200 else "below"
            
            if current_price > sma_200 and sma_50 and sma_50 > sma_200:
                direction = TrendDirection.STRONG_BULLISH
                strength = 85.0
            elif current_price > sma_200:
                direction = TrendDirection.BULLISH
                strength = 65.0
            elif current_price < sma_200 and sma_50 and sma_50 < sma_200:
                direction = TrendDirection.STRONG_BEARISH
                strength = 85.0
            elif current_price < sma_200:
                direction = TrendDirection.BEARISH
                strength = 65.0
            else:
                direction = TrendDirection.NEUTRAL
                strength = 50.0
        else:
            price_vs_sma200 = "unknown"
            direction = TrendDirection.NEUTRAL
            strength = 50.0
        
        # Support/Resistance
        support = ema_20 if ema_20 and ema_20 < current_price else None
        resistance = sma_200 if sma_200 and sma_200 > current_price else None
        
        # Explication
        explanation = f"Prix {'au-dessus' if price_vs_sma200 == 'above' else 'en-dessous'} de la SMA 200. "
        if golden_cross:
            explanation += "Golden Cross détecté ! Signal haussier fort. "
        if death_cross:
            explanation += "Death Cross détecté ! Signal baissier fort. "
        
        return TrendAnalysis(
            direction=direction,
            strength=strength,
            price_vs_sma200=price_vs_sma200,
            golden_cross=golden_cross,
            death_cross=death_cross,
            support_level=support,
            resistance_level=resistance,
            explanation=explanation
        )
    
    def _analyze_momentum(self, df: pd.DataFrame) -> MomentumAnalysis:
        """Analyse du momentum avec détection de divergences"""
        rsi = float(df['RSI'].iloc[-1])
        
        # Zone RSI
        if rsi > 70:
            zone = "overbought"
            signal = Signal.SELL if rsi > 80 else Signal.HOLD
        elif rsi < 30:
            zone = "oversold"
            signal = Signal.BUY if rsi < 20 else Signal.HOLD
        else:
            zone = "neutral"
            signal = Signal.HOLD
        
        # Détection divergences
        divergence_detected, divergence_type = self._detect_divergence(df)
        
        if divergence_detected:
            if divergence_type == "bearish":
                signal = Signal.SELL
            elif divergence_type == "bullish":
                signal = Signal.BUY
        
        # Explication
        explanation = f"RSI à {rsi:.1f} ({zone}). "
        if divergence_detected:
            explanation += f"Divergence {divergence_type} détectée ! Signal fort. "
        elif rsi > 80:
            explanation += "Surachat extrême, correction imminente. "
        elif rsi < 20:
            explanation += "Survente extrême, rebond probable. "
        
        return MomentumAnalysis(
            rsi_value=rsi,
            zone=zone,
            divergence_detected=divergence_detected,
            divergence_type=divergence_type,
            signal=signal,
            explanation=explanation
        )
    
    def _detect_divergence(self, df: pd.DataFrame) -> tuple[bool, Optional[str]]:
        """Détecte les divergences RSI/Prix"""
        if len(df) < self.lookback_divergence:
            return False, None
        
        window = df.tail(self.lookback_divergence)
        
        # Trouver les pics de prix et RSI
        price_peaks = window['Close'].rolling(3, center=True).max() == window['Close']
        rsi_peaks = window['RSI'].rolling(3, center=True).max() == window['RSI']
        
        price_peak_values = window.loc[price_peaks, 'Close']
        rsi_peak_values = window.loc[rsi_peaks, 'RSI']
        
        if len(price_peak_values) >= 2 and len(rsi_peak_values) >= 2:
            # Divergence baisssière : prix monte, RSI descend
            if price_peak_values.iloc[-1] > price_peak_values.iloc[-2] and \
               rsi_peak_values.iloc[-1] < rsi_peak_values.iloc[-2]:
                return True, "bearish"
            
            # Divergence haussière : prix descend, RSI monte
            if price_peak_values.iloc[-1] < price_peak_values.iloc[-2] and \
               rsi_peak_values.iloc[-1] > rsi_peak_values.iloc[-2]:
                return True, "bullish"
        
        return False, None
    
    def _analyze_macd(self, df: pd.DataFrame) -> MACDAnalysis:
        """Analyse MACD"""
        macd = float(df['MACD'].iloc[-1])
        signal_line = float(df['MACD_Signal'].iloc[-1])
        histogram = float(df['MACD_Hist'].iloc[-1])
        
        # Détection croisement
        crossover = None
        if len(df) >= 2:
            prev_macd = float(df['MACD'].iloc[-2])
            prev_signal = float(df['MACD_Signal'].iloc[-2])
            
            if macd > signal_line and prev_macd <= prev_signal:
                crossover = "bullish"
            elif macd < signal_line and prev_macd >= prev_signal:
                crossover = "bearish"
        
        # Direction histogramme
        if len(df) >= 3:
            hist_trend = df['MACD_Hist'].tail(3)
            if hist_trend.is_monotonic_increasing:
                histogram_direction = "increasing"
            elif hist_trend.is_monotonic_decreasing:
                histogram_direction = "decreasing"
            else:
                histogram_direction = "flat"
        else:
            histogram_direction = "flat"
        
        # Signal
        if crossover == "bullish":
            signal = Signal.BUY
        elif crossover == "bearish":
            signal = Signal.SELL
        elif macd > 0 and histogram_direction == "increasing":
            signal = Signal.HOLD  # Tendance haussiere solide
        elif macd < 0 and histogram_direction == "decreasing":
            signal = Signal.HOLD  # Tendance baissiere solide
        else:
            signal = Signal.HOLD
        
        explanation = f"MACD {'au-dessus' if macd > signal_line else 'en-dessous'} de la ligne de signal. "
        if crossover:
            explanation += f"Croisement {crossover} détecté ! "
        if histogram_direction == "decreasing" and macd > 0:
            explanation += "Histogramme en baisse, perte de momentum. "
        
        return MACDAnalysis(
            macd_value=macd,
            signal_value=signal_line,
            histogram_value=histogram,
            crossover=crossover,
            histogram_direction=histogram_direction,
            signal=signal,
            explanation=explanation
        )
    
    def _analyze_volatility(self, df: pd.DataFrame) -> VolatilityAnalysis:
        """Analyse volatilité (Bollinger Bands)"""
        current_price = float(df['Close'].iloc[-1])
        bb_upper = float(df['BB_Upper'].iloc[-1])
        bb_middle = float(df['BB_Middle'].iloc[-1])
        bb_lower = float(df['BB_Lower'].iloc[-1])
        bb_width = bb_upper - bb_lower
        
        # Détection squeeze
        avg_bb_width = df['BB_Width'].tail(20).mean()
        squeeze_detected = bb_width < (avg_bb_width * (1 - self.squeeze_threshold))
        
        # Position du prix
        if current_price > bb_upper:
            price_position = "outside_upper"
        elif current_price < bb_lower:
            price_position = "outside_lower"
        elif current_price > bb_middle:
            price_position = "upper"
        else:
            price_position = "lower"
        
        explanation = f"Prix dans la {'moitié haute' if 'upper' in price_position else 'moitié basse'} des bandes. "
        if squeeze_detected:
            explanation += "Squeeze détecté ! Mouvement explosif imminent. "
        if price_position == "outside_upper":
            explanation += "Prix au-dessus de la bande supérieure, surextension. "
        
        return VolatilityAnalysis(
            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower,
            bb_width=bb_width,
            squeeze_detected=squeeze_detected,
            price_position=price_position,
            explanation=explanation
        )
    
    def _analyze_volume(self, df: pd.DataFrame) -> VolumeAnalysis:
        """Analyse volume (OBV)"""
        # Tendance OBV
        obv_window = df['OBV'].tail(10)
        if obv_window.is_monotonic_increasing:
            obv_trend = "rising"
        elif obv_window.is_monotonic_decreasing:
            obv_trend = "falling"
        else:
            obv_trend = "flat"
        
        # Confirmation volume
        price_change = df['Close'].pct_change().iloc[-1]
        volume_change = df['Volume'].pct_change().iloc[-1]
        volume_confirmation = (price_change > 0 and volume_change > 0) or \
                            (price_change < 0 and volume_change > 0)
        
        # Smart Money (OBV monte alors que prix stagne)
        price_flat = abs(df['Close'].pct_change().tail(5).mean()) < 0.01
        smart_money_accumulation = obv_trend == "rising" and price_flat
        
        explanation = f"OBV en tendance {obv_trend}. "
        if smart_money_accumulation:
            explanation += "Accumulation par les gros (Smart Money) détectée ! "
        if not volume_confirmation:
            explanation += "Mouvement non confirmé par le volume. "
        
        return VolumeAnalysis(
            obv_trend=obv_trend,
            volume_confirmation=volume_confirmation,
            smart_money_accumulation=smart_money_accumulation,
            explanation=explanation
        )
    
    def _compute_overall_signal(self, trend, momentum, macd, volatility, volume) -> tuple[Signal, float]:
        """Calcule le signal global et la confiance"""
        signals = []
        weights = []
        
        # Pondération par importance
        if trend.direction in [TrendDirection.STRONG_BULLISH, TrendDirection.STRONG_BEARISH]:
            weights.append(3.0)
        else:
            weights.append(2.0)
        
        if trend.direction in [TrendDirection.STRONG_BULLISH, TrendDirection.BULLISH]:
            signals.append(1)
        elif trend.direction in [TrendDirection.STRONG_BEARISH, TrendDirection.BEARISH]:
            signals.append(-1)
        else:
            signals.append(0)
        
        # Momentum (poids fort)
        weights.append(3.0)
        if momentum.signal == Signal.BUY:
            signals.append(1)
        elif momentum.signal == Signal.SELL:
            signals.append(-1)
        else:
            signals.append(0)
        
        # MACD
        weights.append(2.5)
        if macd.signal == Signal.BUY:
            signals.append(1)
        elif macd.signal == Signal.SELL:
            signals.append(-1)
        else:
            signals.append(0)
        
        # Volume (confirmation)
        weights.append(1.5)
        if volume.smart_money_accumulation:
            signals.append(1)
        elif volume.obv_trend == "falling":
            signals.append(-1)
        else:
            signals.append(0)
        
        # Calcul pondéré
        weighted_score = sum(s * w for s, w in zip(signals, weights)) / sum(weights)
        
        # Conversion en signal
        if weighted_score > 0.5:
            overall = Signal.STRONG_BUY
            confidence = min(90, 50 + weighted_score * 40)
        elif weighted_score > 0.2:
            overall = Signal.BUY
            confidence = min(80, 50 + weighted_score * 30)
        elif weighted_score < -0.5:
            overall = Signal.STRONG_SELL
            confidence = min(90, 50 + abs(weighted_score) * 40)
        elif weighted_score < -0.2:
            overall = Signal.SELL
            confidence = min(80, 50 + abs(weighted_score) * 30)
        else:
            overall = Signal.HOLD
            confidence = 50 + abs(weighted_score) * 20
        
        return overall, confidence
    
    def _generate_trading_plan(self, df, trend, momentum, macd, volatility, volume, overall_signal) -> str:
        """Génère un plan de trading concret"""
        current_price = float(df['Close'].iloc[-1])
        
        if overall_signal == Signal.STRONG_BUY:
            plan = f"🟢 **ACHAT PRIORITAIRE** \n"
            plan += f"Point d'entrée : {current_price:.2f}$ (maintenant)\n"
            if trend.support_level:
                plan += f"Stop Loss : {trend.support_level:.2f}$ (EMA 20)\n"
            if trend.resistance_level:
                plan += f"Take Profit : {trend.resistance_level:.2f}$ (SMA 200)\n"
        
        elif overall_signal == Signal.BUY:
            plan = f"🟡 **ACHAT sur PULLBACK** \n"
            if trend.support_level:
                plan += f"Attendre un retour vers {trend.support_level:.2f}$ pour entrer\n"
            plan += f"Valider avec une bougie verte + volume\n"
        
        elif overall_signal in [Signal.SELL, Signal.STRONG_SELL]:
            plan = f"🔴 **VENTE ou ATTENTE** \n"
            plan += f"Ne pas acheter. Prendre profits si en position.\n"
            if momentum.rsi_value > 70:
                plan += f"RSI en surachat ({momentum.rsi_value:.1f}), attendre correction.\n"
        
        else:  # HOLD
            plan = f"🟠 **ATTENTISME** \n"
            plan += f"Pas de signal clair. Attendre une confirmation.\n"
            if volatility.squeeze_detected:
                plan += f"⚠️ Squeeze détecté : surveiller la cassure !\n"
        
        return plan
    
    def _assess_risk(self, momentum, volatility, volume) -> str:
        """'Évalue le niveau de risque"""
        risk_score = 0
        
        # RSI extrême
        if momentum.rsi_value > 80 or momentum.rsi_value < 20:
            risk_score += 2
        
        # Squeeze = volatilité comprimée
        if volatility.squeeze_detected:
            risk_score += 1
        
        # Pas de confirmation volume
        if not volume.volume_confirmation:
            risk_score += 1
        
        if risk_score >= 3:
            return "HIGH"
        elif risk_score >= 1:
            return "MEDIUM"
        else:
            return "LOW"
