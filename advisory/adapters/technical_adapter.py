"""
Adaptateur technique : wrappe TechnicalAnalyzer + AdvancedFeaturesV2.

Combine les signaux de l'analyse technique classique (RSI, MACD, Bollinger, etc.)
avec les features avancees V2 (entry_signal, support/resistance, patterns).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from typing import Dict, Optional

from advisory.models import SubSignal

logger = logging.getLogger(__name__)


class TechnicalAdapter:
    """Adapte TechnicalAnalyzer + AdvancedFeaturesV2 vers SubSignal."""

    def analyze(
        self, symbol: str, period: str = "3mo", interval: str = "1h"
    ) -> SubSignal:
        """
        Analyse technique complete pour un symbole.

        Combine TradingSignal (generate_signal) et entry_signal (AdvancedFeaturesV2).
        Blend 70% TradingSignal / 30% entry_signal.

        Returns:
            SubSignal avec signal [-1,+1], confidence [0,1], details des indicateurs
        """
        signal = 0.0
        confidence = 0.0
        details: Dict = {}
        reasons = []

        # --- Signal 1 : TechnicalAnalyzer ---
        ta_signal = 0.0
        ta_confidence = 0.0
        try:
            from dashboard.technical_analysis import TechnicalAnalyzer

            analyzer = TechnicalAnalyzer(symbol, period=period, interval=interval)
            trading_signal = analyzer.generate_signal()
            indicators = analyzer.get_all_indicators()

            # Mapper BUY/SELL/HOLD + strength vers [-1, +1]
            if trading_signal.signal == "BUY":
                ta_signal = trading_signal.strength / 100.0
            elif trading_signal.signal == "SELL":
                ta_signal = -trading_signal.strength / 100.0
            else:
                ta_signal = 0.0

            ta_confidence = trading_signal.confidence
            reasons.extend(trading_signal.reasons)
            details["trading_signal"] = trading_signal.to_dict()
            details["indicators"] = indicators
            details["entry_price"] = trading_signal.entry_price
            details["stop_loss"] = trading_signal.stop_loss
            details["take_profit"] = trading_signal.take_profit

        except Exception as e:
            logger.warning(f"TechnicalAnalyzer indisponible pour {symbol}: {e}")
            ta_confidence = 0.0

        # --- Signal 2 : AdvancedFeaturesV2 entry_signal ---
        entry_signal = 0.0
        entry_confidence = 0.0
        try:
            from core.advanced_features_v2 import AdvancedFeaturesV2

            if "indicators" in details and details["indicators"].get("price"):
                # Reutiliser les donnees de TechnicalAnalyzer si disponibles
                import yfinance as yf

                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)
                if not df.empty and len(df) >= 50:
                    features_v2 = AdvancedFeaturesV2()
                    df_features = features_v2.calculate_all_features(df)
                    latest = df_features.iloc[-1]

                    # entry_signal = buy_score_norm - sell_score_norm (deja en [0,1] chacun)
                    buy_score = float(latest.get("buy_score_norm", 0))
                    sell_score = float(latest.get("sell_score_norm", 0))
                    entry_signal = buy_score - sell_score  # [-1, +1]
                    entry_confidence = 0.6  # Confiance moderee pour les features V2

                    details["buy_score_norm"] = buy_score
                    details["sell_score_norm"] = sell_score
                    details["entry_signal"] = entry_signal

        except Exception as e:
            logger.warning(f"AdvancedFeaturesV2 indisponible pour {symbol}: {e}")
            entry_confidence = 0.0

        # --- Blend : 70% TradingSignal, 30% entry_signal ---
        total_weight = 0.0
        blended_signal = 0.0

        if ta_confidence > 0:
            blended_signal += ta_signal * 0.7
            total_weight += 0.7
        if entry_confidence > 0:
            blended_signal += entry_signal * 0.3
            total_weight += 0.3

        if total_weight > 0:
            signal = blended_signal / total_weight
            confidence = (ta_confidence * 0.7 + entry_confidence * 0.3) / total_weight
        else:
            signal = 0.0
            confidence = 0.0
            reasons.append("Aucune donnee technique disponible")

        # Clamp
        signal = max(-1.0, min(1.0, signal))
        confidence = max(0.0, min(1.0, confidence))

        return SubSignal(
            source="technical",
            signal=signal,
            confidence=confidence,
            details=details,
            reasons=reasons,
        )
