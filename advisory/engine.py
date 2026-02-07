"""
Moteur principal advisory : orchestre les sous-analyseurs et produit
des recommandations d'investissement avec explications.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import asdict

import yfinance as yf

from advisory.models import (
    AdvisoryResult,
    SubSignal,
    ForecastPoint,
    Recommendation,
    score_to_recommendation,
)
from advisory.adapters.technical_adapter import TechnicalAdapter
from advisory.adapters.ml_adapter import MLAdapter
from advisory.adapters.sentiment_analyzer import SentimentAnalyzer
from advisory.adapters.statistical_forecaster import StatisticalForecaster
from advisory.adapters.risk_adapter import RiskAdapter
from advisory.llm.explainer import LLMExplainer
from config.advisory_config import AdvisoryConfig

logger = logging.getLogger(__name__)


class AdvisoryEngine:
    """Moteur d'analyse et de recommandation d'investissement."""

    def __init__(self, config: Optional[AdvisoryConfig] = None):
        self.config = config or AdvisoryConfig()

        logger.info("Initialisation AdvisoryEngine...")

        # Sous-analyseurs (tous avec graceful degradation)
        self.technical = TechnicalAdapter()
        self.ml = MLAdapter()
        self.sentiment = SentimentAnalyzer()
        self.forecaster = StatisticalForecaster(horizon=self.config.forecast_horizon)
        self.risk = RiskAdapter()

        # LLM
        self.explainer = LLMExplainer(
            model=self.config.ollama_model,
            base_url=self.config.ollama_url,
        )

        logger.info("AdvisoryEngine pret")

    def analyze(
        self,
        symbol: str,
        period: Optional[str] = None,
        interval: Optional[str] = None,
        portfolio_value: Optional[float] = None,
        positions: Optional[List[Dict]] = None,
    ) -> AdvisoryResult:
        """
        Analyse complete pour un symbole.

        Orchestre les 5 sous-analyseurs, calcule le score composite,
        genere l'explication LLM, et retourne un AdvisoryResult.
        """
        period = period or self.config.default_period
        interval = interval or self.config.default_interval
        symbol = symbol.upper()

        logger.info(f"Analyse de {symbol} ({period}, {interval})...")

        # Collecter les sous-signaux
        sub_signals: List[SubSignal] = []
        forecast_points: List[ForecastPoint] = []

        # 1. Technique
        try:
            tech_signal = self.technical.analyze(symbol, period, interval)
            sub_signals.append(tech_signal)
        except Exception as e:
            logger.error(f"Erreur technique {symbol}: {e}")
            sub_signals.append(
                SubSignal("technical", 0.0, 0.0, reasons=[str(e)])
            )

        # 2. ML
        try:
            ml_signal = self.ml.analyze(symbol)
            sub_signals.append(ml_signal)
        except Exception as e:
            logger.error(f"Erreur ML {symbol}: {e}")
            sub_signals.append(
                SubSignal("ml_model", 0.0, 0.0, reasons=[str(e)])
            )

        # 3. Sentiment
        try:
            sent_signal = self.sentiment.analyze(symbol)
            sub_signals.append(sent_signal)
        except Exception as e:
            logger.error(f"Erreur sentiment {symbol}: {e}")
            sub_signals.append(
                SubSignal("sentiment", 0.0, 0.0, reasons=[str(e)])
            )

        # 4. Prevision statistique (necessite des donnees daily)
        try:
            df_daily = yf.Ticker(symbol).history(period="1y", interval="1d")
            if not df_daily.empty:
                stat_signal, forecast_points = self.forecaster.analyze(
                    symbol, df_daily
                )
                sub_signals.append(stat_signal)
            else:
                sub_signals.append(
                    SubSignal(
                        "statistical", 0.0, 0.0, reasons=["Pas de donnees daily"]
                    )
                )
        except Exception as e:
            logger.error(f"Erreur prevision {symbol}: {e}")
            sub_signals.append(
                SubSignal("statistical", 0.0, 0.0, reasons=[str(e)])
            )

        # 5. Risque
        try:
            risk_signal = self.risk.analyze(symbol, portfolio_value, positions)
            sub_signals.append(risk_signal)
        except Exception as e:
            logger.error(f"Erreur risque {symbol}: {e}")
            sub_signals.append(
                SubSignal("risk", 0.0, 0.0, reasons=[str(e)])
            )

        # Calculer le score composite
        composite_score, overall_confidence = self._compute_composite(sub_signals)
        recommendation = score_to_recommendation(composite_score)

        # Prix actuel et niveaux
        current_price = 0.0
        entry_price = None
        stop_loss = None
        take_profit = None
        indicators = {}
        ohlcv_data = []

        tech_sub = next((s for s in sub_signals if s.source == "technical"), None)
        if tech_sub and tech_sub.details:
            current_price = (
                tech_sub.details.get("indicators", {})
                .get("price", {})
                .get("current", 0.0)
            )
            entry_price = tech_sub.details.get("entry_price")
            stop_loss = tech_sub.details.get("stop_loss")
            take_profit = tech_sub.details.get("take_profit")
            indicators = tech_sub.details.get("indicators", {})

        # Fallback pour le prix
        if current_price == 0.0:
            try:
                t = yf.Ticker(symbol)
                hist = t.history(period="1d")
                if not hist.empty:
                    current_price = float(hist["Close"].iloc[-1])
            except Exception:
                pass

        # Construire le resultat (sans explication LLM pour l'instant)
        result = AdvisoryResult(
            symbol=symbol,
            timestamp=datetime.now().isoformat(),
            composite_score=composite_score,
            recommendation=recommendation,
            confidence=overall_confidence,
            sub_signals=sub_signals,
            explanation_fr="",
            forecast=forecast_points,
            current_price=current_price,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            indicators=indicators,
        )

        # Generer l'explication LLM (peut etre lent)
        try:
            result_dict = result.to_dict()
            result.explanation_fr = self.explainer.explain(result_dict)
        except Exception as e:
            logger.error(f"Erreur explication {symbol}: {e}")
            result.explanation_fr = f"Recommandation : {recommendation.value}"

        # Persister en DB (non-bloquant)
        try:
            from database.advisory_db import save_advisory_analysis

            save_advisory_analysis(result.to_dict())
        except Exception as e:
            logger.warning(f"Erreur sauvegarde DB: {e}")

        logger.info(
            f"{symbol}: {recommendation.value} (score={composite_score:+.2f}, "
            f"confiance={overall_confidence:.0%})"
        )

        return result

    def analyze_watchlist(
        self,
        period: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> List[AdvisoryResult]:
        """Analyse tous les tickers de la watchlist."""
        try:
            from config.tickers import ALL_TICKERS
        except ImportError:
            ALL_TICKERS = ["NVDA", "MSFT", "AAPL", "GOOGL", "AMZN", "SPY", "QQQ"]

        results = []
        for ticker in ALL_TICKERS:
            try:
                result = self.analyze(ticker, period=period, interval=interval)
                results.append(result)
            except Exception as e:
                logger.error(f"Erreur analyse watchlist {ticker}: {e}")

        # Trier par score composite (du plus fort achat au plus fort vente)
        results.sort(key=lambda r: r.composite_score, reverse=True)
        return results

    def get_top_picks(
        self, n: int = 5, period: Optional[str] = None
    ) -> Dict[str, List[AdvisoryResult]]:
        """Retourne les top N achats et top N ventes."""
        all_results = self.analyze_watchlist(period=period)

        buys = [r for r in all_results if r.composite_score > 0.1]
        sells = [
            r
            for r in reversed(all_results)
            if r.composite_score < -0.1
        ]

        return {
            "top_buys": buys[:n],
            "top_sells": sells[:n],
        }

    def _compute_composite(
        self, sub_signals: List[SubSignal]
    ) -> tuple[float, float]:
        """
        Calcule le score composite et la confiance globale.

        Moyenne ponderee : score_i * confidence_i * weight_i
        Les sous-signaux avec confidence=0 sont ignores.
        """
        weights = self.config.weights
        numerator = 0.0
        denominator = 0.0

        for sub in sub_signals:
            w = weights.get(sub.source, 0.0)
            if sub.confidence > 0 and w > 0:
                numerator += sub.signal * sub.confidence * w
                denominator += sub.confidence * w

        if denominator > 0:
            composite = numerator / denominator
            # Confiance globale = somme des confidences ponderees / somme des poids
            total_weight = sum(weights.get(s.source, 0) for s in sub_signals)
            overall_confidence = denominator / total_weight if total_weight > 0 else 0
        else:
            composite = 0.0
            overall_confidence = 0.0

        composite = max(-1.0, min(1.0, composite))
        overall_confidence = max(0.0, min(1.0, overall_confidence))

        return composite, overall_confidence
