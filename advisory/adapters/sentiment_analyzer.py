"""
Analyseur de sentiment : recupere les news et calcule un score de sentiment.

Utilise Finnhub (free tier) pour les headlines et VADER pour le scoring.
Cache les resultats pour limiter les appels API.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

from advisory.models import SubSignal

logger = logging.getLogger(__name__)

# Import optionnel Finnhub
try:
    import finnhub

    FINNHUB_AVAILABLE = True
except ImportError:
    FINNHUB_AVAILABLE = False
    logger.info("finnhub-python non installe, sentiment desactive")

# Import optionnel VADER
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logger.info("vaderSentiment non installe, sentiment desactive")


class SentimentAnalyzer:
    """Analyse le sentiment des news financieres pour un symbole."""

    def __init__(self, cache_dir: str = "data/sentiment_cache"):
        self._available = FINNHUB_AVAILABLE and VADER_AVAILABLE
        self._finnhub_client = None
        self._vader = None
        self._cache_dir = Path(cache_dir)
        self._cache_ttl = timedelta(hours=6)

        if FINNHUB_AVAILABLE:
            api_key = os.getenv("FINNHUB_API_KEY", "")
            if api_key:
                self._finnhub_client = finnhub.Client(api_key=api_key)
            else:
                logger.warning("FINNHUB_API_KEY non configure, sentiment desactive")
                self._available = False

        if VADER_AVAILABLE:
            self._vader = SentimentIntensityAnalyzer()

        if self._available:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def analyze(self, symbol: str, days: int = 3) -> SubSignal:
        """
        Analyse le sentiment des news pour un symbole.

        1. Verifie le cache
        2. Fetch les headlines via Finnhub
        3. Score chaque headline avec VADER compound [-1, +1]
        4. Agrege avec poids temporel (recent = plus lourd)
        5. Confidence = min(n_articles/10, 1.0) * (1.0 - std(scores))
        """
        if not self._available:
            return SubSignal(
                source="sentiment",
                signal=0.0,
                confidence=0.0,
                details={"available": False},
                reasons=["Analyse de sentiment indisponible"],
            )

        # Verifier le cache
        cached = self._load_cache(symbol)
        if cached is not None:
            return cached

        # Fetch news
        headlines = self._fetch_news(symbol, days)
        if not headlines:
            result = SubSignal(
                source="sentiment",
                signal=0.0,
                confidence=0.0,
                details={"headlines_count": 0},
                reasons=[f"Aucune news trouvee pour {symbol}"],
            )
            self._save_cache(symbol, result)
            return result

        # Score chaque headline
        scores = []
        scored_headlines = []
        now = datetime.now()

        for h in headlines:
            compound = self._score_headline(h["headline"])
            age_hours = (now - h["datetime"]).total_seconds() / 3600
            # Poids temporel : decroit exponentiellement avec l'age
            time_weight = np.exp(-age_hours / (days * 24))

            scores.append(compound)
            scored_headlines.append(
                {
                    "headline": h["headline"],
                    "score": compound,
                    "weight": time_weight,
                    "source": h.get("source", ""),
                    "datetime": h["datetime"].isoformat(),
                }
            )

        # Agreger : moyenne ponderee par le temps
        weights = [sh["weight"] for sh in scored_headlines]
        total_weight = sum(weights)
        if total_weight > 0:
            weighted_avg = sum(s * w for s, w in zip(scores, weights)) / total_weight
        else:
            weighted_avg = 0.0

        # Confidence
        n_articles = len(headlines)
        coverage_factor = min(n_articles / 10.0, 1.0)
        agreement_factor = 1.0 - float(np.std(scores)) if len(scores) > 1 else 0.5
        agreement_factor = max(0.0, agreement_factor)
        confidence = coverage_factor * agreement_factor

        # Raisons
        reasons = []
        positive = sum(1 for s in scores if s > 0.05)
        negative = sum(1 for s in scores if s < -0.05)
        neutral = n_articles - positive - negative

        if weighted_avg > 0.1:
            reasons.append(f"Sentiment positif ({positive}/{n_articles} articles)")
        elif weighted_avg < -0.1:
            reasons.append(f"Sentiment negatif ({negative}/{n_articles} articles)")
        else:
            reasons.append(f"Sentiment mixte ({positive}+/{negative}-/{neutral}n)")

        signal = max(-1.0, min(1.0, weighted_avg))
        confidence = max(0.0, min(1.0, confidence))

        result = SubSignal(
            source="sentiment",
            signal=signal,
            confidence=confidence,
            details={
                "headlines_count": n_articles,
                "positive_count": positive,
                "negative_count": negative,
                "neutral_count": neutral,
                "avg_score": float(np.mean(scores)),
                "headlines": scored_headlines[:10],  # Top 10 pour l'UI
            },
            reasons=reasons,
        )
        self._save_cache(symbol, result)
        return result

    def _fetch_news(self, symbol: str, days: int) -> List[Dict]:
        """Fetch les news depuis Finnhub."""
        try:
            end = datetime.now()
            start = end - timedelta(days=days)

            news = self._finnhub_client.company_news(
                symbol,
                _from=start.strftime("%Y-%m-%d"),
                to=end.strftime("%Y-%m-%d"),
            )

            headlines = []
            for article in news:
                dt = datetime.fromtimestamp(article.get("datetime", 0))
                headlines.append(
                    {
                        "headline": article.get("headline", ""),
                        "source": article.get("source", ""),
                        "datetime": dt,
                        "url": article.get("url", ""),
                    }
                )

            logger.info(f"Sentiment: {len(headlines)} articles pour {symbol}")
            return headlines

        except Exception as e:
            logger.error(f"Erreur Finnhub pour {symbol}: {e}")
            return []

    def _score_headline(self, headline: str) -> float:
        """Score une headline avec VADER. Retourne compound [-1, +1]."""
        if not self._vader or not headline:
            return 0.0
        scores = self._vader.polarity_scores(headline)
        return scores["compound"]

    def _cache_path(self, symbol: str) -> Path:
        return self._cache_dir / f"{symbol.upper()}_sentiment.json"

    def _load_cache(self, symbol: str) -> Optional[SubSignal]:
        """Charge le cache si frais."""
        path = self._cache_path(symbol)
        if not path.exists():
            return None

        try:
            with open(path, "r") as f:
                data = json.load(f)

            cached_time = datetime.fromisoformat(data["timestamp"])
            if datetime.now() - cached_time > self._cache_ttl:
                return None

            return SubSignal(
                source=data["source"],
                signal=data["signal"],
                confidence=data["confidence"],
                details=data["details"],
                reasons=data["reasons"],
            )
        except Exception:
            return None

    def _save_cache(self, symbol: str, result: SubSignal) -> None:
        """Sauvegarde en cache JSON."""
        try:
            data = {
                "timestamp": datetime.now().isoformat(),
                "source": result.source,
                "signal": result.signal,
                "confidence": result.confidence,
                "details": result.details,
                "reasons": result.reasons,
            }
            with open(self._cache_path(symbol), "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Erreur sauvegarde cache sentiment {symbol}: {e}")
