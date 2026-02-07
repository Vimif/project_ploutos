"""
Adaptateur ML : wrappe BrainTrader (modele PPO) pour predictions.

Cache les predictions de predict_all() et indexe par symbole.
Graceful fallback si pas de modele .zip disponible.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from typing import Dict, Optional
from datetime import datetime, timedelta

from advisory.models import SubSignal

logger = logging.getLogger(__name__)


class MLAdapter:
    """Adapte BrainTrader vers SubSignal."""

    def __init__(self, model_path: Optional[str] = None):
        self._predictions_cache: Dict = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=30)
        self._available = False

        try:
            from trading.brain_trader import BrainTrader

            self._brain_trader = BrainTrader(model_path=model_path)
            self._available = True
            logger.info("MLAdapter: BrainTrader charge avec succes")
        except Exception as e:
            logger.warning(f"MLAdapter: BrainTrader indisponible: {e}")
            self._brain_trader = None

    def _refresh_cache(self) -> None:
        """Regenere le cache de predictions si expire."""
        now = datetime.now()
        if (
            self._cache_timestamp is not None
            and now - self._cache_timestamp < self._cache_ttl
        ):
            return

        if not self._available:
            return

        try:
            predictions = self._brain_trader.predict_all()
            # Aplatir le dict par categorie en dict par ticker
            self._predictions_cache = {}
            for _category, preds in predictions.items():
                for pred in preds:
                    self._predictions_cache[pred["ticker"]] = pred
            self._cache_timestamp = now
            logger.info(
                f"MLAdapter: {len(self._predictions_cache)} predictions mises en cache"
            )
        except Exception as e:
            logger.error(f"MLAdapter: Erreur predict_all: {e}")

    def analyze(self, symbol: str) -> SubSignal:
        """
        Prediction ML pour un symbole.

        Mappe BUY -> +0.8, SELL -> -0.8, HOLD -> 0.0.
        Confidence fixe a 0.8 (le modele ne fournit pas de vraie confidence).
        """
        if not self._available:
            return SubSignal(
                source="ml_model",
                signal=0.0,
                confidence=0.0,
                details={"available": False},
                reasons=["Modele PPO indisponible"],
            )

        self._refresh_cache()

        pred = self._predictions_cache.get(symbol.upper())
        if pred is None:
            return SubSignal(
                source="ml_model",
                signal=0.0,
                confidence=0.0,
                details={"available": True, "symbol_found": False},
                reasons=[f"Pas de prediction pour {symbol}"],
            )

        action = pred.get("action", "HOLD")
        action_map = {"BUY": 0.8, "SELL": -0.8, "HOLD": 0.0}
        signal = action_map.get(action, 0.0)

        return SubSignal(
            source="ml_model",
            signal=signal,
            confidence=0.8,
            details={
                "action": action,
                "capital_suggested": pred.get("capital", 0),
            },
            reasons=[f"Modele PPO: {action}"],
        )
