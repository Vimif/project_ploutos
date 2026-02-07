"""
Previsionneur statistique : utilise AutoARIMA pour predire les prix futurs.

Produit un signal base sur la direction et magnitude de la prevision,
ainsi que des ForecastPoints pour le graphique.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from advisory.models import SubSignal, ForecastPoint

logger = logging.getLogger(__name__)

# Import optionnel statsforecast
try:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    STATSFORECAST_AVAILABLE = True
except ImportError:
    STATSFORECAST_AVAILABLE = False
    logger.info("statsforecast non installe, previsions desactivees")


class StatisticalForecaster:
    """Previsions statistiques via AutoARIMA."""

    def __init__(self, horizon: int = 5):
        """
        Args:
            horizon: Nombre de jours a predire (defaut 5 = 1 semaine de trading)
        """
        self._available = STATSFORECAST_AVAILABLE
        self.horizon = horizon

    def analyze(
        self, symbol: str, df: pd.DataFrame
    ) -> Tuple[SubSignal, List[ForecastPoint]]:
        """
        Prevision statistique pour un symbole.

        1. Prepare les prix Close en serie temporelle
        2. Fit AutoARIMA
        3. Predit horizon jours avec intervalles de confiance
        4. Calcule signal base sur la direction et magnitude de la prevision
        5. Confidence basee sur la largeur de l'intervalle

        Args:
            symbol: Ticker
            df: DataFrame OHLCV (doit avoir une colonne 'Close')

        Returns:
            (SubSignal, List[ForecastPoint])
        """
        if not self._available:
            return (
                SubSignal(
                    source="statistical",
                    signal=0.0,
                    confidence=0.0,
                    details={"available": False},
                    reasons=["statsforecast non installe"],
                ),
                [],
            )

        if df is None or len(df) < 60:
            return (
                SubSignal(
                    source="statistical",
                    signal=0.0,
                    confidence=0.0,
                    details={"available": True, "error": "Donnees insuffisantes"},
                    reasons=["Moins de 60 jours de donnees"],
                ),
                [],
            )

        try:
            return self._run_forecast(symbol, df)
        except Exception as e:
            logger.error(f"Erreur prevision {symbol}: {e}")
            return (
                SubSignal(
                    source="statistical",
                    signal=0.0,
                    confidence=0.0,
                    details={"available": True, "error": str(e)},
                    reasons=[f"Erreur de prevision: {e}"],
                ),
                [],
            )

    def _run_forecast(
        self, symbol: str, df: pd.DataFrame
    ) -> Tuple[SubSignal, List[ForecastPoint]]:
        """Execute la prevision AutoARIMA."""
        # Preparer la serie pour statsforecast
        # Format requis: colonnes ['unique_id', 'ds', 'y']
        close = df["Close"].dropna()
        ts_df = pd.DataFrame(
            {
                "unique_id": symbol,
                "ds": close.index,
                "y": close.values,
            }
        )
        # S'assurer que ds est bien un datetime sans timezone
        ts_df["ds"] = pd.to_datetime(ts_df["ds"]).dt.tz_localize(None)

        # Fit + predict
        model = StatsForecast(
            models=[AutoARIMA(season_length=5)],
            freq="B",  # Business days
        )
        model.fit(ts_df)
        forecast = model.predict(h=self.horizon, level=[80, 95])

        # Extraire les resultats
        current_price = float(close.iloc[-1])
        forecast_points = []

        for _, row in forecast.iterrows():
            fp = ForecastPoint(
                date=str(row["ds"].date()),
                predicted=float(row["AutoARIMA"]),
                lower_80=float(row.get("AutoARIMA-lo-80", row["AutoARIMA"])),
                upper_80=float(row.get("AutoARIMA-hi-80", row["AutoARIMA"])),
                lower_95=float(row.get("AutoARIMA-lo-95", row["AutoARIMA"])),
                upper_95=float(row.get("AutoARIMA-hi-95", row["AutoARIMA"])),
            )
            forecast_points.append(fp)

        # Calculer signal
        if not forecast_points:
            return (
                SubSignal(
                    source="statistical",
                    signal=0.0,
                    confidence=0.0,
                    details={},
                    reasons=["Prevision vide"],
                ),
                [],
            )

        last_forecast = forecast_points[-1]
        predicted_return = (last_forecast.predicted - current_price) / current_price

        # ATR pour normaliser le signal
        if "High" in df.columns and "Low" in df.columns:
            atr = float(
                (df["High"] - df["Low"]).rolling(window=14).mean().iloc[-1]
            )
            atr_pct = atr / current_price if current_price > 0 else 0.01
        else:
            atr_pct = 0.02  # Defaut 2%

        # Signal normalise via tanh
        signal = float(np.tanh(predicted_return / max(atr_pct, 0.001)))

        # Confidence basee sur la largeur de l'intervalle de confiance
        interval_width_80 = last_forecast.upper_80 - last_forecast.lower_80
        relative_width = interval_width_80 / current_price if current_price > 0 else 1.0
        # Plus l'intervalle est etroit, plus on est confiant
        # width ~0% -> confidence ~1.0, width ~10%+ -> confidence ~0.2
        confidence = max(0.1, min(1.0, 1.0 - relative_width * 5))

        # Raisons
        reasons = []
        pct_change = predicted_return * 100
        if pct_change > 1:
            reasons.append(
                f"Prevision hausse +{pct_change:.1f}% a J+{self.horizon}"
            )
        elif pct_change < -1:
            reasons.append(
                f"Prevision baisse {pct_change:.1f}% a J+{self.horizon}"
            )
        else:
            reasons.append(f"Prevision stable ({pct_change:+.1f}%) a J+{self.horizon}")

        reasons.append(
            f"Intervalle 80%: ${last_forecast.lower_80:.2f} - ${last_forecast.upper_80:.2f}"
        )

        details = {
            "current_price": current_price,
            "predicted_price_end": last_forecast.predicted,
            "predicted_return_pct": pct_change,
            "interval_80_width_pct": relative_width * 100,
            "atr_pct": atr_pct * 100,
        }

        signal = max(-1.0, min(1.0, signal))
        confidence = max(0.0, min(1.0, confidence))

        return (
            SubSignal(
                source="statistical",
                signal=signal,
                confidence=confidence,
                details=details,
                reasons=reasons,
            ),
            forecast_points,
        )
