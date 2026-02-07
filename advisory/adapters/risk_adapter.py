"""
Adaptateur risque : wrappe RiskManager pour evaluer le risque d'un investissement.

Produit un signal negatif si l'exposition est trop elevee ou si le risque
depasse les seuils configures.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from typing import Dict, List, Optional

from advisory.models import SubSignal

logger = logging.getLogger(__name__)


class RiskAdapter:
    """Adapte RiskManager vers SubSignal."""

    def __init__(self):
        try:
            from core.risk_manager import RiskManager

            self.risk_manager = RiskManager()
            self._available = True
        except Exception as e:
            logger.warning(f"RiskManager indisponible: {e}")
            self._available = False
            self.risk_manager = None

    def analyze(
        self,
        symbol: str,
        portfolio_value: Optional[float] = None,
        positions: Optional[List[Dict]] = None,
    ) -> SubSignal:
        """
        Evalue le risque pour un symbole.

        Signal = 0 si sain, negatif si surexpose.
        Confidence = 0.7 fixe (le risque est un modificateur, pas un predicteur).
        """
        if not self._available:
            return SubSignal(
                source="risk",
                signal=0.0,
                confidence=0.0,
                details={"available": False},
                reasons=["RiskManager indisponible"],
            )

        signal = 0.0
        reasons = []
        details: Dict = {}

        # Evaluer l'exposition si on a un contexte de portfolio
        if portfolio_value and positions:
            exposure = self.risk_manager.calculate_portfolio_exposure(
                positions, portfolio_value
            )
            details["exposure_pct"] = exposure

            should_reduce, reason = self.risk_manager.should_reduce_exposure(
                positions, portfolio_value
            )
            if should_reduce:
                signal = -0.3
                reasons.append(reason)

            # Verifier si ce symbole est deja en portefeuille
            for pos in positions:
                if pos.get("symbol") == symbol:
                    risk_assessment = self.risk_manager.assess_position_risk(
                        symbol=symbol,
                        position_value=pos.get("market_value", 0),
                        portfolio_value=portfolio_value,
                        unrealized_plpc=pos.get("unrealized_plpc", 0),
                        days_held=pos.get("days_held", 0),
                    )
                    details["position_risk"] = risk_assessment

                    if risk_assessment["risk_level"] == "CRITIQUE":
                        signal = -0.8
                        reasons.extend(risk_assessment["warnings"])
                    elif risk_assessment["risk_level"] == "ELEVE":
                        signal = -0.4
                        reasons.extend(risk_assessment["warnings"])
                    elif risk_assessment["risk_level"] == "MOYEN":
                        signal = -0.2
                        reasons.extend(risk_assessment["warnings"])
                    break
        else:
            reasons.append("Pas de contexte portfolio, risque neutre")

        signal = max(-1.0, min(1.0, signal))

        return SubSignal(
            source="risk",
            signal=signal,
            confidence=0.7,
            details=details,
            reasons=reasons,
        )
