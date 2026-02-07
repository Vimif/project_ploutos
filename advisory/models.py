"""
Modeles de donnees pour le systeme advisory.

Definit les structures de donnees partagees par tous les sous-analyseurs
et le moteur principal.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from enum import Enum


class Recommendation(str, Enum):
    """Recommandation d'investissement."""
    ACHAT_FORT = "ACHAT_FORT"
    ACHAT = "ACHAT"
    NEUTRE = "NEUTRE"
    VENTE = "VENTE"
    VENTE_FORTE = "VENTE_FORTE"


@dataclass
class SubSignal:
    """Resultat d'un sous-analyseur individuel."""
    source: str                    # "technical", "ml_model", "sentiment", "statistical", "risk"
    signal: float                  # -1.0 (vente forte) a +1.0 (achat fort)
    confidence: float              # 0.0 a 1.0
    details: Dict = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)


@dataclass
class ForecastPoint:
    """Un point de prevision statistique."""
    date: str
    predicted: float
    lower_80: float
    upper_80: float
    lower_95: float
    upper_95: float


@dataclass
class AdvisoryResult:
    """Resultat complet d'analyse advisory pour un symbole."""
    symbol: str
    timestamp: str
    composite_score: float         # -1.0 a +1.0
    recommendation: Recommendation
    confidence: float              # 0.0 a 1.0
    sub_signals: List[SubSignal]
    explanation_fr: str            # Texte LLM ou template en francais
    forecast: List[ForecastPoint]
    current_price: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    indicators: Dict = field(default_factory=dict)
    ohlcv_data: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Conversion en dict serialisable JSON."""
        d = asdict(self)
        d["recommendation"] = self.recommendation.value
        return d


def score_to_recommendation(score: float) -> Recommendation:
    """Convertit un score composite [-1, +1] en recommandation."""
    if score >= 0.5:
        return Recommendation.ACHAT_FORT
    elif score >= 0.2:
        return Recommendation.ACHAT
    elif score > -0.2:
        return Recommendation.NEUTRE
    elif score > -0.5:
        return Recommendation.VENTE
    else:
        return Recommendation.VENTE_FORTE
