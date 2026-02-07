"""Configuration pour le moteur advisory."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class AdvisoryConfig:
    """Configuration du systeme advisory."""

    # Poids des sous-analyseurs (doivent sommer a 1.0)
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "technical": 0.30,
            "ml_model": 0.25,
            "sentiment": 0.20,
            "statistical": 0.15,
            "risk": 0.10,
        }
    )

    # LLM (Ollama)
    ollama_model: str = "mistral"
    ollama_url: str = "http://localhost:11434"

    # Sentiment
    sentiment_provider: str = "finnhub"
    sentiment_cache_hours: int = 6

    # Previsions statistiques
    forecast_method: str = "auto_arima"
    forecast_horizon: int = 5

    # Cache general
    cache_ttl_minutes: int = 30

    # Parametres d'analyse par defaut
    default_period: str = "3mo"
    default_interval: str = "1h"

    # Seuils de recommandation
    strong_buy_threshold: float = 0.5
    buy_threshold: float = 0.2
    sell_threshold: float = -0.2
    strong_sell_threshold: float = -0.5
