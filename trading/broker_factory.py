# trading/broker_factory.py
"""Factory pour créer le bon client broker selon la configuration."""

import logging
import os

from dotenv import load_dotenv

from trading.broker_interface import BrokerInterface

load_dotenv()
logger = logging.getLogger(__name__)

# Broker par défaut: lire depuis env, sinon 'etoro'
DEFAULT_BROKER = os.getenv("BROKER", "etoro").lower()


def create_broker(broker_name: str = None, paper_trading: bool = True) -> BrokerInterface:
    """
    Créer une instance du broker demandé.

    Args:
        broker_name: 'alpaca' ou 'etoro' (si None, utilise BROKER env var)
        paper_trading: True pour paper/demo trading

    Returns:
        Instance de BrokerInterface

    Raises:
        ValueError: Si le broker n'est pas supporté
    """
    name = (broker_name or DEFAULT_BROKER).lower().strip()

    if name == "etoro":
        from trading.etoro_client import EToroClient

        logger.info("Initialisation broker: eToro")
        return EToroClient(paper_trading=paper_trading)

    elif name == "alpaca":
        from trading.alpaca_client import AlpacaClient

        logger.info("Initialisation broker: Alpaca")
        return AlpacaClient(paper_trading=paper_trading)

    else:
        raise ValueError(
            f"Broker '{name}' non supporté. " f"Brokers disponibles: 'etoro', 'alpaca'"
        )


def get_available_brokers() -> list:
    """Retourne la liste des brokers disponibles et configurés."""
    brokers = []

    # Check eToro
    if os.getenv("ETORO_SUBSCRIPTION_KEY") and os.getenv("ETORO_USERNAME"):
        brokers.append("etoro")

    # Check Alpaca
    alpaca_key = os.getenv("ALPACA_PAPER_API_KEY") or os.getenv("ALPACA_API_KEY")
    alpaca_secret = os.getenv("ALPACA_PAPER_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
    if alpaca_key and alpaca_secret:
        brokers.append("alpaca")

    return brokers
