# trading/broker_interface.py
"""Interface abstraite pour les brokers (Alpaca, eToro, etc.)"""

from abc import ABC, abstractmethod
from typing import Optional


class BrokerInterface(ABC):
    """
    Interface commune pour tous les brokers.
    Permet de switcher entre Alpaca et eToro sans modifier le code de trading.
    """

    @abstractmethod
    def get_account(self) -> Optional[dict]:
        """
        Obtenir les infos du compte.

        Returns:
            dict avec au minimum:
                - cash (float)
                - portfolio_value (float)
                - buying_power (float)
                - equity (float)
            ou None en cas d'erreur
        """
        pass

    @abstractmethod
    def get_positions(self) -> list:
        """
        Obtenir toutes les positions ouvertes.

        Returns:
            list de dicts avec:
                - symbol (str)
                - qty (float)
                - market_value (float)
                - cost_basis (float)
                - unrealized_pl (float)
                - unrealized_plpc (float)
                - current_price (float)
                - avg_entry_price (float)
        """
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[dict]:
        """
        Obtenir une position spécifique.

        Args:
            symbol: Ticker (ex: 'AAPL')

        Returns:
            dict position ou None si pas de position
        """
        pass

    @abstractmethod
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Obtenir le prix actuel d'un ticker.

        Args:
            symbol: Ticker

        Returns:
            float prix ou None
        """
        pass

    @abstractmethod
    def place_market_order(self, symbol: str, qty: int, side: str = 'buy', reason: str = '') -> Optional[dict]:
        """
        Passer un ordre au marché.

        Args:
            symbol: Ticker
            qty: Quantité
            side: 'buy' ou 'sell'
            reason: Raison du trade

        Returns:
            dict avec détails de l'ordre ou None
        """
        pass

    @abstractmethod
    def place_limit_order(self, symbol: str, qty: int, limit_price: float, side: str = 'buy') -> Optional[dict]:
        """
        Passer un ordre limite.

        Args:
            symbol: Ticker
            qty: Quantité
            limit_price: Prix limite
            side: 'buy' ou 'sell'

        Returns:
            dict avec détails de l'ordre ou None
        """
        pass

    @abstractmethod
    def close_position(self, symbol: str, reason: str = '') -> bool:
        """
        Fermer complètement une position.

        Args:
            symbol: Ticker
            reason: Raison de la fermeture

        Returns:
            bool: Succès/échec
        """
        pass

    @abstractmethod
    def close_all_positions(self) -> bool:
        """
        Fermer toutes les positions.

        Returns:
            bool: Succès/échec
        """
        pass

    @abstractmethod
    def get_orders(self, status: str = 'open', limit: int = 50) -> list:
        """
        Obtenir les ordres.

        Args:
            status: 'open', 'closed', 'all'
            limit: Nombre max d'ordres

        Returns:
            list de dicts ordres
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Annuler un ordre.

        Args:
            order_id: ID de l'ordre

        Returns:
            bool: Succès/échec
        """
        pass

    def cancel_orders_for_symbol(self, symbol: str) -> int:
        """
        Annuler tous les ordres en cours pour un ticker.
        Implémentation par défaut utilisant get_orders + cancel_order.

        Args:
            symbol: Ticker

        Returns:
            int: Nombre d'ordres annulés
        """
        try:
            orders = self.get_orders(status='open')
            cancelled = 0
            for order in orders:
                if order.get('symbol') == symbol:
                    if self.cancel_order(order['id']):
                        cancelled += 1
            return cancelled
        except Exception:
            return 0

    def log_current_positions(self):
        """Logger les positions actuelles (optionnel)."""
        pass
