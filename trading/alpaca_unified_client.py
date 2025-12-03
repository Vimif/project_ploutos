# trading/alpaca_unified_client.py
"""Client unifié pour Stocks ET Crypto"""

# === FIX PATH ===
import sys
from pathlib import Path
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))
# ================

from trading.alpaca_client import AlpacaClient
from trading.alpaca_crypto_client import AlpacaCryptoClient
from core.utils import setup_logging

logger = setup_logging(__name__, 'alpaca_unified.log')

class AlpacaUnifiedClient:
    """Client unifié gérant stocks ET cryptos"""
    
    def __init__(self, paper_trading=True):
        """
        Initialiser les 2 clients
        
        Args:
            paper_trading: True pour paper, False pour live
        """
        self.paper_trading = paper_trading
        
        # Client stocks
        self.stock_client = AlpacaClient(paper_trading=paper_trading)
        
        # Client crypto
        self.crypto_client = AlpacaCryptoClient(paper_trading=paper_trading)
        
        logger.info("✅ Client unifié initialisé (Stocks + Crypto)")
    
    def is_crypto(self, symbol):
        """Déterminer si c'est une crypto"""
        crypto_keywords = ['BTC', 'ETH', 'DOGE', 'LTC', 'BCH', 'AVAX', 'DOT', 'LINK', 'UNI', 'AAVE']
        return any(crypto in symbol.upper() for crypto in crypto_keywords)
    
    def get_account(self):
        """Obtenir infos du compte (même pour les 2)"""
        return self.stock_client.get_account()
    
    def get_current_price(self, symbol):
        """Obtenir prix (routage auto stock/crypto)"""
        if self.is_crypto(symbol):
            return self.crypto_client.get_crypto_price(symbol)
        else:
            return self.stock_client.get_current_price(symbol)
    
    def place_market_order(self, symbol, qty=None, side='buy', notional=None):
        """
        Ordre market (routage auto)
        
        Args:
            symbol: Ticker
            qty: Quantité (actions ou crypto)
            side: 'buy' ou 'sell'
            notional: Montant en $ (pour crypto, optionnel)
        """
        if self.is_crypto(symbol):
            return self.crypto_client.place_crypto_market_order(
                symbol, qty, side, notional=notional
            )
        else:
            return self.stock_client.place_market_order(symbol, qty, side)
    
    def place_limit_order(self, symbol, qty, limit_price, side='buy'):
        """Ordre limite (routage auto)"""
        if self.is_crypto(symbol):
            return self.crypto_client.place_crypto_limit_order(
                symbol, qty, limit_price, side
            )
        else:
            return self.stock_client.place_limit_order(symbol, qty, limit_price, side)
    
    def get_positions(self):
        """Obtenir TOUTES les positions (stocks + crypto)"""
        stock_positions = self.stock_client.get_positions()
        crypto_positions = self.crypto_client.get_crypto_positions()
        return stock_positions + crypto_positions
    
    def get_position(self, symbol):
        """Obtenir une position spécifique"""
        if self.is_crypto(symbol):
            return self.crypto_client.get_crypto_position(symbol)
        else:
            return self.stock_client.get_position(symbol)
    
    def close_position(self, symbol):
        """Fermer une position"""
        if self.is_crypto(symbol):
            return self.crypto_client.close_crypto_position(symbol)
        else:
            return self.stock_client.close_position(symbol)
    
    def close_all_positions(self):
        """Fermer toutes les positions (stocks uniquement via API)"""
        # Note: Alpaca close_all ne gère que les stocks
        # Il faut fermer les cryptos individuellement
        result = self.stock_client.close_all_positions()
        
        # Fermer aussi les positions crypto
        crypto_positions = self.crypto_client.get_crypto_positions()
        for pos in crypto_positions:
            self.crypto_client.close_crypto_position(pos['symbol'])
        
        return result
    
    def get_orders(self, status='open'):
        """Obtenir les ordres (via stock client, marche pour les 2)"""
        return self.stock_client.get_orders(status=status)
    
    def cancel_order(self, order_id):
        """Annuler un ordre"""
        return self.stock_client.cancel_order(order_id)