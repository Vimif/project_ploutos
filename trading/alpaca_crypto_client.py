# trading/alpaca_crypto_client.py
"""Client pour l'API Crypto d'Alpaca"""

# === FIX PATH ===
import sys
from pathlib import Path
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))
# ================

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoLatestQuoteRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass

import os
from dotenv import load_dotenv
from core.utils import setup_logging
from datetime import datetime, timedelta

logger = setup_logging(__name__, 'alpaca_crypto.log')
load_dotenv()

class AlpacaCryptoClient:
    """Client spécialisé pour les cryptos sur Alpaca"""
    
    # Mapping des symboles crypto
    CRYPTO_SYMBOLS = {
        'BTC-USD': 'BTC/USD',
        'ETH-USD': 'ETH/USD',
        'DOGE-USD': 'DOGE/USD',
        'LTC-USD': 'LTC/USD',
        'BCH-USD': 'BCH/USD',
        'AVAX-USD': 'AVAX/USD',
        'DOT-USD': 'DOT/USD',
        'LINK-USD': 'LINK/USD',
        'UNI-USD': 'UNI/USD',
        'AAVE-USD': 'AAVE/USD',
    }
    
    def __init__(self, paper_trading=True):
        """
        Initialiser le client crypto
        
        Args:
            paper_trading: True pour paper, False pour live
        """
        self.paper_trading = paper_trading
        
        if paper_trading:
            api_key = os.getenv('ALPACA_PAPER_API_KEY')
            api_secret = os.getenv('ALPACA_PAPER_SECRET_KEY')
        else:
            api_key = os.getenv('ALPACA_LIVE_API_KEY')
            api_secret = os.getenv('ALPACA_LIVE_SECRET_KEY')
        
        if not api_key or not api_secret:
            raise ValueError("❌ Clés API Alpaca manquantes")
        
        # Client pour les données historiques crypto
        self.data_client = CryptoHistoricalDataClient(
            api_key=api_key,
            secret_key=api_secret
        )
        
        # Client pour le trading
        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=api_secret,
            paper=paper_trading
        )
        
        logger.info(f"✅ Client Crypto Alpaca initialisé ({'Paper' if paper_trading else 'LIVE'})")
    
    def normalize_symbol(self, symbol):
        """
        Convertir symbole au format Alpaca crypto
        
        Args:
            symbol: 'BTC-USD' ou 'BTC/USD'
        
        Returns:
            'BTC/USD' (format Alpaca)
        """
        if symbol in self.CRYPTO_SYMBOLS:
            return self.CRYPTO_SYMBOLS[symbol]
        
        # Si déjà au bon format
        if '/' in symbol:
            return symbol
        
        # Convertir - en /
        return symbol.replace('-', '/')
    
    def get_crypto_price(self, symbol):
        """
        Obtenir le prix actuel d'une crypto
        
        Args:
            symbol: 'BTC-USD' ou 'BTC/USD'
        
        Returns:
            float: prix actuel ou None
        """
        try:
            alpaca_symbol = self.normalize_symbol(symbol)
            
            request = CryptoLatestQuoteRequest(symbol_or_symbols=alpaca_symbol)
            quotes = self.data_client.get_crypto_latest_quote(request)
            
            if alpaca_symbol in quotes:
                quote = quotes[alpaca_symbol]
                # Prix moyen bid/ask
                price = (float(quote.bid_price) + float(quote.ask_price)) / 2
                return price
            
            logger.warning(f"⚠️  Prix non disponible pour {symbol}")
            return None
        
        except Exception as e:
            logger.error(f"❌ Erreur prix crypto {symbol}: {e}")
            return None
    
    def get_crypto_bars(self, symbol, days=90):
        """
        Obtenir l'historique d'une crypto
        
        Args:
            symbol: 'BTC-USD'
            days: Nombre de jours d'historique
        
        Returns:
            DataFrame pandas ou None
        """
        try:
            alpaca_symbol = self.normalize_symbol(symbol)
            
            request = CryptoBarsRequest(
                symbol_or_symbols=alpaca_symbol,
                timeframe=TimeFrame.Day,
                start=datetime.now() - timedelta(days=days)
            )
            
            bars = self.data_client.get_crypto_bars(request)
            
            if alpaca_symbol in bars:
                df = bars[alpaca_symbol].df
                return df
            
            return None
        
        except Exception as e:
            logger.error(f"❌ Erreur historique {symbol}: {e}")
            return None
    
    def place_crypto_market_order(self, symbol, qty, side='buy', notional=None):
        """
        Passer un ordre market sur crypto
        
        Args:
            symbol: 'BTC-USD' ou 'BTC/USD'
            qty: Quantité de crypto (ex: 0.001 BTC) OU None si notional
            side: 'buy' ou 'sell'
            notional: Montant en $ (ex: 100 pour acheter 100$ de BTC)
        
        Returns:
            Order dict ou None
        """
        try:
            alpaca_symbol = self.normalize_symbol(symbol)
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            
            # Ordre par quantité ou par montant
            if notional:
                order_data = MarketOrderRequest(
                    symbol=alpaca_symbol,
                    notional=notional,  # Montant en $
                    side=order_side,
                    time_in_force=TimeInForce.GTC  # Good Till Cancel
                )
                logger.info(f"📝 Ordre CRYPTO {side.upper()}: {alpaca_symbol} pour ${notional}")
            else:
                order_data = MarketOrderRequest(
                    symbol=alpaca_symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=TimeInForce.GTC
                )
                logger.info(f"📝 Ordre CRYPTO {side.upper()}: {alpaca_symbol} x{qty}")
            
            order = self.trading_client.submit_order(order_data)
            
            logger.info(f"✅ Ordre crypto placé: {order.id}")
            
            return {
                'id': order.id,
                'symbol': order.symbol,
                'qty': float(order.qty) if order.qty else None,
                'notional': float(order.notional) if hasattr(order, 'notional') else None,
                'side': order.side,
                'status': order.status,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None
            }
        
        except Exception as e:
            logger.error(f"❌ Erreur ordre crypto {symbol}: {e}")
            return None
    
    def place_crypto_limit_order(self, symbol, qty, limit_price, side='buy'):
        """Ordre limite sur crypto"""
        try:
            alpaca_symbol = self.normalize_symbol(symbol)
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            
            order_data = LimitOrderRequest(
                symbol=alpaca_symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.GTC,
                limit_price=limit_price
            )
            
            order = self.trading_client.submit_order(order_data)
            logger.info(f"✅ Ordre LIMIT crypto placé: {alpaca_symbol} x{qty} @ ${limit_price}")
            
            return {
                'id': order.id,
                'symbol': order.symbol,
                'qty': float(order.qty),
                'limit_price': float(order.limit_price),
                'status': order.status
            }
        
        except Exception as e:
            logger.error(f"❌ Erreur ordre limite crypto: {e}")
            return None
    
    def get_crypto_positions(self):
        """Obtenir les positions crypto ouvertes"""
        try:
            all_positions = self.trading_client.get_all_positions()
            
            # Filtrer uniquement les cryptos
            crypto_positions = []
            for pos in all_positions:
                # Les cryptos ont le format 'BTC/USD'
                if '/' in pos.symbol:
                    crypto_positions.append({
                        'symbol': pos.symbol,
                        'qty': float(pos.qty),
                        'market_value': float(pos.market_value),
                        'cost_basis': float(pos.cost_basis),
                        'unrealized_pl': float(pos.unrealized_pl),
                        'unrealized_plpc': float(pos.unrealized_plpc),
                        'current_price': float(pos.current_price),
                        'avg_entry_price': float(pos.avg_entry_price)
                    })
            
            return crypto_positions
        
        except Exception as e:
            logger.error(f"❌ Erreur récupération positions crypto: {e}")
            return []
    
    def get_crypto_position(self, symbol):
        """Obtenir une position crypto spécifique"""
        try:
            alpaca_symbol = self.normalize_symbol(symbol)
            pos = self.trading_client.get_open_position(alpaca_symbol)
            
            return {
                'symbol': pos.symbol,
                'qty': float(pos.qty),
                'market_value': float(pos.market_value),
                'unrealized_pl': float(pos.unrealized_pl),
                'unrealized_plpc': float(pos.unrealized_plpc),
                'current_price': float(pos.current_price),
                'avg_entry_price': float(pos.avg_entry_price)
            }
        
        except:
            return None
    
    def close_crypto_position(self, symbol):
        """Fermer une position crypto"""
        try:
            alpaca_symbol = self.normalize_symbol(symbol)
            self.trading_client.close_position(alpaca_symbol)
            logger.info(f"✅ Position crypto fermée: {alpaca_symbol}")
            return True
        
        except Exception as e:
            logger.error(f"❌ Erreur fermeture position crypto {symbol}: {e}")
            return False
    
    def get_available_cryptos(self):
        """Lister les cryptos disponibles sur Alpaca"""
        return list(self.CRYPTO_SYMBOLS.keys())
