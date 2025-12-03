# trading/alpaca_client.py
"""Client pour l'API Alpaca"""

# === FIX PATH ===
import sys
from pathlib import Path
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))
# ================

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from core.utils import setup_logging

logger = setup_logging(__name__, 'alpaca.log')

# Charger variables d'environnement
load_dotenv()

class AlpacaClient:
    """Client Alpaca pour le trading"""
    
    def __init__(self, paper_trading=True):
        """
        Initialiser le client Alpaca
        
        Args:
            paper_trading: True pour paper trading, False pour live
        """
        self.paper_trading = paper_trading
        
        # Récupérer les clés API
        if paper_trading:
            api_key = os.getenv('ALPACA_PAPER_API_KEY')
            api_secret = os.getenv('ALPACA_PAPER_SECRET_KEY')
        else:
            api_key = os.getenv('ALPACA_LIVE_API_KEY')
            api_secret = os.getenv('ALPACA_LIVE_SECRET_KEY')
        
        if not api_key or not api_secret:
            raise ValueError("❌ Clés API Alpaca manquantes dans .env")
        
        # Client de trading
        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=api_secret,
            paper=paper_trading
        )
        
        # Client de données
        self.data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=api_secret
        )
        
        logger.info(f"✅ Client Alpaca initialisé ({'Paper' if paper_trading else 'LIVE'})")
    
    def get_account(self):
        """Obtenir les infos du compte"""
        try:
            account = self.trading_client.get_account()
            return {
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'equity': float(account.equity),
                'daytrade_count': int(account.daytrade_count),
                'pattern_day_trader': account.pattern_day_trader
            }
        except Exception as e:
            logger.error(f"❌ Erreur récupération compte: {e}")
            return None
    
    def get_positions(self):
        """Obtenir toutes les positions ouvertes"""
        try:
            positions = self.trading_client.get_all_positions()
            return [{
                'symbol': pos.symbol,
                'qty': float(pos.qty),
                'market_value': float(pos.market_value),
                'cost_basis': float(pos.cost_basis),
                'unrealized_pl': float(pos.unrealized_pl),
                'unrealized_plpc': float(pos.unrealized_plpc),
                'current_price': float(pos.current_price),
                'avg_entry_price': float(pos.avg_entry_price)
            } for pos in positions]
        except Exception as e:
            logger.error(f"❌ Erreur récupération positions: {e}")
            return []
    
    def get_position(self, symbol):
        """Obtenir une position spécifique"""
        try:
            pos = self.trading_client.get_open_position(symbol)
            return {
                'symbol': pos.symbol,
                'qty': float(pos.qty),
                'market_value': float(pos.market_value),
                'unrealized_pl': float(pos.unrealized_pl),
                'current_price': float(pos.current_price),
                'avg_entry_price': float(pos.avg_entry_price)
            }
        except:
            return None
    
    def get_current_price(self, symbol):
        """Obtenir le prix actuel d'un ticker"""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.data_client.get_stock_latest_quote(request)
            
            if symbol in quotes:
                quote = quotes[symbol]
                # Prix moyen bid/ask
                price = (float(quote.bid_price) + float(quote.ask_price)) / 2
                return price
            return None
        except Exception as e:
            logger.error(f"❌ Erreur prix pour {symbol}: {e}")
            return None
    
    def place_market_order(self, symbol, qty, side='buy'):
        """
        Passer un ordre au marché
        
        Args:
            symbol: Ticker (ex: 'AAPL')
            qty: Quantité (nombre d'actions ou montant en $)
            side: 'buy' ou 'sell'
        
        Returns:
            Order object ou None
        """
        try:
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY
            )
            
            order = self.trading_client.submit_order(order_data)
            
            logger.info(f"✅ Ordre {side.upper()} placé: {symbol} x{qty}")
            
            return {
                'id': order.id,
                'symbol': order.symbol,
                'qty': float(order.qty),
                'side': order.side,
                'status': order.status,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None
            }
        
        except Exception as e:
            logger.error(f"❌ Erreur ordre {side} pour {symbol}: {e}")
            return None
    
    def place_limit_order(self, symbol, qty, limit_price, side='buy'):
        """Passer un ordre limite"""
        try:
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            
            order_data = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price
            )
            
            order = self.trading_client.submit_order(order_data)
            
            logger.info(f"✅ Ordre LIMIT {side.upper()} placé: {symbol} x{qty} @ ${limit_price}")
            
            return {
                'id': order.id,
                'symbol': order.symbol,
                'qty': float(order.qty),
                'limit_price': float(order.limit_price),
                'status': order.status
            }
        
        except Exception as e:
            logger.error(f"❌ Erreur ordre limite pour {symbol}: {e}")
            return None
    
    def close_position(self, symbol):
        """Fermer complètement une position"""
        try:
            self.trading_client.close_position(symbol)
            logger.info(f"✅ Position fermée: {symbol}")
            return True
        except Exception as e:
            logger.error(f"❌ Erreur fermeture position {symbol}: {e}")
            return False
    
    def close_all_positions(self):
        """Fermer toutes les positions"""
        try:
            self.trading_client.close_all_positions(cancel_orders=True)
            logger.info("✅ Toutes les positions fermées")
            return True
        except Exception as e:
            logger.error(f"❌ Erreur fermeture globale: {e}")
            return False
    
    def get_orders(self, status='open'):
        """Obtenir les ordres"""
        try:
            orders = self.trading_client.get_orders(filter={'status': status})
            return [{
                'id': order.id,
                'symbol': order.symbol,
                'qty': float(order.qty),
                'side': order.side,
                'status': order.status,
                'order_type': order.order_type
            } for order in orders]
        except Exception as e:
            logger.error(f"❌ Erreur récupération ordres: {e}")
            return []
    
    def cancel_order(self, order_id):
        """Annuler un ordre"""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info(f"✅ Ordre annulé: {order_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Erreur annulation ordre: {e}")
            return False
