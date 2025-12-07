# trading/alpaca_client.py
"""Client pour l'API Alpaca avec intégration BDD"""

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

# ========== INTÉGRATION BASE DE DONNÉES ==========
try:
    from database.db import log_trade, log_all_positions
    DB_AVAILABLE = True
    logger.info("✅ Module database disponible")
except ImportError:
    DB_AVAILABLE = False
    logger.warning("⚠️  Module database non disponible")

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
            api_key = os.getenv('ALPACA_PAPER_API_KEY') or os.getenv('ALPACA_API_KEY')
            api_secret = os.getenv('ALPACA_PAPER_SECRET_KEY') or os.getenv('ALPACA_SECRET_KEY')
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
                'last_equity': float(account.last_equity),
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
                'unrealized_plpc': float(pos.unrealized_plpc),
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
    
    def cancel_orders_for_symbol(self, symbol):
        """
        ★ NOUVEAU: Annuler tous les ordres en cours pour un ticker
        Utile pour éviter les wash trades
        
        Args:
            symbol: Ticker
        
        Returns:
            int: Nombre d'ordres annulés
        """
        try:
            orders = self.get_orders(status='open')
            cancelled = 0
            
            for order in orders:
                if order['symbol'] == symbol:
                    self.cancel_order(order['id'])
                    cancelled += 1
            
            if cancelled > 0:
                logger.info(f"✅ {symbol}: {cancelled} ordre(s) annulé(s)")
            
            return cancelled
        except Exception as e:
            logger.error(f"❌ Erreur annulation ordres pour {symbol}: {e}")
            return 0
    
    def place_market_order(self, symbol, qty, side='buy', reason=''):
        """
        Passer un ordre au marché AVEC LOGGING BDD
        
        Args:
            symbol: Ticker (ex: 'AAPL')
            qty: Quantité (nombre d'actions)
            side: 'buy' ou 'sell'
            reason: Raison du trade (optionnel)
        
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
            
            order_dict = {
                'id': order.id,
                'symbol': order.symbol,
                'qty': float(order.qty),
                'side': order.side,
                'status': order.status,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None
            }
            
            # ========== LOGGER DANS LA BDD (SAFE) ==========
            if DB_AVAILABLE:
                try:
                    account = self.get_account()
                    price = float(order.filled_avg_price) if order.filled_avg_price else self.get_current_price(symbol)
                    
                    if price and price > 0:
                        log_trade(
                            symbol=symbol,
                            action=side.upper(),
                            quantity=qty,
                            price=price,
                            amount=qty * price,
                            reason=reason,
                            portfolio_value=account['portfolio_value'] if account else None,
                            order_id=order.id
                        )
                except Exception as db_error:
                    # ★ NE PAS CRASH SI BDD FAIL
                    logger.warning(f"⚠️  Log BDD échoué: {db_error}")
            
            return order_dict
        
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
    
    def close_position(self, symbol, reason=''):
        """
        ★ FIX: Fermer complètement une position AVEC GESTION WASH TRADE
        
        Args:
            symbol: Ticker
            reason: Raison de la fermeture
        
        Returns:
            bool: Succès/échec
        """
        try:
            # ★ 1. Vérifier que la position existe
            position = self.get_position(symbol)
            
            if not position:
                logger.warning(f"⚠️  {symbol}: Pas de position à fermer")
                return False
            
            qty = position['qty']
            current_price = position['current_price']
            
            # ★ 2. Annuler tous les ordres en cours pour éviter wash trade
            self.cancel_orders_for_symbol(symbol)
            
            # ★ 3. Attendre 1 seconde pour que les annulations soient prises en compte
            import time
            time.sleep(1)
            
            # ★ 4. Fermer la position
            self.trading_client.close_position(symbol)
            logger.info(f"✅ Position fermée: {symbol}")
            
            # ★ 5. Logger le SELL (SAFE)
            if DB_AVAILABLE:
                try:
                    account = self.get_account()
                    log_trade(
                        symbol=symbol,
                        action='SELL',
                        quantity=qty,
                        price=current_price,
                        amount=qty * current_price,
                        reason=reason or 'Fermeture position',
                        portfolio_value=account['portfolio_value'] if account else None
                    )
                except Exception as db_error:
                    logger.warning(f"⚠️  Log BDD échoué: {db_error}")
            
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
    
    def get_orders(self, status='open', limit=50):
        """Obtenir les ordres"""
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            
            status_map = {
                'open': QueryOrderStatus.OPEN,
                'closed': QueryOrderStatus.CLOSED,
                'all': QueryOrderStatus.ALL
            }
            
            request = GetOrdersRequest(
                status=status_map.get(status, QueryOrderStatus.OPEN),
                limit=limit
            )
            
            orders = self.trading_client.get_orders(filter=request)
            
            return [{
                'id': order.id,
                'symbol': order.symbol,
                'qty': float(order.qty) if order.qty else 0,
                'side': order.side.value if hasattr(order.side, 'value') else str(order.side),
                'status': order.status.value if hasattr(order.status, 'value') else str(order.status),
                'order_type': order.order_type.value if hasattr(order.order_type, 'value') else str(order.order_type),
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else 0,
                'filled_at': str(order.filled_at) if order.filled_at else '',
                'created_at': str(order.created_at) if order.created_at else ''
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
    
    def log_current_positions(self):
        """Logger toutes les positions actuelles dans la BDD (SAFE)"""
        if not DB_AVAILABLE:
            logger.warning("⚠️  BDD non disponible pour log positions")
            return
        
        try:
            positions = self.get_positions()
            if positions:
                log_all_positions(positions)
                logger.info(f"✅ {len(positions)} positions loggées dans BDD")
        except Exception as e:
            logger.warning(f"⚠️  Erreur log_current_positions: {e}")
