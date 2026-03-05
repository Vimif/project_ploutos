# trading/alpaca_client.py
"""Client pour l'API Alpaca avec logging JSON"""

import json
import os
import time
from datetime import datetime

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
from dotenv import load_dotenv

from core.utils import setup_logging
from trading.broker_interface import BrokerInterface

logger = setup_logging(__name__, 'alpaca.log')

# Charger variables d'environnement
load_dotenv()

# ★ CRÉER DOSSIER LOGS TRADES
TRADES_LOG_DIR = 'logs/trades'
os.makedirs(TRADES_LOG_DIR, exist_ok=True)

def log_trade_to_json(symbol, action, quantity, price, amount, reason='', portfolio_value=None, order_id=None):
    """
    Logger un trade en JSON au lieu de PostgreSQL
    
    Args:
        symbol: Ticker
        action: 'BUY' ou 'SELL'
        quantity: Nombre d'actions
        price: Prix unitaire
        amount: Montant total
        reason: Raison du trade
        portfolio_value: Valeur du portfolio
        order_id: ID de l'ordre Alpaca
    """
    try:
        trade_data = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'amount': amount,
            'reason': reason,
            'portfolio_value': portfolio_value,
            'order_id': order_id
        }

        # Nom fichier: trades_2025-12-07.json
        filename = f"{TRADES_LOG_DIR}/trades_{datetime.now().strftime('%Y-%m-%d')}.json"

        # Lire trades existants
        if os.path.exists(filename):
            with open(filename) as f:
                trades = json.load(f)
        else:
            trades = []

        # Ajouter nouveau trade
        trades.append(trade_data)

        # Sauvegarder
        with open(filename, 'w') as f:
            json.dump(trades, f, indent=2)

        logger.debug(f"✅ Trade loggé en JSON: {symbol} {action}")

    except Exception as e:
        logger.warning(f"⚠️  Échec log JSON: {e}")

class AlpacaClient(BrokerInterface):
    """Client Alpaca pour le trading (implémente BrokerInterface)"""

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
        logger.info(f"📝 Trades loggés dans: {TRADES_LOG_DIR}/")

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
                'avg_entry_price': float(pos.avg_entry_price),
                'purchase_date': getattr(pos, 'created_at', None)  # Alpaca Position lacks purchase_date
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
        except Exception as e:
            # Position not found is expected, but log other errors
            logger.debug(f"Could not get position for {symbol}: {e}")
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
        Annuler tous les ordres en cours pour un ticker
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

    def wait_for_order_fill(self, order_id, timeout=30):
        """
        ★ ATTENDRE QU'UN ORDRE SOIT EXÉCUTÉ
        
        Args:
            order_id: ID de l'ordre
            timeout: Timeout en secondes
        
        Returns:
            bool: True si exécuté, False si timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                order = self.trading_client.get_order_by_id(order_id)
                status = str(order.status).lower()

                if 'filled' in status or 'completed' in status:
                    logger.debug(f"  ✅ Ordre {order_id[:8]}... exécuté")
                    return True

                elif 'canceled' in status or 'expired' in status or 'rejected' in status:
                    logger.warning(f"  ⚠️  Ordre {order_id[:8]}... {status}")
                    return False

                # Attendre 0.5s avant de revérifier
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"  ❌ Erreur vérif ordre: {e}")
                return False

        logger.warning(f"  ⏱️  Timeout ordre {order_id[:8]}...")
        return False

    def place_market_order(self, symbol, qty, side='buy', reason=''):
        """
        Passer un ordre au marché avec logging JSON
        
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

            # ★ ATTENDRE EXÉCUTION
            if not self.wait_for_order_fill(order.id, timeout=30):
                logger.warning(f"⚠️  {symbol}: Ordre non exécuté dans les délais")
                return None

            # Récupérer ordre mis à jour
            order = self.trading_client.get_order_by_id(order.id)

            order_dict = {
                'id': order.id,
                'symbol': order.symbol,
                'qty': float(order.qty),
                'side': order.side,
                'status': order.status,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None
            }

            # ★ LOGGER EN JSON (SAFE)
            account = self.get_account()
            price = float(order.filled_avg_price) if order.filled_avg_price else self.get_current_price(symbol)

            if price and price > 0:
                log_trade_to_json(
                    symbol=symbol,
                    action=side.upper(),
                    quantity=qty,
                    price=price,
                    amount=qty * price,
                    reason=reason,
                    portfolio_value=account['portfolio_value'] if account else None,
                    order_id=order.id
                )

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
        Fermer complètement une position avec gestion wash trade
        ★ ATTEND QUE L'ORDRE SOIT EXÉCUTÉ
        
        Args:
            symbol: Ticker
            reason: Raison de la fermeture
        
        Returns:
            bool: Succès/échec
        """
        try:
            # 1. Vérifier que la position existe
            position = self.get_position(symbol)

            if not position:
                logger.warning(f"⚠️  {symbol}: Pas de position à fermer")
                return False

            qty = position['qty']
            current_price = position['current_price']

            # 2. Annuler tous les ordres en cours pour éviter wash trade
            self.cancel_orders_for_symbol(symbol)

            # 3. Attendre 1 seconde
            time.sleep(1)

            # 4. Fermer la position (crée un ordre SELL)
            response = self.trading_client.close_position(symbol)

            # Récupérer l'ID de l'ordre créé
            order_id = response.id if hasattr(response, 'id') else None

            if order_id:
                logger.info(f"✅ Ordre SELL créé: {symbol}")

                # ★ 5. ATTENDRE EXÉCUTION
                if not self.wait_for_order_fill(order_id, timeout=30):
                    logger.error(f"❌ {symbol}: Ordre SELL non exécuté")
                    return False

                logger.info(f"✅ Position fermée: {symbol}")
            else:
                # API a directement fermé (cas rare)
                logger.info(f"✅ Position fermée: {symbol}")

            # ★ 6. Logger le SELL
            account = self.get_account()
            log_trade_to_json(
                symbol=symbol,
                action='SELL',
                quantity=qty,
                price=current_price,
                amount=qty * current_price,
                reason=reason or 'Fermeture position',
                portfolio_value=account['portfolio_value'] if account else None
            )

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
            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest

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
        """Logger les positions (JSON)"""
        try:
            positions = self.get_positions()
            if positions:
                filename = f"{TRADES_LOG_DIR}/positions_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.json"
                with open(filename, 'w') as f:
                    json.dump(positions, f, indent=2)
                logger.info(f"✅ {len(positions)} positions loggées: {filename}")
        except Exception as e:
            logger.warning(f"⚠️  Échec log positions: {e}")
