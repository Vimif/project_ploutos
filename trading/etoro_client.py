# trading/etoro_client.py
"""Client pour l'API eToro avec logging JSON"""

import os
import json
import time
import requests
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

from trading.broker_interface import BrokerInterface
from core.utils import setup_logging

logger = setup_logging(__name__, 'etoro.log')

# Charger variables d'environnement
load_dotenv()

# Dossier logs trades
TRADES_LOG_DIR = 'logs/trades'
os.makedirs(TRADES_LOG_DIR, exist_ok=True)


def log_trade_to_json(symbol, action, quantity, price, amount, reason='', portfolio_value=None, order_id=None):
    """Logger un trade en JSON."""
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
            'order_id': order_id,
            'broker': 'etoro'
        }

        filename = f"{TRADES_LOG_DIR}/trades_{datetime.now().strftime('%Y-%m-%d')}.json"

        if os.path.exists(filename):
            with open(filename, 'r') as f:
                trades = json.load(f)
        else:
            trades = []

        trades.append(trade_data)

        with open(filename, 'w') as f:
            json.dump(trades, f, indent=2)

        logger.debug(f"Trade logged: {symbol} {action}")

    except Exception as e:
        logger.warning(f"Failed to log trade to JSON: {e}")


class EToroClient(BrokerInterface):
    """
    Client eToro pour le trading via l'API publique.

    API eToro:
    - Base URL: https://api.etoro.com
    - Auth: Ocp-Apim-Subscription-Key header + Login tokens
    - Trading: POST /EntryOrder/{System}, POST /ExitOrder/{System}
    - Positions: GET /Trade/{System}
    - Account: GET /Credit/{System}, GET /Equity/{System}

    System = "Demo" pour paper trading, "Real" pour live
    """

    BASE_URL = "https://api.etoro.com"

    def __init__(self, paper_trading: bool = True):
        """
        Initialiser le client eToro.

        Args:
            paper_trading: True pour Demo, False pour Real

        Env vars requises:
            ETORO_SUBSCRIPTION_KEY: Clé d'abonnement API (Ocp-Apim-Subscription-Key)
            ETORO_USERNAME: Nom d'utilisateur eToro
            ETORO_PASSWORD: Mot de passe eToro
            ETORO_API_KEY: Clé API développeur (optionnelle, certaines routes)
        """
        self.paper_trading = paper_trading
        self.system = "Demo" if paper_trading else "Real"

        # Récupérer les credentials
        self.subscription_key = os.getenv('ETORO_SUBSCRIPTION_KEY')
        self.username = os.getenv('ETORO_USERNAME')
        self.password = os.getenv('ETORO_PASSWORD')
        self.api_key = os.getenv('ETORO_API_KEY', '')

        if not self.subscription_key:
            raise ValueError(
                "ETORO_SUBSCRIPTION_KEY manquante dans .env. "
                "Obtenez-la sur https://api-portal.etoro.com/"
            )

        if not self.username or not self.password:
            raise ValueError(
                "ETORO_USERNAME et ETORO_PASSWORD requis dans .env"
            )

        # Session HTTP avec retry
        self.session = requests.Session()
        self.session.headers.update({
            'Ocp-Apim-Subscription-Key': self.subscription_key,
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        })

        # Tokens d'authentification
        self._token = None
        self._csrf_token = None

        # Cache instruments (InstrumentId <-> Symbol)
        self._instrument_cache = {}
        self._symbol_cache = {}

        # Login
        self._login()

        logger.info(f"Client eToro initialise ({'Demo' if paper_trading else 'LIVE'})")
        logger.info(f"Trades logges dans: {TRADES_LOG_DIR}/")

    # ========== AUTHENTIFICATION ==========

    def _login(self):
        """S'authentifier et obtenir les tokens de session."""
        try:
            payload = {
                'Username': self.username,
                'Password': self.password,
            }
            if self.api_key:
                payload['ApiKey'] = self.api_key

            resp = self._request('POST', '/Account/Login', json_data=payload, auth_required=False)

            if resp and resp.status_code == 200:
                data = resp.json() if resp.text else {}
                # Les tokens peuvent venir du body ou des headers
                self._token = (
                    data.get('Token')
                    or data.get('token')
                    or resp.headers.get('x-token')
                )
                self._csrf_token = (
                    data.get('CsrfToken')
                    or data.get('csrf_token')
                    or resp.headers.get('x-csrf-token')
                )

                if self._token:
                    self.session.headers.update({
                        'x-token': self._token,
                    })
                if self._csrf_token:
                    self.session.headers.update({
                        'x-csrf-token': self._csrf_token,
                    })

                logger.info("Authentification eToro reussie")
            else:
                status = resp.status_code if resp else 'no response'
                raise ConnectionError(f"Login eToro echoue (status: {status})")

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Impossible de se connecter a eToro: {e}")

    def _ensure_auth(self):
        """Re-login si le token a expiré."""
        if not self._token:
            self._login()

    # ========== HTTP HELPER ==========

    def _request(self, method: str, path: str, json_data=None, params=None,
                 auth_required=True, retries=3) -> Optional[requests.Response]:
        """
        Effectuer une requête HTTP vers l'API eToro avec retry.

        Args:
            method: GET, POST, PUT, DELETE
            path: Chemin API (ex: /Trade/Demo)
            json_data: Body JSON
            params: Query params
            auth_required: Si True, vérifie l'auth
            retries: Nombre de tentatives

        Returns:
            Response ou None
        """
        if auth_required:
            self._ensure_auth()

        url = f"{self.BASE_URL}{path}"

        for attempt in range(retries):
            try:
                resp = self.session.request(
                    method=method,
                    url=url,
                    json=json_data,
                    params=params,
                    timeout=30,
                )

                # Token expiré → re-login
                if resp.status_code == 401 and auth_required:
                    logger.warning("Token expire, re-authentification...")
                    self._login()
                    continue

                if resp.status_code == 429:
                    # Rate limit
                    wait = 2 ** attempt
                    logger.warning(f"Rate limit eToro, attente {wait}s...")
                    time.sleep(wait)
                    continue

                return resp

            except requests.exceptions.RequestException as e:
                logger.error(f"Erreur HTTP ({method} {path}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)

        return None

    # ========== INSTRUMENTS ==========

    def _get_instrument_id(self, symbol: str) -> Optional[int]:
        """
        Convertir un symbole ticker en InstrumentId eToro.
        Utilise un cache pour éviter les appels répétés.
        """
        if symbol in self._instrument_cache:
            return self._instrument_cache[symbol]

        try:
            resp = self._request('GET', f'/Metadata/{self.system}')
            if resp and resp.status_code == 200:
                instruments = resp.json()
                if isinstance(instruments, list):
                    for inst in instruments:
                        sym = inst.get('SymbolFull', '') or inst.get('Symbol', '')
                        inst_id = inst.get('InstrumentID') or inst.get('InstrumentId')
                        if sym and inst_id:
                            self._instrument_cache[sym.upper()] = inst_id
                            self._symbol_cache[inst_id] = sym.upper()

            # Essayer aussi l'endpoint Metadata/V1/Instruments
            if symbol not in self._instrument_cache:
                resp = self._request(
                    'GET', '/Metadata/V1/Instruments',
                    params={'InstrumentIds': ''},
                    auth_required=True
                )
                if resp and resp.status_code == 200:
                    instruments = resp.json()
                    if isinstance(instruments, list):
                        for inst in instruments:
                            sym = inst.get('SymbolFull', '') or inst.get('Symbol', '')
                            inst_id = inst.get('InstrumentID') or inst.get('InstrumentId')
                            if sym and inst_id:
                                self._instrument_cache[sym.upper()] = inst_id
                                self._symbol_cache[inst_id] = sym.upper()

        except Exception as e:
            logger.error(f"Erreur chargement instruments: {e}")

        return self._instrument_cache.get(symbol.upper())

    def _get_symbol(self, instrument_id: int) -> str:
        """Convertir un InstrumentId en symbole ticker."""
        if instrument_id in self._symbol_cache:
            return self._symbol_cache[instrument_id]
        # Forcer le chargement du cache
        self._get_instrument_id('__LOAD__')
        return self._symbol_cache.get(instrument_id, f'ID:{instrument_id}')

    # ========== COMPTE ==========

    def get_account(self) -> Optional[dict]:
        """Obtenir les infos du compte eToro."""
        try:
            # Récupérer credit (cash disponible)
            credit_resp = self._request('GET', f'/Credit/{self.system}')
            equity_resp = self._request('GET', f'/Equity/{self.system}')

            cash = 0.0
            equity = 0.0
            portfolio_value = 0.0

            if credit_resp and credit_resp.status_code == 200:
                credit_data = credit_resp.json()
                cash = float(
                    credit_data.get('Credit', 0)
                    or credit_data.get('credit', 0)
                    or credit_data.get('AvailableBalance', 0)
                    or credit_data
                    if isinstance(credit_data, (int, float)) else 0
                )

            if equity_resp and equity_resp.status_code == 200:
                equity_data = equity_resp.json()
                equity = float(
                    equity_data.get('Equity', 0)
                    or equity_data.get('equity', 0)
                    or equity_data
                    if isinstance(equity_data, (int, float)) else 0
                )

            # Portfolio value = equity (includes positions)
            portfolio_value = equity if equity > 0 else cash

            return {
                'cash': cash,
                'portfolio_value': portfolio_value,
                'buying_power': cash,  # eToro: buying power ~ cash disponible
                'equity': equity,
                'last_equity': equity,
                'daytrade_count': 0,  # eToro n'a pas de PDT rule
                'pattern_day_trader': False,
            }

        except Exception as e:
            logger.error(f"Erreur recuperation compte: {e}")
            return None

    # ========== POSITIONS ==========

    def get_positions(self) -> list:
        """Obtenir toutes les positions ouvertes."""
        try:
            resp = self._request('GET', f'/Trade/{self.system}')

            if not resp or resp.status_code != 200:
                return []

            trades_data = resp.json()
            if not isinstance(trades_data, list):
                trades_data = trades_data.get('Positions', []) if isinstance(trades_data, dict) else []

            positions = []
            # Regrouper par instrument
            grouped = {}
            for trade in trades_data:
                inst_id = trade.get('InstrumentID') or trade.get('InstrumentId')
                if inst_id not in grouped:
                    grouped[inst_id] = []
                grouped[inst_id].append(trade)

            for inst_id, trades in grouped.items():
                symbol = self._get_symbol(inst_id)

                total_qty = 0.0
                total_invested = 0.0
                total_current_value = 0.0

                for trade in trades:
                    amount = float(trade.get('Amount', 0) or trade.get('amount', 0))
                    open_rate = float(trade.get('OpenRate', 0) or trade.get('openRate', 0))
                    current_rate = float(trade.get('CurrentRate', 0) or trade.get('currentRate', open_rate))
                    leverage = float(trade.get('Leverage', 1) or trade.get('leverage', 1))
                    is_buy = trade.get('IsBuy', True) if 'IsBuy' in trade else trade.get('isBuy', True)

                    # eToro utilise Amount (en $), pas qty d'actions directement
                    if open_rate > 0:
                        qty = amount / open_rate
                    else:
                        qty = 0

                    total_qty += qty
                    total_invested += amount
                    if current_rate > 0 and open_rate > 0:
                        total_current_value += amount * (current_rate / open_rate)
                    else:
                        total_current_value += amount

                unrealized_pl = total_current_value - total_invested
                unrealized_plpc = unrealized_pl / total_invested if total_invested > 0 else 0
                avg_entry = total_invested / total_qty if total_qty > 0 else 0
                current_price = total_current_value / total_qty if total_qty > 0 else 0

                # Trouver la date d'achat la plus ancienne pour cet instrument
                open_dates = []
                for t in trades:
                    od = t.get('OpenDateTime') or t.get('openDateTime')
                    if od:
                        open_dates.append(od)
                purchase_date = min(open_dates) if open_dates else None

                positions.append({
                    'symbol': symbol,
                    'qty': round(total_qty, 6),
                    'market_value': round(total_current_value, 2),
                    'cost_basis': round(total_invested, 2),
                    'unrealized_pl': round(unrealized_pl, 2),
                    'unrealized_plpc': round(unrealized_plpc, 6),
                    'current_price': round(current_price, 2),
                    'avg_entry_price': round(avg_entry, 2),
                    'purchase_date': purchase_date,
                    # Métadonnées eToro-spécifiques
                    '_etoro_instrument_id': inst_id,
                    '_etoro_position_ids': [
                        t.get('PositionID') or t.get('positionId') for t in trades
                    ],
                })

            return positions

        except Exception as e:
            logger.error(f"Erreur recuperation positions: {e}")
            return []

    def get_position(self, symbol: str) -> Optional[dict]:
        """Obtenir une position spécifique."""
        positions = self.get_positions()
        for pos in positions:
            if pos['symbol'].upper() == symbol.upper():
                return pos
        return None

    # ========== PRIX ==========

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Obtenir le prix actuel d'un ticker.
        Utilise les endpoints de rates/metadata eToro.
        """
        try:
            inst_id = self._get_instrument_id(symbol)
            if not inst_id:
                logger.warning(f"Instrument non trouve pour {symbol}")
                return None

            # Essayer l'endpoint rates
            resp = self._request(
                'GET', f'/Rates/{self.system}',
                params={'InstrumentIds': str(inst_id)}
            )

            if resp and resp.status_code == 200:
                rates = resp.json()
                if isinstance(rates, list) and rates:
                    rate = rates[0]
                    ask = float(rate.get('Ask', 0) or rate.get('ask', 0))
                    bid = float(rate.get('Bid', 0) or rate.get('bid', 0))
                    if ask > 0 and bid > 0:
                        return (ask + bid) / 2
                    last = float(rate.get('LastExecution', 0) or rate.get('lastExecution', 0))
                    if last > 0:
                        return last

            # Fallback: utiliser le prix depuis les positions
            pos = self.get_position(symbol)
            if pos and pos['current_price'] > 0:
                return pos['current_price']

            return None

        except Exception as e:
            logger.error(f"Erreur prix pour {symbol}: {e}")
            return None

    # ========== ORDRES ==========

    def place_market_order(self, symbol: str, qty: int, side: str = 'buy',
                           reason: str = '') -> Optional[dict]:
        """
        Passer un ordre au marché via eToro.

        eToro utilise des montants en $, pas des quantités d'actions.
        On convertit qty * current_price pour obtenir le montant.
        """
        try:
            inst_id = self._get_instrument_id(symbol)
            if not inst_id:
                logger.error(f"Instrument non trouve pour {symbol}")
                return None

            current_price = self.get_current_price(symbol)
            if not current_price or current_price <= 0:
                logger.error(f"Prix indisponible pour {symbol}")
                return None

            amount = round(qty * current_price, 2)
            is_buy = side.lower() == 'buy'

            # Construire l'ordre eToro (EntryOrder)
            order_payload = {
                'InstrumentID': inst_id,
                'OrderType': 'MKT',        # Market order
                'ExecutionType': 'IOC',     # Immediate-or-Cancel
                'IsBuy': is_buy,
                'Amount': amount,
                'Leverage': 1,              # Pas de levier (achat réel d'actions)
            }

            resp = self._request('POST', f'/EntryOrder/{self.system}', json_data=order_payload)

            if not resp:
                logger.error(f"Pas de reponse pour ordre {side} {symbol}")
                return None

            if resp.status_code in (200, 201):
                data = resp.json() if resp.text else {}

                order_id = (
                    data.get('OrderId')
                    or data.get('orderId')
                    or data.get('PositionID')
                    or data.get('positionId')
                    or ''
                )

                filled_price = float(
                    data.get('ExecutionRate', 0)
                    or data.get('executionRate', 0)
                    or current_price
                )

                order_dict = {
                    'id': str(order_id),
                    'symbol': symbol,
                    'qty': qty,
                    'side': side.lower(),
                    'status': 'filled',
                    'filled_avg_price': filled_price,
                }

                logger.info(f"Ordre {side.upper()} place: {symbol} x{qty} (~${amount})")

                # Logger en JSON
                account = self.get_account()
                log_trade_to_json(
                    symbol=symbol,
                    action=side.upper(),
                    quantity=qty,
                    price=filled_price,
                    amount=qty * filled_price,
                    reason=reason,
                    portfolio_value=account['portfolio_value'] if account else None,
                    order_id=str(order_id),
                )

                return order_dict

            else:
                error_text = resp.text[:200] if resp.text else 'Unknown error'
                logger.error(f"Erreur ordre {side} {symbol}: HTTP {resp.status_code} - {error_text}")
                return None

        except Exception as e:
            logger.error(f"Erreur ordre {side} pour {symbol}: {e}")
            return None

    def place_limit_order(self, symbol: str, qty: int, limit_price: float,
                          side: str = 'buy') -> Optional[dict]:
        """Passer un ordre limite via eToro."""
        try:
            inst_id = self._get_instrument_id(symbol)
            if not inst_id:
                logger.error(f"Instrument non trouve pour {symbol}")
                return None

            amount = round(qty * limit_price, 2)
            is_buy = side.lower() == 'buy'

            order_payload = {
                'InstrumentID': inst_id,
                'OrderType': 'LMT',        # Limit order
                'ExecutionType': 'GTC',     # Good-Till-Cancelled
                'IsBuy': is_buy,
                'Amount': amount,
                'Leverage': 1,
                'Rate': limit_price,        # Prix limite
            }

            resp = self._request('POST', f'/EntryOrder/{self.system}', json_data=order_payload)

            if resp and resp.status_code in (200, 201):
                data = resp.json() if resp.text else {}

                order_id = (
                    data.get('OrderId')
                    or data.get('orderId')
                    or ''
                )

                logger.info(f"Ordre LIMIT {side.upper()} place: {symbol} x{qty} @ ${limit_price}")

                return {
                    'id': str(order_id),
                    'symbol': symbol,
                    'qty': qty,
                    'limit_price': limit_price,
                    'status': 'pending',
                }

            else:
                error_text = resp.text[:200] if resp and resp.text else 'Unknown error'
                status = resp.status_code if resp else 'N/A'
                logger.error(f"Erreur ordre limite {symbol}: HTTP {status} - {error_text}")
                return None

        except Exception as e:
            logger.error(f"Erreur ordre limite pour {symbol}: {e}")
            return None

    # ========== FERMETURE POSITIONS ==========

    def close_position(self, symbol: str, reason: str = '') -> bool:
        """
        Fermer complètement une position eToro.
        Crée un ExitOrder pour chaque position ouverte sur l'instrument.
        """
        try:
            position = self.get_position(symbol)
            if not position:
                logger.warning(f"{symbol}: Pas de position a fermer")
                return False

            qty = position['qty']
            current_price = position['current_price']
            position_ids = position.get('_etoro_position_ids', [])

            if not position_ids:
                logger.error(f"{symbol}: Pas de PositionID pour fermer")
                return False

            # Annuler les ordres en attente
            self.cancel_orders_for_symbol(symbol)
            time.sleep(0.5)

            # Fermer chaque sous-position
            all_closed = True
            for pos_id in position_ids:
                if not pos_id:
                    continue

                exit_payload = {
                    'PositionID': pos_id,
                    'ExecutionType': 'IOC',
                }

                resp = self._request('POST', f'/ExitOrder/{self.system}', json_data=exit_payload)

                if resp and resp.status_code in (200, 201):
                    logger.info(f"Position fermee: {symbol} (ID: {pos_id})")
                else:
                    error_text = resp.text[:200] if resp and resp.text else 'Unknown'
                    status = resp.status_code if resp else 'N/A'
                    logger.error(f"Echec fermeture {symbol} (ID: {pos_id}): HTTP {status} - {error_text}")
                    all_closed = False

            if all_closed:
                # Logger le SELL
                account = self.get_account()
                log_trade_to_json(
                    symbol=symbol,
                    action='SELL',
                    quantity=qty,
                    price=current_price,
                    amount=qty * current_price,
                    reason=reason or 'Fermeture position',
                    portfolio_value=account['portfolio_value'] if account else None,
                )
                logger.info(f"Position fermee: {symbol}")
                return True

            return False

        except Exception as e:
            logger.error(f"Erreur fermeture position {symbol}: {e}")
            return False

    def close_all_positions(self) -> bool:
        """Fermer toutes les positions."""
        try:
            positions = self.get_positions()
            if not positions:
                logger.info("Aucune position a fermer")
                return True

            success = True
            for pos in positions:
                if not self.close_position(pos['symbol'], reason='Fermeture globale'):
                    success = False

            if success:
                logger.info("Toutes les positions fermees")
            else:
                logger.warning("Certaines positions n'ont pas pu etre fermees")

            return success

        except Exception as e:
            logger.error(f"Erreur fermeture globale: {e}")
            return False

    # ========== GESTION ORDRES ==========

    def get_orders(self, status: str = 'open', limit: int = 50) -> list:
        """Obtenir les ordres en attente (entry orders)."""
        try:
            resp = self._request('GET', f'/EntryOrder/{self.system}')

            if not resp or resp.status_code != 200:
                return []

            orders_data = resp.json()
            if not isinstance(orders_data, list):
                orders_data = orders_data.get('Orders', []) if isinstance(orders_data, dict) else []

            orders = []
            for order in orders_data[:limit]:
                inst_id = order.get('InstrumentID') or order.get('InstrumentId')
                symbol = self._get_symbol(inst_id) if inst_id else 'UNKNOWN'
                is_buy = order.get('IsBuy', True)
                order_type = order.get('OrderType', 'MKT')

                orders.append({
                    'id': str(order.get('OrderId') or order.get('orderId', '')),
                    'symbol': symbol,
                    'qty': 0,  # eToro utilise des montants, pas des quantités
                    'side': 'buy' if is_buy else 'sell',
                    'status': 'open',
                    'order_type': order_type,
                    'filled_avg_price': 0,
                    'filled_at': '',
                    'created_at': str(order.get('CreationTime', '') or order.get('creationTime', '')),
                    '_etoro_instrument_id': inst_id,
                })

            return orders

        except Exception as e:
            logger.error(f"Erreur recuperation ordres: {e}")
            return []

    def cancel_order(self, order_id: str) -> bool:
        """Annuler un ordre en attente."""
        try:
            resp = self._request('DELETE', f'/EntryOrder/{self.system}/{order_id}')

            if resp and resp.status_code in (200, 204):
                logger.info(f"Ordre annule: {order_id}")
                return True
            else:
                error_text = resp.text[:200] if resp and resp.text else 'Unknown'
                status = resp.status_code if resp else 'N/A'
                logger.error(f"Erreur annulation ordre {order_id}: HTTP {status} - {error_text}")
                return False

        except Exception as e:
            logger.error(f"Erreur annulation ordre {order_id}: {e}")
            return False

    def cancel_orders_for_symbol(self, symbol: str) -> int:
        """Annuler tous les ordres pour un symbole."""
        try:
            orders = self.get_orders(status='open')
            cancelled = 0

            for order in orders:
                if order['symbol'].upper() == symbol.upper():
                    if self.cancel_order(order['id']):
                        cancelled += 1

            if cancelled > 0:
                logger.info(f"{symbol}: {cancelled} ordre(s) annule(s)")

            return cancelled

        except Exception as e:
            logger.error(f"Erreur annulation ordres pour {symbol}: {e}")
            return 0

    # ========== TRADE HISTORY ==========

    def get_trade_history(self) -> list:
        """Obtenir l'historique des trades fermés."""
        try:
            resp = self._request('GET', f'/Trade/{self.system}/History')

            if not resp or resp.status_code != 200:
                return []

            data = resp.json()
            if not isinstance(data, list):
                data = data.get('Trades', []) if isinstance(data, dict) else []

            return data

        except Exception as e:
            logger.error(f"Erreur historique trades: {e}")
            return []

    # ========== MODIFY POSITION ==========

    def modify_position(self, position_id: str, stop_loss_rate: float = None,
                        take_profit_rate: float = None) -> bool:
        """
        Modifier le SL/TP d'une position.

        Args:
            position_id: ID de la position eToro
            stop_loss_rate: Nouveau taux de stop loss
            take_profit_rate: Nouveau taux de take profit
        """
        try:
            payload = {}
            if stop_loss_rate is not None:
                payload['StopLossRate'] = stop_loss_rate
            if take_profit_rate is not None:
                payload['TakeProfitRate'] = take_profit_rate

            if not payload:
                return True

            resp = self._request('PUT', f'/Trade/{self.system}/{position_id}', json_data=payload)

            if resp and resp.status_code == 200:
                logger.info(f"Position {position_id} modifiee (SL={stop_loss_rate}, TP={take_profit_rate})")
                return True

            return False

        except Exception as e:
            logger.error(f"Erreur modification position {position_id}: {e}")
            return False

    # ========== FEES ==========

    def get_fees(self) -> Optional[dict]:
        """Obtenir les frais de trading."""
        try:
            resp = self._request('GET', f'/Fees/{self.system}')
            if resp and resp.status_code == 200:
                return resp.json()
            return None
        except Exception as e:
            logger.error(f"Erreur recuperation frais: {e}")
            return None

    # ========== LOGGING ==========

    def log_current_positions(self):
        """Logger les positions actuelles."""
        try:
            positions = self.get_positions()
            if positions:
                filename = f"{TRADES_LOG_DIR}/positions_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.json"
                with open(filename, 'w') as f:
                    json.dump(positions, f, indent=2)
                logger.info(f"{len(positions)} positions loggees: {filename}")
        except Exception as e:
            logger.warning(f"Echec log positions: {e}")

    # ========== LOGOUT ==========

    def logout(self):
        """Se déconnecter de l'API eToro."""
        try:
            self._request('DELETE', '/Account/Login')
            self._token = None
            self._csrf_token = None
            logger.info("Deconnexion eToro")
        except Exception:
            pass

    def __del__(self):
        """Cleanup."""
        try:
            self.logout()
        except Exception:
            pass
