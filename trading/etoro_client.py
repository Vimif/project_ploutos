"""Client eToro aligned with the current public API."""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime
from typing import Any, Optional

import requests
from dotenv import load_dotenv

from core.utils import setup_logging
from trading.broker_interface import BrokerInterface

logger = setup_logging(__name__, "etoro.log")

load_dotenv()

TRADES_LOG_DIR = "logs/trades"
os.makedirs(TRADES_LOG_DIR, exist_ok=True)


def log_trade_to_json(
    symbol,
    action,
    quantity,
    price,
    amount,
    reason="",
    portfolio_value=None,
    order_id=None,
):
    """Persist a trade event as JSON for lightweight local audit trails."""

    try:
        trade_data = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": price,
            "amount": amount,
            "reason": reason,
            "portfolio_value": portfolio_value,
            "order_id": order_id,
            "broker": "etoro",
        }

        filename = f"{TRADES_LOG_DIR}/trades_{datetime.now().strftime('%Y-%m-%d')}.json"
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as handle:
                trades = json.load(handle)
        else:
            trades = []

        trades.append(trade_data)
        with open(filename, "w", encoding="utf-8") as handle:
            json.dump(trades, handle, indent=2)
    except Exception as exc:
        logger.warning("Failed to log trade to JSON: %s", exc)


class EToroClient(BrokerInterface):
    """Broker adapter for the current eToro Public API."""

    BASE_URL = "https://public-api.etoro.com/api/v1"

    def __init__(self, paper_trading: bool = True):
        self.paper_trading = paper_trading
        self.account_scope = "demo" if paper_trading else "real"
        self.execution_scope = f"/trading/execution/{self.account_scope}"
        self.info_scope = f"/trading/info/{self.account_scope}"

        self.public_api_key = (
            os.getenv("ETORO_PUBLIC_API_KEY") or os.getenv("ETORO_SUBSCRIPTION_KEY") or ""
        ).strip()
        self.user_key = (os.getenv("ETORO_USER_KEY") or os.getenv("ETORO_API_KEY") or "").strip()

        self.username = (os.getenv("ETORO_USERNAME") or "").strip()
        self.password = (os.getenv("ETORO_PASSWORD") or "").strip()

        if not self.public_api_key:
            raise ValueError(
                "ETORO_PUBLIC_API_KEY manquante. "
                "Configure ETORO_PUBLIC_API_KEY ou reuse ETORO_SUBSCRIPTION_KEY "
                "avec la cle publique actuelle du portail eToro."
            )
        if not self.user_key:
            raise ValueError(
                "ETORO_USER_KEY manquante. "
                "L'API publique eToro actuelle utilise x-api-key + x-user-key; "
                "le login username/password n'est plus supporte par ce client. "
                "Genere une user key dans le portail eToro puis renseigne "
                "ETORO_USER_KEY (ou ETORO_API_KEY en alias legacy)."
            )

        self.session = requests.Session()
        self.session.headers.update(
            {
                "x-api-key": self.public_api_key,
                "x-user-key": self.user_key,
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

        self._identity: dict[str, Any] = {}
        self._instrument_cache: dict[str, int] = {}
        self._symbol_cache: dict[int, str] = {}

        self._authenticate()
        logger.info(
            "Client eToro initialised (%s)",
            "Demo" if paper_trading else "Real",
        )

    def warmup_instruments(self, symbols: list[str]) -> None:
        for symbol in symbols:
            try:
                self._get_instrument_id(symbol)
            except Exception as exc:
                logger.warning("Warmup instrument failed for %s: %s", symbol, exc)

    def _authenticate(self) -> None:
        """Validate API keys against a documented portfolio endpoint."""

        response = self._request(
            "GET",
            f"{self.info_scope}/pnl",
            auth_required=False,
            retries=1,
        )
        if response is None:
            raise ConnectionError("Authentication check to eToro returned no response")
        if response.status_code != 200:
            raise ConnectionError(
                "eToro authentication failed "
                f"(HTTP {response.status_code}): {self._extract_error_message(response)}"
            )

        payload = self._safe_json(response)
        if isinstance(payload, dict):
            self._identity = payload.get("clientPortfolio") or payload
        else:
            self._identity = {"raw": payload}

    def _ensure_auth(self) -> None:
        if not self._identity:
            self._authenticate()

    def _request(
        self,
        method: str,
        path: str,
        json_data=None,
        params=None,
        auth_required: bool = True,
        retries: int = 3,
    ) -> Optional[requests.Response]:
        if auth_required:
            self._ensure_auth()

        url = f"{self.BASE_URL}{path}"
        retried_auth = False

        for attempt in range(max(retries, 1)):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    json=json_data,
                    params=params,
                    timeout=30,
                    headers={"x-request-id": str(uuid.uuid4())},
                )

                if response.status_code == 401 and auth_required and not retried_auth:
                    logger.warning("eToro returned 401, refreshing authentication context")
                    self._identity = {}
                    self._authenticate()
                    retried_auth = True
                    continue

                if response.status_code == 429 and attempt < retries - 1:
                    wait_seconds = 2**attempt or 1
                    logger.warning("eToro rate limit hit, waiting %ss", wait_seconds)
                    time.sleep(wait_seconds)
                    continue

                return response
            except requests.exceptions.RequestException as exc:
                logger.error("HTTP error during %s %s: %s", method, path, exc)
                if attempt < retries - 1:
                    time.sleep(2**attempt or 1)

        return None

    def _safe_json(self, response: requests.Response) -> Any:
        if not getattr(response, "text", ""):
            return {}
        try:
            return response.json()
        except Exception:
            return {}

    def _extract_error_message(self, response: Optional[requests.Response]) -> str:
        if response is None:
            return "no response"

        payload = self._safe_json(response)
        if isinstance(payload, dict):
            for key in ("message", "error", "details", "title", "description"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            errors = payload.get("errors")
            if isinstance(errors, list) and errors:
                first = errors[0]
                if isinstance(first, dict):
                    for key in ("message", "error", "details"):
                        value = first.get(key)
                        if isinstance(value, str) and value.strip():
                            return value.strip()
                if isinstance(first, str) and first.strip():
                    return first.strip()

        text = getattr(response, "text", "") or ""
        return text[:200].strip() or "unknown error"

    def _get_portfolio_snapshot(self) -> dict[str, Any]:
        response = self._request("GET", f"{self.info_scope}/pnl")
        if response is None or response.status_code != 200:
            raise ConnectionError(
                "Unable to retrieve eToro portfolio snapshot: "
                f"{self._extract_error_message(response)}"
            )

        payload = self._safe_json(response)
        if isinstance(payload, dict):
            portfolio = payload.get("clientPortfolio")
            if isinstance(portfolio, dict):
                return portfolio
        return {}

    def _iter_instrument_candidates(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if not isinstance(payload, dict):
            return []

        if any(
            key in payload
            for key in ("instrumentId", "InstrumentId", "instrumentID", "InstrumentID")
        ):
            return [payload]

        for key in ("items", "results", "searchResults", "instruments", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return []

    def _normalize_symbol(self, instrument: dict[str, Any]) -> str:
        for key in (
            "internalSymbolFull",
            "InternalSymbolFull",
            "symbolFull",
            "SymbolFull",
            "symbol",
            "Symbol",
        ):
            value = instrument.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().upper()
        return ""

    def _normalize_instrument_id(self, instrument: dict[str, Any]) -> Optional[int]:
        for key in ("instrumentId", "InstrumentId", "instrumentID", "InstrumentID"):
            value = instrument.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    def _get_instrument_id(self, symbol: str) -> Optional[int]:
        normalized_symbol = symbol.upper()
        if normalized_symbol in self._instrument_cache:
            return self._instrument_cache[normalized_symbol]

        response = self._request(
            "GET",
            "/market-data/search",
            params={"internalSymbolFull": normalized_symbol},
        )
        if response is None or response.status_code != 200:
            return None

        payload = self._safe_json(response)
        for instrument in self._iter_instrument_candidates(payload):
            candidate_symbol = self._normalize_symbol(instrument)
            instrument_id = self._normalize_instrument_id(instrument)
            if not candidate_symbol or instrument_id is None:
                continue
            self._instrument_cache[candidate_symbol] = instrument_id
            self._symbol_cache[instrument_id] = candidate_symbol
            if candidate_symbol == normalized_symbol:
                return instrument_id

        return self._instrument_cache.get(normalized_symbol)

    def _get_symbol(self, instrument_id: int) -> str:
        return self._symbol_cache.get(int(instrument_id), f"ID:{instrument_id}")

    def _normalize_positions(self, positions_payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[int, list[dict[str, Any]]] = {}
        for position in positions_payload:
            try:
                instrument_id = int(position.get("instrumentId"))
            except (TypeError, ValueError):
                continue
            grouped.setdefault(instrument_id, []).append(position)

        positions: list[dict[str, Any]] = []
        for instrument_id, entries in grouped.items():
            symbol = self._get_symbol(instrument_id)
            total_qty = 0.0
            total_cost_basis = 0.0
            total_market_value = 0.0
            total_pnl = 0.0
            position_ids: list[str] = []
            entry_dates: list[str] = []
            weighted_entry_price = 0.0

            for entry in entries:
                units = float(entry.get("units") or entry.get("initialUnits") or 0.0)
                if units <= 0:
                    open_rate = float(entry.get("openRate") or 0.0)
                    amount = float(
                        entry.get("initialAmountInDollars") or entry.get("amount") or 0.0
                    )
                    units = amount / open_rate if open_rate > 0 else 0.0

                cost_basis = float(
                    entry.get("initialAmountInDollars") or entry.get("amount") or 0.0
                )
                pnl = float(entry.get("pnL") or entry.get("pnl") or 0.0)
                close_rate = float(entry.get("closeRate") or 0.0)
                open_rate = float(entry.get("openRate") or 0.0)

                market_value = cost_basis + pnl
                if market_value <= 0 and close_rate > 0 and units > 0:
                    market_value = units * close_rate
                if cost_basis <= 0 and open_rate > 0 and units > 0:
                    cost_basis = units * open_rate

                total_qty += units
                total_cost_basis += cost_basis
                total_market_value += market_value
                total_pnl += pnl
                weighted_entry_price += open_rate * units

                position_id = entry.get("positionId")
                if position_id is not None:
                    position_ids.append(str(position_id))
                open_date = entry.get("openDateTime")
                if isinstance(open_date, str) and open_date:
                    entry_dates.append(open_date)

            avg_entry_price = weighted_entry_price / total_qty if total_qty > 0 else 0.0
            current_price = total_market_value / total_qty if total_qty > 0 else 0.0
            unrealized_plpc = total_pnl / total_cost_basis if total_cost_basis > 0 else 0.0

            positions.append(
                {
                    "symbol": symbol,
                    "qty": round(total_qty, 6),
                    "market_value": round(total_market_value, 2),
                    "cost_basis": round(total_cost_basis, 2),
                    "unrealized_pl": round(total_pnl, 2),
                    "unrealized_plpc": round(unrealized_plpc, 6),
                    "current_price": round(current_price, 6),
                    "avg_entry_price": round(avg_entry_price, 6),
                    "purchase_date": min(entry_dates) if entry_dates else None,
                    "_etoro_instrument_id": instrument_id,
                    "_etoro_position_ids": position_ids,
                }
            )

        return positions

    def _status_from_id(self, status_id: Any) -> str:
        mapping = {
            1: "open",
            2: "filled",
            3: "cancelled",
            4: "rejected",
            5: "expired",
        }
        try:
            return mapping.get(int(status_id), "open")
        except (TypeError, ValueError):
            return "open"

    def _normalize_orders(
        self,
        orders_payload: list[dict[str, Any]],
        *,
        default_status: str,
        cancel_path_template: str,
    ) -> list[dict[str, Any]]:
        orders: list[dict[str, Any]] = []
        for order in orders_payload:
            try:
                instrument_id = int(order.get("instrumentId"))
            except (TypeError, ValueError):
                instrument_id = 0

            status = self._status_from_id(order.get("statusId")) or default_status
            order_id = order.get("orderId")
            orders.append(
                {
                    "id": str(order_id or ""),
                    "symbol": self._get_symbol(instrument_id) if instrument_id else "UNKNOWN",
                    "qty": float(order.get("units") or order.get("amountInUnits") or 0.0),
                    "side": "buy" if bool(order.get("isBuy", True)) else "sell",
                    "status": status,
                    "order_type": str(order.get("orderType", "")),
                    "filled_avg_price": float(order.get("rate") or 0.0),
                    "filled_at": "",
                    "created_at": str(order.get("openDateTime") or ""),
                    "_cancel_path": cancel_path_template.format(order_id=order_id),
                    "_etoro_instrument_id": instrument_id,
                }
            )
        return orders

    def get_account(self) -> Optional[dict]:
        try:
            snapshot = self._get_portfolio_snapshot()
            positions = self._normalize_positions(snapshot.get("positions") or [])
            cash = float(snapshot.get("credit") or 0.0)
            position_value = sum(float(position["market_value"]) for position in positions)
            equity = cash + position_value
            return {
                "cash": cash,
                "portfolio_value": equity,
                "buying_power": cash,
                "equity": equity,
                "last_equity": equity,
                "daytrade_count": 0,
                "pattern_day_trader": False,
            }
        except Exception as exc:
            logger.error("Erreur recuperation compte: %s", exc)
            return None

    def get_positions(self) -> list:
        try:
            snapshot = self._get_portfolio_snapshot()
            return self._normalize_positions(snapshot.get("positions") or [])
        except Exception as exc:
            logger.error("Erreur recuperation positions: %s", exc)
            return []

    def get_position(self, symbol: str) -> Optional[dict]:
        normalized_symbol = symbol.upper()
        for position in self.get_positions():
            if position["symbol"].upper() == normalized_symbol:
                return position
        return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        instrument_id = self._get_instrument_id(symbol)
        if instrument_id is None:
            return None

        response = self._request(
            "GET",
            "/market-data/instruments/rates",
            params={"instrumentIds": str(instrument_id)},
        )
        if response is None or response.status_code != 200:
            return None

        payload = self._safe_json(response)
        rates = payload.get("rates") if isinstance(payload, dict) else payload
        if not isinstance(rates, list) or not rates:
            return None

        rate = rates[0]
        ask = float(rate.get("ask") or 0.0)
        bid = float(rate.get("bid") or 0.0)
        if ask > 0 and bid > 0:
            return (ask + bid) / 2

        last_execution = float(rate.get("lastExecution") or 0.0)
        return last_execution if last_execution > 0 else None

    def place_market_order(
        self,
        symbol: str,
        qty: int,
        side: str = "buy",
        reason: str = "",
    ) -> Optional[dict]:
        normalized_side = side.lower()
        if normalized_side == "sell":
            success = self.close_position(symbol, reason=reason)
            return (
                {
                    "id": "",
                    "symbol": symbol.upper(),
                    "qty": float(qty),
                    "side": "sell",
                    "status": "filled" if success else "rejected",
                    "filled_avg_price": float(self.get_current_price(symbol) or 0.0),
                }
                if success
                else None
            )

        instrument_id = self._get_instrument_id(symbol)
        current_price = self.get_current_price(symbol)
        if instrument_id is None or current_price is None or current_price <= 0:
            logger.error("Prix ou instrument indisponible pour %s", symbol)
            return None

        amount = round(float(qty) * float(current_price), 2)
        payload = {
            "InstrumentId": instrument_id,
            "Amount": amount,
            "Leverage": 1,
            "IsBuy": True,
        }
        response = self._request(
            "POST",
            f"{self.execution_scope}/market-open-orders/by-amount",
            json_data=payload,
        )
        if response is None or response.status_code not in {200, 201, 202}:
            logger.error(
                "Erreur ordre BUY %s: HTTP %s - %s",
                symbol,
                response.status_code if response else "N/A",
                self._extract_error_message(response),
            )
            return None

        data = self._safe_json(response)
        order_id = (
            data.get("orderId")
            or data.get("positionId")
            or data.get("token")
            or data.get("id")
            or ""
            if isinstance(data, dict)
            else ""
        )
        account = self.get_account()
        log_trade_to_json(
            symbol=symbol.upper(),
            action="BUY",
            quantity=float(qty),
            price=float(current_price),
            amount=amount,
            reason=reason,
            portfolio_value=account["portfolio_value"] if account else None,
            order_id=str(order_id),
        )
        return {
            "id": str(order_id),
            "symbol": symbol.upper(),
            "qty": float(qty),
            "side": "buy",
            "status": "submitted",
            "filled_avg_price": float(current_price),
        }

    def place_limit_order(
        self,
        symbol: str,
        qty: int,
        limit_price: float,
        side: str = "buy",
    ) -> Optional[dict]:
        instrument_id = self._get_instrument_id(symbol)
        if instrument_id is None:
            return None

        payload = {
            "InstrumentId": instrument_id,
            "Amount": round(float(qty) * float(limit_price), 2),
            "Rate": float(limit_price),
            "Leverage": 1,
            "IsBuy": side.lower() == "buy",
        }
        response = self._request(
            "POST",
            f"{self.execution_scope}/limit-orders",
            json_data=payload,
        )
        if response is None or response.status_code not in {200, 201, 202}:
            return None

        data = self._safe_json(response)
        order_id = data.get("orderId") or data.get("id") or "" if isinstance(data, dict) else ""
        return {
            "id": str(order_id),
            "symbol": symbol.upper(),
            "qty": float(qty),
            "limit_price": float(limit_price),
            "status": "pending",
        }

    def close_position(self, symbol: str, reason: str = "") -> bool:
        position = self.get_position(symbol)
        if not position:
            logger.warning("%s: no position to close", symbol)
            return False

        current_price = float(position.get("current_price", 0.0))
        qty = float(position.get("qty", 0.0))
        all_closed = True

        for position_id in position.get("_etoro_position_ids", []):
            response = self._request(
                "POST",
                f"{self.execution_scope}/market-close-orders/positions/{position_id}",
                json_data={"UnitsToDeduct": None},
            )
            if response is None or response.status_code not in {200, 201, 202}:
                logger.error(
                    "Erreur fermeture %s (%s): HTTP %s - %s",
                    symbol,
                    position_id,
                    response.status_code if response else "N/A",
                    self._extract_error_message(response),
                )
                all_closed = False

        if all_closed:
            account = self.get_account()
            log_trade_to_json(
                symbol=symbol.upper(),
                action="SELL",
                quantity=qty,
                price=current_price,
                amount=qty * current_price,
                reason=reason or "close_position",
                portfolio_value=account["portfolio_value"] if account else None,
            )
        return all_closed

    def close_all_positions(self) -> bool:
        success = True
        for position in self.get_positions():
            if not self.close_position(position["symbol"], reason="close_all_positions"):
                success = False
        return success

    def get_orders(self, status: str = "open", limit: int = 50) -> list:
        try:
            snapshot = self._get_portfolio_snapshot()
            all_orders = []
            all_orders.extend(
                self._normalize_orders(
                    snapshot.get("ordersForOpen") or [],
                    default_status="open",
                    cancel_path_template=f"{self.execution_scope}/market-open-orders/{{order_id}}",
                )
            )
            all_orders.extend(
                self._normalize_orders(
                    snapshot.get("ordersForClose") or [],
                    default_status="open",
                    cancel_path_template=f"{self.execution_scope}/market-close-orders/{{order_id}}",
                )
            )
            all_orders.extend(
                self._normalize_orders(
                    snapshot.get("orders") or [],
                    default_status="open",
                    cancel_path_template=f"{self.execution_scope}/market-open-orders/{{order_id}}",
                )
            )

            if status != "all":
                all_orders = [
                    order
                    for order in all_orders
                    if (status == "open" and order["status"] == "open")
                    or (status == "closed" and order["status"] != "open")
                ]
            return all_orders[:limit]
        except Exception as exc:
            logger.error("Erreur recuperation ordres: %s", exc)
            return []

    def cancel_order(self, order_id: str) -> bool:
        for order in self.get_orders(status="all"):
            if order.get("id") != str(order_id):
                continue
            cancel_path = order.get("_cancel_path")
            if not cancel_path:
                break
            response = self._request("DELETE", cancel_path)
            return response is not None and response.status_code in {200, 202, 204}
        return False

    def cancel_orders_for_symbol(self, symbol: str) -> int:
        cancelled = 0
        normalized_symbol = symbol.upper()
        for order in self.get_orders(status="open"):
            if order.get("symbol", "").upper() != normalized_symbol:
                continue
            if self.cancel_order(order.get("id", "")):
                cancelled += 1
        return cancelled

    def get_trade_history(self) -> list:
        try:
            snapshot = self._get_portfolio_snapshot()
            return list(snapshot.get("closedPositions") or [])
        except Exception as exc:
            logger.error("Erreur historique trades: %s", exc)
            return []

    def wait_for_order_fill(
        self,
        order_id: str,
        symbol: str | None = None,
        timeout: int = 30,
        poll_interval: float = 1.0,
        expect_position_closed: bool = False,
    ) -> bool:
        deadline = time.time() + timeout
        normalized_symbol = symbol.upper() if symbol else None

        while time.time() < deadline:
            if normalized_symbol:
                position = self.get_position(normalized_symbol)
                if expect_position_closed and position is None:
                    return True
                if not expect_position_closed and position is not None:
                    return True

            if order_id:
                for order in self.get_orders(status="all"):
                    if order.get("id") != str(order_id):
                        continue
                    if order.get("status") in {"filled", "closed", "completed"}:
                        return True
                    if order.get("status") in {"rejected", "cancelled", "expired"}:
                        return False

            time.sleep(poll_interval)
        return False

    def modify_position(
        self,
        position_id: str,
        stop_loss_rate: float = None,
        take_profit_rate: float = None,
    ) -> bool:
        payload = {}
        if stop_loss_rate is not None:
            payload["StopLossRate"] = float(stop_loss_rate)
        if take_profit_rate is not None:
            payload["TakeProfitRate"] = float(take_profit_rate)
        if not payload:
            return True

        response = self._request(
            "PATCH",
            f"{self.execution_scope}/positions/{position_id}",
            json_data=payload,
        )
        return response is not None and response.status_code in {200, 202, 204}

    def get_fees(self) -> Optional[dict]:
        response = self._request("GET", "/market-data/fees")
        if response is None or response.status_code != 200:
            return None
        payload = self._safe_json(response)
        return payload if isinstance(payload, dict) else None

    def log_current_positions(self):
        try:
            positions = self.get_positions()
            if not positions:
                return
            filename = (
                f"{TRADES_LOG_DIR}/positions_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.json"
            )
            with open(filename, "w", encoding="utf-8") as handle:
                json.dump(positions, handle, indent=2)
        except Exception as exc:
            logger.warning("Echec log positions: %s", exc)

    def logout(self):
        self._identity = {}

    def __del__(self):
        try:
            self.logout()
        except Exception:
            pass
