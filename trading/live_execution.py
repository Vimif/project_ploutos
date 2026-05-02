"""Normalized live execution helpers for simulate, Alpaca, and eToro."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from core.transaction_costs import AdvancedTransactionModel
from trading.broker_factory import create_broker


@dataclass
class NormalizedOrderResult:
    """Normalized order result returned by the live execution layer."""

    success: bool
    symbol: str
    side: str
    qty: float
    requested_notional: float
    filled_price: float
    status: str
    order_id: str = ""
    reason: str = ""
    raw: Optional[dict] = None


@dataclass
class LiveExecutionConfig:
    """Runtime execution settings for live paper trading."""

    ensemble_size: int = 3
    min_confidence: float = 0.67
    buy_pct: float = 0.15
    max_position_pct: float = 0.15
    max_open_positions: int = 5
    stop_loss_pct: float = 0.06
    take_profit_pct: float = 0.15
    max_cost_pct: float = 0.005
    max_drawdown: float = 0.12
    max_daily_loss: float = 0.03
    inactivity_hours: float = 4.0
    interval_minutes: int = 60
    regime_fast_ma: int = 20
    regime_slow_ma: int = 50
    regime_vix_threshold: float = 30.0
    dedupe_window_seconds: int = 3_300
    history_days: int = 30
    order_fill_timeout_seconds: int = 30
    order_poll_interval_seconds: float = 1.0
    promotion_sharpe_min: float = 0.8
    promotion_win_fold_ratio_min: float = 0.6
    promotion_cumulative_return_min: float = 0.0
    promotion_loss_rate_max: float = 0.20
    order_min_notional: float = 10.0

    @classmethod
    def from_dict(cls, data: Optional[dict] = None) -> "LiveExecutionConfig":
        data = data or {}
        known = {field_name: data[field_name] for field_name in cls.__dataclass_fields__ if field_name in data}
        return cls(**known)


@dataclass
class MarketRegime:
    """Current regime filter state."""

    risk_on: bool
    reason: str
    fast_ma: float
    slow_ma: float
    vix: Optional[float]


class SimulatedBroker:
    """Simple broker for local mode with the same normalized adapter."""

    def __init__(self, initial_balance: float):
        self.cash = float(initial_balance)
        self.initial_balance = float(initial_balance)
        self.positions: dict[str, dict] = {}
        self.last_prices: dict[str, float] = {}

    def update_prices(self, prices: dict[str, float]) -> None:
        self.last_prices.update({symbol: float(price) for symbol, price in prices.items()})

    def get_account(self) -> dict:
        equity = self.cash
        for symbol, position in self.positions.items():
            price = self.last_prices.get(symbol, position["avg_price"])
            equity += position["qty"] * price
        return {
            "cash": self.cash,
            "portfolio_value": equity,
            "buying_power": self.cash,
            "equity": equity,
            "last_equity": equity,
            "daytrade_count": 0,
            "pattern_day_trader": False,
        }

    def get_positions(self) -> list[dict]:
        positions = []
        for symbol, position in self.positions.items():
            current_price = self.last_prices.get(symbol, position["avg_price"])
            market_value = position["qty"] * current_price
            cost_basis = position["qty"] * position["avg_price"]
            unrealized_pl = market_value - cost_basis
            unrealized_plpc = unrealized_pl / cost_basis if cost_basis > 0 else 0.0
            positions.append(
                {
                    "symbol": symbol,
                    "qty": position["qty"],
                    "market_value": market_value,
                    "cost_basis": cost_basis,
                    "unrealized_pl": unrealized_pl,
                    "unrealized_plpc": unrealized_plpc,
                    "current_price": current_price,
                    "avg_entry_price": position["avg_price"],
                }
            )
        return positions

    def get_position(self, symbol: str) -> Optional[dict]:
        symbol = symbol.upper()
        for position in self.get_positions():
            if position["symbol"] == symbol:
                return position
        return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        return self.last_prices.get(symbol.upper())

    def buy(self, symbol: str, price: float, amount_usd: float) -> float:
        symbol = symbol.upper()
        if amount_usd > self.cash:
            amount_usd = self.cash
        if amount_usd <= 0:
            return 0.0
        qty = amount_usd / price
        self.cash -= qty * price
        if symbol in self.positions:
            old = self.positions[symbol]
            new_qty = old["qty"] + qty
            self.positions[symbol] = {
                "qty": new_qty,
                "avg_price": (old["avg_price"] * old["qty"] + price * qty) / new_qty,
            }
        else:
            self.positions[symbol] = {"qty": qty, "avg_price": price}
        self.last_prices[symbol] = price
        return qty

    def sell(self, symbol: str, price: float, qty: Optional[float] = None) -> float:
        symbol = symbol.upper()
        position = self.positions.get(symbol)
        if not position:
            return 0.0
        sell_qty = position["qty"] if qty is None else min(position["qty"], qty)
        if sell_qty <= 0:
            return 0.0
        self.cash += sell_qty * price
        remaining = position["qty"] - sell_qty
        if remaining <= 1e-6:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol]["qty"] = remaining
        self.last_prices[symbol] = price
        return sell_qty


class OrderDeduplicator:
    """Blocks repeated orders for the same symbol/side within a short window."""

    def __init__(self, window_seconds: int):
        self.window_seconds = int(window_seconds)
        self._last_orders: dict[tuple[str, str], float] = {}

    def should_block(self, symbol: str, side: str, now_ts: Optional[float] = None) -> bool:
        now_ts = time.time() if now_ts is None else now_ts
        key = (symbol.upper(), side.lower())
        previous = self._last_orders.get(key)
        return previous is not None and (now_ts - previous) < self.window_seconds

    def record(self, symbol: str, side: str, now_ts: Optional[float] = None) -> None:
        now_ts = time.time() if now_ts is None else now_ts
        self._last_orders[(symbol.upper(), side.lower())] = now_ts


class LiveBrokerAdapter:
    """Normalizes live execution across broker implementations."""

    def __init__(
        self,
        broker,
        *,
        broker_name: str,
        fill_timeout: int = 30,
        poll_interval: float = 1.0,
    ):
        self.broker = broker
        self.broker_name = broker_name
        self.fill_timeout = int(fill_timeout)
        self.poll_interval = float(poll_interval)
        self.transaction_model = AdvancedTransactionModel()

    def warmup_symbols(self, symbols: list[str]) -> None:
        if hasattr(self.broker, "warmup_instruments"):
            self.broker.warmup_instruments(symbols)

    def update_market_prices(self, prices: dict[str, float]) -> None:
        if hasattr(self.broker, "update_prices"):
            self.broker.update_prices(prices)

    def get_account(self) -> dict:
        if hasattr(self.broker, "get_account"):
            account = self.broker.get_account()
            if account:
                return account
        if hasattr(self.broker, "get_equity"):
            equity = float(self.broker.get_equity())
            return {
                "cash": equity,
                "portfolio_value": equity,
                "buying_power": equity,
                "equity": equity,
                "last_equity": equity,
                "daytrade_count": 0,
                "pattern_day_trader": False,
            }
        raise ValueError("Broker does not expose account information")

    def get_equity(self, prices: Optional[dict[str, float]] = None) -> float:
        account = self.get_account()
        if prices:
            self.update_market_prices(prices)
        return float(account.get("portfolio_value") or account.get("equity") or 0.0)

    def get_positions_map(self) -> dict[str, dict]:
        raw_positions = self.broker.get_positions()
        if isinstance(raw_positions, dict):
            normalized = {}
            for symbol, position in raw_positions.items():
                symbol = symbol.upper()
                qty = float(position.get("qty", 0.0))
                avg_price = float(
                    position.get("avg_entry_price", position.get("avg_price", 0.0))
                )
                current_price = float(position.get("current_price", avg_price))
                market_value = float(position.get("market_value", qty * current_price))
                cost_basis = float(position.get("cost_basis", qty * avg_price))
                unrealized_pl = float(position.get("unrealized_pl", market_value - cost_basis))
                unrealized_plpc = float(
                    position.get(
                        "unrealized_plpc",
                        unrealized_pl / cost_basis if cost_basis > 0 else 0.0,
                    )
                )
                normalized[symbol] = {
                    "symbol": symbol,
                    "qty": qty,
                    "avg_entry_price": avg_price,
                    "current_price": current_price,
                    "market_value": market_value,
                    "cost_basis": cost_basis,
                    "unrealized_pl": unrealized_pl,
                    "unrealized_plpc": unrealized_plpc,
                }
            return normalized

        normalized = {}
        for position in raw_positions:
            symbol = str(position.get("symbol", "")).upper()
            if not symbol:
                continue
            normalized[symbol] = {
                "symbol": symbol,
                "qty": float(position.get("qty", 0.0)),
                "avg_entry_price": float(position.get("avg_entry_price", 0.0)),
                "current_price": float(position.get("current_price", 0.0)),
                "market_value": float(position.get("market_value", 0.0)),
                "cost_basis": float(position.get("cost_basis", 0.0)),
                "unrealized_pl": float(position.get("unrealized_pl", 0.0)),
                "unrealized_plpc": float(position.get("unrealized_plpc", 0.0)),
                "_raw": position,
            }
        return normalized

    def get_open_orders(self) -> list[dict]:
        if hasattr(self.broker, "get_orders"):
            return list(self.broker.get_orders(status="open"))
        return []

    def get_current_price(self, symbol: str, price_hint: Optional[float] = None) -> Optional[float]:
        symbol = symbol.upper()
        if hasattr(self.broker, "get_current_price"):
            price = self.broker.get_current_price(symbol)
            if price and price > 0:
                return float(price)
        return price_hint

    def estimate_trade_cost(
        self,
        symbol: str,
        *,
        notional: float,
        price_hint: float,
        current_volume: float,
        recent_prices: Optional[pd.Series],
    ) -> dict:
        qty = notional / price_hint if price_hint > 0 else 0.0
        return self.transaction_model.estimate_total_cost(
            ticker=symbol,
            price=price_hint,
            order_size=qty,
            volume=current_volume,
            side="buy",
            recent_prices=recent_prices,
        )

    def buy_notional(self, symbol: str, notional: float, *, price_hint: float, reason: str = "") -> NormalizedOrderResult:
        symbol = symbol.upper()
        price = self.get_current_price(symbol, price_hint=price_hint)
        if price is None or price <= 0:
            return NormalizedOrderResult(
                success=False,
                symbol=symbol,
                side="buy",
                qty=0.0,
                requested_notional=float(notional),
                filled_price=0.0,
                status="rejected",
                reason="price_unavailable",
            )

        qty = float(notional) / float(price)
        if self.broker_name == "alpaca":
            qty = int(qty)
        if qty <= 0:
            return NormalizedOrderResult(
                success=False,
                symbol=symbol,
                side="buy",
                qty=0.0,
                requested_notional=float(notional),
                filled_price=float(price),
                status="rejected",
                reason="size_too_small",
            )

        if hasattr(self.broker, "place_market_order"):
            raw_order = self.broker.place_market_order(symbol, qty, side="buy", reason=reason)
            if not raw_order:
                return NormalizedOrderResult(
                    success=False,
                    symbol=symbol,
                    side="buy",
                    qty=float(qty),
                    requested_notional=float(notional),
                    filled_price=float(price),
                    status="rejected",
                    reason="broker_rejected",
                )
            order_id = str(raw_order.get("id", ""))
            status = str(raw_order.get("status", "submitted"))
            if hasattr(self.broker, "wait_for_order_fill") and order_id:
                waited = self.broker.wait_for_order_fill(
                    order_id,
                    symbol=symbol,
                    timeout=self.fill_timeout,
                    poll_interval=self.poll_interval,
                )
                if waited:
                    status = "filled"
            filled_price = float(raw_order.get("filled_avg_price") or price)
            return NormalizedOrderResult(
                success=status.lower() in {"filled", "completed", "closed", "submitted"},
                symbol=symbol,
                side="buy",
                qty=float(raw_order.get("qty", qty)),
                requested_notional=float(notional),
                filled_price=filled_price,
                status=status,
                order_id=order_id,
                raw=raw_order,
            )

        if hasattr(self.broker, "buy"):
            filled_qty = float(self.broker.buy(symbol, price, notional))
            success = filled_qty > 0
            return NormalizedOrderResult(
                success=success,
                symbol=symbol,
                side="buy",
                qty=filled_qty,
                requested_notional=float(notional),
                filled_price=float(price),
                status="filled" if success else "rejected",
                reason="" if success else "broker_rejected",
            )

        raise ValueError("Broker does not support buy execution")

    def sell_all(self, symbol: str, *, price_hint: float, reason: str = "") -> NormalizedOrderResult:
        symbol = symbol.upper()
        position = self.get_positions_map().get(symbol)
        qty = float(position["qty"]) if position else 0.0
        price = self.get_current_price(symbol, price_hint=price_hint) or 0.0
        if qty <= 0:
            return NormalizedOrderResult(
                success=False,
                symbol=symbol,
                side="sell",
                qty=0.0,
                requested_notional=0.0,
                filled_price=float(price),
                status="rejected",
                reason="no_position",
            )

        if hasattr(self.broker, "close_position"):
            success = bool(self.broker.close_position(symbol, reason=reason))
            return NormalizedOrderResult(
                success=success,
                symbol=symbol,
                side="sell",
                qty=qty,
                requested_notional=qty * float(price),
                filled_price=float(price),
                status="filled" if success else "rejected",
                reason="" if success else "broker_rejected",
            )

        if hasattr(self.broker, "sell"):
            sold_qty = float(self.broker.sell(symbol, price))
            success = sold_qty > 0
            return NormalizedOrderResult(
                success=success,
                symbol=symbol,
                side="sell",
                qty=sold_qty,
                requested_notional=sold_qty * float(price),
                filled_price=float(price),
                status="filled" if success else "rejected",
                reason="" if success else "broker_rejected",
            )

        raise ValueError("Broker does not support sell execution")


class LiveExecutionEngine:
    """Risk and execution filters layered on top of broker adapters."""

    def __init__(self, broker_adapter: LiveBrokerAdapter, config: LiveExecutionConfig):
        self.broker = broker_adapter
        self.config = config
        self.deduplicator = OrderDeduplicator(config.dedupe_window_seconds)

    def evaluate_market_regime(
        self,
        market_data: dict[str, pd.DataFrame],
        macro_data: Optional[pd.DataFrame],
    ) -> MarketRegime:
        spy = market_data.get("SPY")
        if spy is None or len(spy) < self.config.regime_slow_ma:
            return MarketRegime(
                risk_on=False,
                reason="spy_history_unavailable",
                fast_ma=0.0,
                slow_ma=0.0,
                vix=None,
            )

        fast_ma = float(spy["Close"].tail(self.config.regime_fast_ma).mean())
        slow_ma = float(spy["Close"].tail(self.config.regime_slow_ma).mean())
        vix = None
        if macro_data is not None and not macro_data.empty and "vix_close" in macro_data.columns:
            vix = float(macro_data["vix_close"].iloc[-1])

        risk_on = fast_ma > slow_ma and (vix is None or vix < self.config.regime_vix_threshold)
        reason = "risk_on" if risk_on else "risk_filter_blocked"
        return MarketRegime(risk_on=risk_on, reason=reason, fast_ma=fast_ma, slow_ma=slow_ma, vix=vix)

    def get_risk_exit_signals(self, positions_map: dict[str, dict]) -> dict[str, str]:
        signals = {}
        for symbol, position in positions_map.items():
            plpc = float(position.get("unrealized_plpc", 0.0))
            if plpc <= -self.config.stop_loss_pct:
                signals[symbol] = "stop_loss"
            elif plpc >= self.config.take_profit_pct:
                signals[symbol] = "take_profit"
        return signals

    def should_buy(
        self,
        symbol: str,
        *,
        confidence: float,
        price: float,
        current_volume: float,
        recent_prices: Optional[pd.Series],
        positions_map: dict[str, dict],
        equity: float,
        regime: MarketRegime,
    ) -> tuple[bool, str, dict]:
        symbol = symbol.upper()
        if confidence < self.config.min_confidence:
            return False, "low_confidence", {"confidence": confidence}
        if not regime.risk_on:
            return False, regime.reason, {"fast_ma": regime.fast_ma, "slow_ma": regime.slow_ma, "vix": regime.vix}
        if self.deduplicator.should_block(symbol, "buy"):
            return False, "duplicate_signal", {}
        if any(order.get("symbol", "").upper() == symbol for order in self.broker.get_open_orders()):
            return False, "open_order_exists", {}

        active_positions = {
            key: value
            for key, value in positions_map.items()
            if float(value.get("qty", 0.0)) > 0
        }
        if symbol not in active_positions and len(active_positions) >= self.config.max_open_positions:
            return False, "max_open_positions", {"count": len(active_positions)}

        current_position = active_positions.get(symbol)
        if current_position and equity > 0:
            current_position_pct = float(current_position.get("market_value", 0.0)) / equity
            if current_position_pct >= self.config.max_position_pct:
                return False, "max_position_pct", {"position_pct": current_position_pct}

        notional = equity * self.config.buy_pct
        cost_estimate = self.broker.estimate_trade_cost(
            symbol,
            notional=notional,
            price_hint=price,
            current_volume=current_volume,
            recent_prices=recent_prices,
        )
        if float(cost_estimate["breakdown"]["total_cost"]) > self.config.max_cost_pct:
            return False, "cost_too_high", {
                "estimated_cost_pct": float(cost_estimate["breakdown"]["total_cost"]),
            }

        return True, "ok", {
            "notional": notional,
            "cost_estimate": cost_estimate,
        }

    def execute_buy(
        self,
        symbol: str,
        *,
        confidence: float,
        price: float,
        current_volume: float,
        recent_prices: Optional[pd.Series],
        positions_map: dict[str, dict],
        equity: float,
        regime: MarketRegime,
        reason: str,
    ) -> tuple[NormalizedOrderResult, dict]:
        allowed, block_reason, details = self.should_buy(
            symbol,
            confidence=confidence,
            price=price,
            current_volume=current_volume,
            recent_prices=recent_prices,
            positions_map=positions_map,
            equity=equity,
            regime=regime,
        )
        if not allowed:
            return (
                NormalizedOrderResult(
                    success=False,
                    symbol=symbol.upper(),
                    side="buy",
                    qty=0.0,
                    requested_notional=float(details.get("notional", equity * self.config.buy_pct)),
                    filled_price=float(price),
                    status="blocked",
                    reason=block_reason,
                ),
                details,
            )

        notional = float(details["notional"])
        result = self.broker.buy_notional(symbol, notional, price_hint=price, reason=reason)
        if result.success:
            self.deduplicator.record(symbol, "buy")
        return result, details

    def execute_sell(
        self,
        symbol: str,
        *,
        price: float,
        reason: str,
    ) -> NormalizedOrderResult:
        if self.deduplicator.should_block(symbol, "sell"):
            return NormalizedOrderResult(
                success=False,
                symbol=symbol.upper(),
                side="sell",
                qty=0.0,
                requested_notional=0.0,
                filled_price=float(price),
                status="blocked",
                reason="duplicate_signal",
            )
        result = self.broker.sell_all(symbol, price_hint=price, reason=reason)
        if result.success:
            self.deduplicator.record(symbol, "sell")
        return result


def create_live_broker_adapter(
    mode: str,
    *,
    initial_balance: float,
    fill_timeout: int,
    poll_interval: float,
) -> LiveBrokerAdapter:
    """Create a normalized broker adapter for the requested mode."""

    mode = mode.lower().strip()
    if mode == "simulate":
        broker = SimulatedBroker(initial_balance=initial_balance)
        return LiveBrokerAdapter(
            broker,
            broker_name="simulate",
            fill_timeout=fill_timeout,
            poll_interval=poll_interval,
        )

    broker = create_broker(mode, paper_trading=True)
    return LiveBrokerAdapter(
        broker,
        broker_name=mode,
        fill_timeout=fill_timeout,
        poll_interval=poll_interval,
    )
