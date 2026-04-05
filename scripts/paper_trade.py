#!/usr/bin/env python3
"""Paper trading pipeline with eToro Demo as the primary execution path."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.schema import validate_config
from core.artifacts import (
    DEMO_SESSION_EQUITY_FILENAME,
    DEMO_SESSION_EVENTS_FILENAME,
    DEMO_SESSION_LEGACY_JOURNAL_FILENAME,
    DEMO_SESSION_META_FILENAME,
    DEMO_SESSION_REPORT_FILENAME,
    append_jsonl,
    save_json,
)
from core.ensemble import EnsemblePredictor
from core.live_observation import LiveObservationEngine
from core.macro_data import MacroDataFetcher
from training.strategy_policies import StrategyContext, build_strategy_policy
from trading.live_execution import (
    LiveExecutionConfig,
    LiveExecutionEngine,
    create_live_broker_adapter,
)

load_dotenv()

logger = logging.getLogger("PaperTrader")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

DEFAULT_INITIAL_BALANCE = 100_000
KILL_SWITCH_MAX_DRAWDOWN = 0.15
KILL_SWITCH_MAX_DAILY_LOSS = 0.05
KILL_SWITCH_INACTIVITY_HOURS = 4
TRADE_INTERVAL_MINUTES = 60

PPO_LIVE_FAMILIES = {"ppo_single", "ppo_ensemble"}
CONFIG_DRIVEN_LIVE_FAMILIES = {"rule_momentum_regime", "supervised_ranker"}


class KillSwitch:
    """Monitor automatic stop conditions."""

    def __init__(
        self,
        initial_balance: float,
        max_drawdown: float = KILL_SWITCH_MAX_DRAWDOWN,
        max_daily_loss: float = KILL_SWITCH_MAX_DAILY_LOSS,
        inactivity_hours: float = KILL_SWITCH_INACTIVITY_HOURS,
    ):
        self.initial_balance = float(initial_balance)
        self.max_drawdown = float(max_drawdown)
        self.max_daily_loss = float(max_daily_loss)
        self.inactivity_hours = float(inactivity_hours)
        self.peak_equity = float(initial_balance)
        self.daily_start_equity = float(initial_balance)
        self.last_trade_time = datetime.now()
        self.last_daily_reset = datetime.now().date()
        self.triggered = False
        self.trigger_reason: Optional[str] = None
        self.alerts: list[str] = []

    def check(self, current_equity: float) -> tuple[bool, Optional[str]]:
        now = datetime.now()
        if now.date() != self.last_daily_reset:
            self.daily_start_equity = float(current_equity)
            self.last_daily_reset = now.date()

        if current_equity > self.peak_equity:
            self.peak_equity = float(current_equity)

        drawdown = (self.peak_equity - current_equity) / max(self.peak_equity, 1e-8)
        if drawdown >= self.max_drawdown:
            self.triggered = True
            self.trigger_reason = f"MAX_DRAWDOWN {drawdown:.1%} >= {self.max_drawdown:.1%}"
            return True, self.trigger_reason

        daily_loss = (self.daily_start_equity - current_equity) / max(self.daily_start_equity, 1e-8)
        if daily_loss >= self.max_daily_loss:
            self.triggered = True
            self.trigger_reason = f"MAX_DAILY_LOSS {daily_loss:.1%} >= {self.max_daily_loss:.1%}"
            return True, self.trigger_reason

        hours_since_trade = (now - self.last_trade_time).total_seconds() / 3600
        if hours_since_trade >= self.inactivity_hours:
            alert = f"INACTIVITY {hours_since_trade:.1f}h since last trade"
            if alert not in self.alerts:
                self.alerts.append(alert)
                logger.warning("  [WARN] %s", alert)

        return False, None

    def record_trade(self) -> None:
        self.last_trade_time = datetime.now()


class LiveTradeJournal:
    """Append-only session telemetry for the demo dashboard."""

    def __init__(self, session_dir: Path, session_meta: dict[str, Any]):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = self.session_dir.name
        self.start_time = datetime.now()
        self.trades: list[dict[str, Any]] = []
        self.rejections: list[dict[str, Any]] = []
        self.signals: list[dict[str, Any]] = []
        self.alerts: list[dict[str, Any]] = []
        self.equity_curve: list[dict[str, Any]] = []
        self._seen_alert_keys: set[str] = set()

        self.meta_path = self.session_dir / DEMO_SESSION_META_FILENAME
        self.events_path = self.session_dir / DEMO_SESSION_EVENTS_FILENAME
        self.equity_path = self.session_dir / DEMO_SESSION_EQUITY_FILENAME
        self.report_path = self.session_dir / DEMO_SESSION_REPORT_FILENAME
        self.legacy_journal_path = self.session_dir / DEMO_SESSION_LEGACY_JOURNAL_FILENAME

        payload = dict(session_meta)
        payload["session_id"] = self.session_id
        payload["session_dir"] = str(self.session_dir)
        payload["created_at"] = datetime.now().isoformat()
        save_json(self.meta_path, payload)

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        append_jsonl(path, payload)

    def _append_event(self, payload: dict[str, Any]) -> None:
        event = dict(payload)
        event.setdefault("timestamp", datetime.now().isoformat())
        event.setdefault("session_id", self.session_id)
        self._append_jsonl(self.events_path, event)

    def record_signal(
        self,
        ticker: str,
        action: str,
        *,
        confidence: float,
        price: float,
        reason: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        payload = {
            "type": "signal",
            "ticker": ticker,
            "action": action,
            "confidence": float(confidence),
            "price": float(price),
            "reason": reason,
            "details": details or {},
        }
        self.signals.append(payload)
        self._append_event(payload)

    def record_trade(
        self,
        ticker: str,
        side: str,
        price: float,
        qty: float,
        total_value: float,
        *,
        reason: str = "",
        confidence: Optional[float] = None,
        order_id: str = "",
        status: str = "filled",
    ) -> None:
        trade = {
            "type": "trade",
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "symbol": ticker,
            "side": side,
            "action": side,
            "price": float(price),
            "qty": float(qty),
            "quantity": float(qty),
            "total_value": float(total_value),
            "amount": float(total_value),
            "reason": reason,
            "confidence": None if confidence is None else float(confidence),
            "order_id": order_id,
            "status": status,
        }
        self.trades.append(trade)
        self._append_event(trade)
        logger.info(
            "  TRADE: %s %.4fx %s @ $%.2f = $%.2f [%s]",
            side,
            qty,
            ticker,
            price,
            total_value,
            reason or "n/a",
        )

    def record_rejection(
        self,
        ticker: str,
        side: str,
        reason: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        rejection = {
            "type": "rejection",
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "symbol": ticker,
            "side": side,
            "action": side,
            "reason": reason,
            "details": details or {},
        }
        self.rejections.append(rejection)
        self._append_event(rejection)
        logger.info("  BLOCKED: %s %s (%s)", side, ticker, reason)

    def record_alert(
        self,
        level: str,
        reason: str,
        details: Optional[dict[str, Any]] = None,
        *,
        dedupe_key: Optional[str] = None,
    ) -> None:
        key = dedupe_key or f"{level}:{reason}"
        if key in self._seen_alert_keys:
            return
        self._seen_alert_keys.add(key)
        alert = {
            "type": "alert",
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "reason": reason,
            "details": details or {},
        }
        self.alerts.append(alert)
        self._append_event(alert)
        logger.warning("  ALERT: %s | %s", level.upper(), reason)

    def record_kill_switch(self, reason: str, details: Optional[dict[str, Any]] = None) -> None:
        self._append_event(
            {
                "type": "kill_switch",
                "timestamp": datetime.now().isoformat(),
                "reason": reason,
                "details": details or {},
            }
        )

    def record_equity(
        self,
        equity: float,
        *,
        balance: float,
        n_positions: int,
        drawdown: float,
        exposure: float,
        source: str = "journal",
    ) -> None:
        point = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "equity": float(equity),
            "balance": float(balance),
            "n_positions": int(n_positions),
            "drawdown": float(drawdown),
            "exposure": float(exposure),
            "source": source,
        }
        self.equity_curve.append(point)
        self._append_jsonl(self.equity_path, point)

    def get_summary(self, initial_balance: float) -> dict[str, Any]:
        if not self.equity_curve:
            return {
                "n_trades": 0,
                "n_rejections": len(self.rejections),
                "n_buys": 0,
                "n_sells": 0,
                "total_return": 0.0,
                "final_equity": float(initial_balance),
                "duration_hours": 0.0,
            }

        final_equity = float(self.equity_curve[-1]["equity"])
        return {
            "n_trades": len(self.trades),
            "n_rejections": len(self.rejections),
            "n_signals": len(self.signals),
            "n_alerts": len(self.alerts),
            "n_buys": sum(1 for trade in self.trades if trade["side"] == "BUY"),
            "n_sells": sum(1 for trade in self.trades if trade["side"] == "SELL"),
            "total_return": (final_equity - initial_balance) / max(initial_balance, 1e-8),
            "final_equity": final_equity,
            "duration_hours": (datetime.now() - self.start_time).total_seconds() / 3600,
        }

    def export(self, path: Path) -> None:
        payload = {
            "session_id": self.session_id,
            "trades": self.trades,
            "rejections": self.rejections,
            "signals": self.signals,
            "alerts": self.alerts,
            "equity_curve": self.equity_curve,
            "start_time": self.start_time.isoformat(),
            "export_time": datetime.now().isoformat(),
        }
        save_json(path, payload)
        logger.info("  Journal exporte: %s", path)

    def finalize(self, report: dict[str, Any]) -> None:
        self.export(self.legacy_journal_path)
        save_json(self.report_path, report)
        logger.info("  Rapport: %s", self.report_path)


@dataclass
class LiveInferenceRuntime:
    """Resolved live inference backend for the selected strategy family."""

    family: str
    tickers: list[str]
    effective_config: dict[str, Any]
    model_obs_size: Optional[int]
    vecnorm_path: Optional[Path]
    model_paths: list[Path]
    predictor: Optional[EnsemblePredictor] = None
    policy: Optional[Any] = None
    policy_ready: bool = False


def merge_configs(base: Optional[dict], override: Optional[dict]) -> dict:
    """Merge sectioned configs recursively."""

    merged = dict(base or {})
    for key, value in dict(override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_runtime_config(config_path: Optional[str]) -> dict:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    warnings = validate_config(config)
    for warning in warnings:
        logger.warning("  %s", warning)
    return config


def _strategy_family(config: Optional[dict]) -> str:
    strategy_cfg = (config or {}).get("strategy", {})
    family = str(strategy_cfg.get("family", "ppo_ensemble")).strip().lower()
    return family or "ppo_ensemble"


def discover_model_paths(model_path: str | Path, ensemble_size: int) -> list[Path]:
    """Discover up to ensemble_size model paths near the requested model."""

    requested = Path(model_path)
    discovered: list[Path] = []
    candidates: list[Path] = []

    if requested.is_dir():
        if (requested / "model.zip").exists():
            candidates.append(requested / "model.zip")
        candidates.extend(sorted(requested.glob("fold_*/model.zip")))
        candidates.extend(sorted(requested.glob("*.zip")))
    else:
        candidates.append(requested)
        if requested.parent.name.startswith("fold_"):
            candidates.extend(sorted(requested.parent.parent.glob("fold_*/model.zip")))
        else:
            candidates.extend(sorted(requested.parent.glob("fold_*/model.zip")))

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate.exists() and candidate not in seen:
            seen.add(candidate)
            discovered.append(candidate)
        if len(discovered) >= ensemble_size:
            break

    if not discovered:
        raise FileNotFoundError(f"No model found for {model_path}")
    return discovered


def resolve_vecnormalize_path(
    primary_model: str | Path,
    candidate: Optional[str | Path] = None,
) -> Optional[Path]:
    """Resolve VecNormalize stats across old and new artifact layouts."""

    if candidate:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate_path

    model_path = Path(primary_model)
    fallbacks = [
        model_path.with_name("vecnormalize.pkl"),
        model_path.with_name(f"{model_path.stem}_vecnormalize.pkl"),
        model_path.parent / "vecnormalize.pkl",
        model_path.parent / f"{model_path.stem}_vecnormalize.pkl",
    ]
    for path in fallbacks:
        if path.exists():
            return path
    return None


def load_predictor_bundle(
    model_path: str | Path,
    live_settings: LiveExecutionConfig,
    config: dict,
    *,
    requested_models: Optional[int] = None,
) -> tuple[EnsemblePredictor, list[str], int, dict, Optional[Path], list[Path]]:
    """Load the ensemble predictor and metadata required for live trading."""

    from stable_baselines3 import PPO
    from scripts.backtest_ultimate import load_model_metadata

    model_paths = discover_model_paths(model_path, requested_models or live_settings.ensemble_size)
    primary_model = model_paths[0]
    metadata, model_config, vecnorm_path = load_model_metadata(str(primary_model))
    vecnorm_path = resolve_vecnormalize_path(primary_model, vecnorm_path)
    base_model = PPO.load(str(primary_model))
    obs_shape = tuple(base_model.observation_space.shape)

    predictor = EnsemblePredictor.load(
        [str(path) for path in model_paths],
        vecnorm_path=str(vecnorm_path) if vecnorm_path else None,
        obs_shape=obs_shape,
        use_recurrent=False,
    )
    tickers = metadata.get("tickers") if metadata else None
    if not tickers:
        tickers = config.get("data", {}).get("tickers", [])
    if not tickers:
        raise ValueError("No tickers found in model metadata or runtime config")

    return predictor, list(tickers), obs_shape[0], model_config or {}, vecnorm_path, model_paths


def resolve_live_inference_runtime(
    model_path: Optional[str | Path],
    *,
    live_settings: LiveExecutionConfig,
    runtime_config: dict,
) -> LiveInferenceRuntime:
    """Resolve the live inference backend from the configured strategy family."""

    family = _strategy_family(runtime_config)
    if family in PPO_LIVE_FAMILIES:
        if not model_path:
            raise ValueError(f"strategy.family={family} requires --model")
        requested_models = 1 if family == "ppo_single" else live_settings.ensemble_size
        predictor, tickers, model_obs_size, model_config, vecnorm_path, model_paths = load_predictor_bundle(
            model_path,
            live_settings,
            runtime_config,
            requested_models=requested_models,
        )
        effective_config = merge_configs(model_config, runtime_config)
        return LiveInferenceRuntime(
            family=family,
            tickers=list(tickers),
            effective_config=effective_config,
            model_obs_size=model_obs_size,
            vecnorm_path=vecnorm_path,
            model_paths=model_paths,
            predictor=predictor,
        )

    if family == "recurrent_ppo":
        raise NotImplementedError(
            "strategy.family=recurrent_ppo is not supported yet for live demo trading. "
            "Use ppo_single, ppo_ensemble, supervised_ranker, or rule_momentum_regime."
        )

    if family in CONFIG_DRIVEN_LIVE_FAMILIES:
        tickers = list(runtime_config.get("data", {}).get("tickers", []))
        if not tickers:
            raise ValueError(f"strategy.family={family} requires data.tickers in the runtime config")
        policy = build_strategy_policy(family, runtime_config)
        return LiveInferenceRuntime(
            family=family,
            tickers=tickers,
            effective_config=dict(runtime_config),
            model_obs_size=None,
            vecnorm_path=None,
            model_paths=[],
            policy=policy,
        )

    raise ValueError(f"Unsupported live strategy family: {family}")


def fetch_live_data(
    tickers: list[str],
    *,
    interval: str = "1h",
    history_days: int = 30,
) -> dict:
    """Fetch recent bars for the configured ticker universe."""

    from core.data_fetcher import UniversalDataFetcher

    fetcher = UniversalDataFetcher()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=history_days)
    data = {}
    for ticker in tickers:
        try:
            df = fetcher.fetch(
                ticker,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                interval=interval,
            )
            if df is not None and len(df) >= 60:
                data[ticker] = df
        except Exception as exc:
            logger.warning("  %s: fetch erreur (%s)", ticker, exc)
    return data


def fetch_live_macro_data(
    market_data: dict,
    *,
    interval: str = "1h",
) -> Optional[object]:
    """Fetch macro data aligned to the live history window."""

    if not market_data:
        return None

    ref_ticker = next(iter(market_data))
    ref_df = market_data[ref_ticker]
    fetcher = MacroDataFetcher()
    try:
        macro_data = fetcher.fetch_all(
            start_date=str(ref_df.index[0].date()),
            end_date=str(ref_df.index[-1].date()),
            interval=interval,
        )
        return macro_data if not macro_data.empty else None
    except Exception as exc:
        logger.warning("  Macro fetch failed: %s", exc)
        return None


def get_model_actions(
    predictor: EnsemblePredictor,
    observation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Get ensemble actions with per-asset confidence scores."""

    actions, confidences = predictor.predict_with_asset_confidences(
        observation,
        deterministic=True,
    )
    return np.array(actions).flatten(), np.array(confidences).flatten()


def _macro_row_from_snapshot(snapshot) -> Optional[dict[str, float]]:
    aligned_macro = getattr(snapshot, "aligned_macro", None)
    current_step = int(getattr(snapshot, "current_step", -1))
    if aligned_macro is None or aligned_macro.empty or current_step < 0:
        return None
    row = aligned_macro.iloc[current_step]
    return {column: float(row[column]) for column in aligned_macro.columns}


def _build_strategy_context(
    *,
    snapshot,
    tickers: list[str],
    prices: dict[str, float],
    positions_map: dict[str, dict],
    regime,
    interval: str,
) -> StrategyContext:
    portfolio = {
        ticker: float(positions_map.get(ticker, {}).get("qty", 0.0))
        for ticker in tickers
    }
    entry_prices = {
        ticker: float(positions_map.get(ticker, {}).get("avg_entry_price", 0.0))
        for ticker in tickers
    }
    return StrategyContext(
        observation=np.asarray(snapshot.observation, dtype=np.float32),
        current_step=int(snapshot.current_step),
        tickers=list(tickers),
        prices=dict(prices),
        portfolio=portfolio,
        entry_prices=entry_prices,
        processed_data=snapshot.processed_data,
        feature_columns=list(getattr(snapshot, "feature_columns", [])),
        macro_columns=list(getattr(snapshot, "macro_columns", [])),
        macro_row=_macro_row_from_snapshot(snapshot),
        interval=interval,
        regime_risk_on=bool(regime.risk_on),
    )


def fit_runtime_policy_if_needed(
    runtime: LiveInferenceRuntime,
    *,
    market_data: dict,
    macro_data,
) -> None:
    """Fit config-driven live policies once from the recent historical window."""

    if runtime.policy is None or runtime.policy_ready:
        return
    runtime.policy.fit(market_data, macro_data)
    runtime.policy_ready = True


def get_live_actions(
    runtime: LiveInferenceRuntime,
    *,
    snapshot,
    positions_map: dict[str, dict],
    prices: dict[str, float],
    regime,
    interval: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return actions/confidences for either PPO artifacts or config-driven policies."""

    if runtime.predictor is not None:
        return get_model_actions(runtime.predictor, snapshot.observation)

    if runtime.policy is None:
        raise RuntimeError("No live predictor or policy has been configured")

    context = _build_strategy_context(
        snapshot=snapshot,
        tickers=runtime.tickers,
        prices=prices,
        positions_map=positions_map,
        regime=regime,
        interval=interval,
    )
    actions = np.asarray(runtime.policy.predict_actions(context), dtype=np.int64).flatten()
    confidences = np.where(actions == 0, 0.0, 1.0).astype(np.float32)
    return actions, confidences


def _compute_exposure(positions_map: dict[str, dict], equity: float) -> float:
    total_market_value = sum(float(position.get("market_value", 0.0)) for position in positions_map.values())
    return total_market_value / max(equity, 1e-8)


def _latest_session_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = base_dir / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def _build_session_meta(
    *,
    mode: str,
    strategy_family: str,
    config_path: Optional[str],
    tickers: list[str],
    model_path: Optional[str | Path],
    model_paths: list[Path],
    live_settings: LiveExecutionConfig,
    interval: str,
) -> dict[str, Any]:
    return {
        "broker": mode,
        "mode": mode,
        "strategy_family": strategy_family,
        "config_path": config_path,
        "tickers": list(tickers),
        "model_path": str(model_path) if model_path is not None else None,
        "resolved_models": [str(path) for path in model_paths],
        "start_time": datetime.now().isoformat(),
        "interval": interval,
        "live_settings": vars(live_settings),
    }


def _record_threshold_alerts(
    journal: LiveTradeJournal,
    kill_switch: KillSwitch,
    *,
    equity: float,
    live_settings: LiveExecutionConfig,
) -> None:
    drawdown = (kill_switch.peak_equity - equity) / max(kill_switch.peak_equity, 1e-8)
    daily_loss = (kill_switch.daily_start_equity - equity) / max(kill_switch.daily_start_equity, 1e-8)

    if drawdown >= (live_settings.max_drawdown * 0.8):
        journal.record_alert(
            "warning",
            "drawdown_near_limit",
            {
                "drawdown": drawdown,
                "threshold": live_settings.max_drawdown,
            },
            dedupe_key="drawdown_near_limit",
        )
    if daily_loss >= (live_settings.max_daily_loss * 0.8):
        journal.record_alert(
            "warning",
            "daily_loss_near_limit",
            {
                "daily_loss": daily_loss,
                "threshold": live_settings.max_daily_loss,
            },
            dedupe_key="daily_loss_near_limit",
        )


def run_paper_trading(
    model_path,
    mode: str = "simulate",
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    initial_balance: float = DEFAULT_INITIAL_BALANCE,
    interval_minutes: Optional[int] = None,
    max_hours: float = 24,
    buy_pct: Optional[float] = None,
    config_path: Optional[str] = "config/config.yaml",
    max_iterations: Optional[int] = None,
):
    """Main paper trading loop."""

    mode = mode.lower().strip()
    runtime_config = load_runtime_config(config_path)
    live_settings = LiveExecutionConfig.from_dict(runtime_config.get("live"))
    if interval_minutes is not None:
        live_settings.interval_minutes = int(interval_minutes)
    if buy_pct is not None:
        live_settings.buy_pct = float(buy_pct)

    if mode == "alpaca":
        if api_key:
            os.environ["ALPACA_PAPER_API_KEY"] = api_key
            os.environ["ALPACA_API_KEY"] = api_key
        if api_secret:
            os.environ["ALPACA_PAPER_SECRET_KEY"] = api_secret
            os.environ["ALPACA_SECRET_KEY"] = api_secret

    inference_runtime = resolve_live_inference_runtime(
        model_path,
        live_settings=live_settings,
        runtime_config=runtime_config,
    )
    effective_config = dict(inference_runtime.effective_config)
    tickers = list(inference_runtime.tickers)
    model_obs_size = inference_runtime.model_obs_size
    vecnorm_path = inference_runtime.vecnorm_path
    model_paths = list(inference_runtime.model_paths)
    data_cfg = effective_config.get("data", {})
    env_cfg = effective_config.get("environment", {})
    initial_balance = float(env_cfg.get("initial_balance", initial_balance))

    logger.info("\n%s", "=" * 70)
    logger.info("PAPER TRADER")
    logger.info("%s", "=" * 70)
    logger.info("  Strategy   : %s", inference_runtime.family)
    logger.info(
        "  Models     : %s",
        ", ".join(path.name for path in model_paths) if model_paths else "config-driven",
    )
    logger.info("  Mode       : %s", mode)
    logger.info("  Tickers    : %s", ", ".join(tickers))
    logger.info("  VecNorm    : %s", vecnorm_path if vecnorm_path else "none")
    logger.info("  Interval   : %s min", live_settings.interval_minutes)
    logger.info("  Max heures : %s", max_hours)
    logger.info("%s\n", "=" * 70)

    broker_adapter = create_live_broker_adapter(
        mode,
        initial_balance=initial_balance,
        fill_timeout=live_settings.order_fill_timeout_seconds,
        poll_interval=live_settings.order_poll_interval_seconds,
    )
    broker_adapter.warmup_symbols(tickers)

    account = broker_adapter.get_account()
    initial_balance = float(account.get("portfolio_value", initial_balance))
    kill_switch = KillSwitch(
        initial_balance,
        max_drawdown=live_settings.max_drawdown,
        max_daily_loss=live_settings.max_daily_loss,
        inactivity_hours=live_settings.inactivity_hours,
    )
    observation_engine = LiveObservationEngine(
        tickers=tickers,
        initial_balance=initial_balance,
        max_features_per_ticker=int(env_cfg.get("max_features_per_ticker", 0)),
        target_observation_size=model_obs_size,
    )
    if hasattr(observation_engine, "reset"):
        observation_engine.reset()
    execution_engine = LiveExecutionEngine(broker_adapter, live_settings)

    interval = data_cfg.get("interval", "1h")
    start_time = datetime.now()
    session_root = Path("logs/paper_trading")
    session_dir = _latest_session_dir(session_root)
    journal = LiveTradeJournal(
        session_dir,
        _build_session_meta(
            mode=mode,
            strategy_family=inference_runtime.family,
            config_path=config_path,
            tickers=tickers,
            model_path=model_path,
            model_paths=model_paths,
            live_settings=live_settings,
            interval=interval,
        ),
    )
    running = True
    iteration = 0
    last_alert_count = 0

    def signal_handler(sig, frame):
        del sig, frame
        nonlocal running
        logger.info("\n[STOP] Shutdown requested")
        running = False

    signal.signal(signal.SIGINT, signal_handler)

    while running:
        if max_iterations is not None and iteration >= max_iterations:
            break

        elapsed_hours = (datetime.now() - start_time).total_seconds() / 3600
        if elapsed_hours >= max_hours:
            logger.info("\n[TIME] Duree max atteinte (%sh)", max_hours)
            break

        iteration += 1
        logger.info(
            "\n--- Iteration %s | %s | %.1fh ---",
            iteration,
            datetime.now().strftime("%H:%M:%S"),
            elapsed_hours,
        )

        market_data = fetch_live_data(
            tickers,
            interval=interval,
            history_days=live_settings.history_days,
        )
        missing_tickers = [ticker for ticker in tickers if ticker not in market_data]
        if missing_tickers:
            logger.warning("  Missing tickers: %s. Skip.", ", ".join(missing_tickers))
            journal.record_rejection(
                "ALL",
                "FETCH",
                "missing_market_data",
                {"missing": missing_tickers},
            )
            if running and (max_iterations is None or iteration < max_iterations):
                time.sleep(max(live_settings.interval_minutes, 0) * 60)
            continue

        macro_data = fetch_live_macro_data(market_data, interval=interval)
        if inference_runtime.policy is not None and not inference_runtime.policy_ready:
            try:
                fit_runtime_policy_if_needed(
                    inference_runtime,
                    market_data=market_data,
                    macro_data=macro_data,
                )
            except Exception as exc:
                logger.exception("  Strategy initialization failed: %s", exc)
                journal.record_alert(
                    "critical",
                    "strategy_initialization_failed",
                    {"family": inference_runtime.family, "error": str(exc)},
                    dedupe_key=f"strategy_init:{inference_runtime.family}",
                )
                journal.record_rejection(
                    "ALL",
                    "STRATEGY",
                    "strategy_initialization_failed",
                    {"family": inference_runtime.family, "error": str(exc)},
                )
                if running and (max_iterations is None or iteration < max_iterations):
                    time.sleep(max(live_settings.interval_minutes, 0) * 60)
                continue
        price_hints = {ticker: float(df["Close"].iloc[-1]) for ticker, df in market_data.items()}
        prices = {
            ticker: float(
                broker_adapter.get_current_price(ticker, price_hint=price_hints[ticker]) or price_hints[ticker]
            )
            for ticker in tickers
        }
        broker_adapter.update_market_prices(prices)

        positions_map = broker_adapter.get_positions_map()
        account = broker_adapter.get_account()
        equity = float(account.get("portfolio_value") or account.get("equity") or 0.0)
        balance = float(account.get("cash", equity))
        drawdown = (kill_switch.peak_equity - equity) / max(kill_switch.peak_equity, 1e-8)
        exposure = _compute_exposure(positions_map, equity)
        journal.record_equity(
            equity,
            balance=balance,
            n_positions=len([p for p in positions_map.values() if float(p.get("qty", 0.0)) > 0]),
            drawdown=drawdown,
            exposure=exposure,
            source="broker",
        )

        triggered, reason = kill_switch.check(equity)
        if len(kill_switch.alerts) > last_alert_count:
            for alert in kill_switch.alerts[last_alert_count:]:
                journal.record_alert("warning", "kill_switch_inactivity", {"message": alert}, dedupe_key=alert)
            last_alert_count = len(kill_switch.alerts)

        _record_threshold_alerts(journal, kill_switch, equity=equity, live_settings=live_settings)
        if triggered:
            journal.record_kill_switch(reason or "kill_switch_triggered")
            logger.error("  KILL SWITCH: %s", reason)
            break

        exit_signals = execution_engine.get_risk_exit_signals(positions_map)
        for ticker, exit_reason in exit_signals.items():
            journal.record_signal(
                ticker,
                "SELL",
                confidence=1.0,
                price=prices.get(ticker, 0.0),
                reason=exit_reason,
                details={"source": "risk_exit"},
            )
            result = execution_engine.execute_sell(
                ticker,
                price=prices.get(ticker, 0.0),
                reason=exit_reason,
            )
            if result.success:
                journal.record_trade(
                    ticker,
                    "SELL",
                    result.filled_price,
                    result.qty,
                    result.qty * result.filled_price,
                    reason=exit_reason,
                    order_id=result.order_id,
                    status=result.status,
                )
                kill_switch.record_trade()
            else:
                journal.record_rejection(
                    ticker,
                    "SELL",
                    result.reason or "risk_exit_failed",
                )

        positions_map = broker_adapter.get_positions_map()
        account = broker_adapter.get_account()
        equity = float(account.get("portfolio_value") or account.get("equity") or 0.0)
        balance = float(account.get("cash", equity))

        snapshot = observation_engine.build_snapshot(
            market_data,
            positions_map,
            balance=balance,
            equity=equity,
            macro_data=macro_data,
        )
        if model_obs_size is not None and snapshot.observation.shape[0] != model_obs_size:
            journal.record_alert(
                "critical",
                "observation_mismatch",
                {
                    "expected": model_obs_size,
                    "actual": int(snapshot.observation.shape[0]),
                },
                dedupe_key="observation_mismatch",
            )
            journal.record_rejection(
                "ALL",
                "OBS",
                "observation_mismatch",
                {
                    "expected": model_obs_size,
                    "actual": int(snapshot.observation.shape[0]),
                },
            )
            logger.warning(
                "  Obs mismatch: expected=%s actual=%s",
                model_obs_size,
                snapshot.observation.shape[0],
            )
            if running and (max_iterations is None or iteration < max_iterations):
                time.sleep(max(live_settings.interval_minutes, 0) * 60)
            continue

        regime = execution_engine.evaluate_market_regime(market_data, macro_data)
        actions, confidences = get_live_actions(
            inference_runtime,
            snapshot=snapshot,
            positions_map=positions_map,
            prices=prices,
            regime=regime,
            interval=interval,
        )
        buy_reason = f"{inference_runtime.family}_buy"
        sell_reason = f"{inference_runtime.family}_sell"

        for idx, ticker in enumerate(tickers):
            action = int(actions[idx]) if idx < len(actions) else 0
            confidence = float(confidences[idx]) if idx < len(confidences) else 0.0
            price = prices.get(ticker, 0.0)

            if price <= 0:
                journal.record_alert(
                    "warning",
                    "price_unavailable",
                    {"ticker": ticker},
                    dedupe_key=f"price_unavailable:{ticker}",
                )
                journal.record_rejection(ticker, "NONE", "price_unavailable")
                continue

            if action == 2:
                if float(positions_map.get(ticker, {}).get("qty", 0.0)) <= 0:
                    continue
                journal.record_signal(
                    ticker,
                    "SELL",
                    confidence=confidence,
                    price=price,
                    reason=sell_reason,
                    details={"regime": regime.reason, "family": inference_runtime.family},
                )
                result = execution_engine.execute_sell(
                    ticker,
                    price=price,
                    reason=sell_reason,
                )
                if result.success:
                    journal.record_trade(
                        ticker,
                        "SELL",
                        result.filled_price,
                        result.qty,
                        result.qty * result.filled_price,
                        reason=sell_reason,
                        confidence=confidence,
                        order_id=result.order_id,
                        status=result.status,
                    )
                    kill_switch.record_trade()
                    positions_map = broker_adapter.get_positions_map()
                else:
                    journal.record_rejection(
                        ticker,
                        "SELL",
                        result.reason or "sell_failed",
                        {"confidence": confidence},
                    )
                continue

            if action == 1:
                journal.record_signal(
                    ticker,
                    "BUY",
                    confidence=confidence,
                    price=price,
                    reason=buy_reason,
                    details={"regime": regime.reason, "family": inference_runtime.family},
                )
                result, details = execution_engine.execute_buy(
                    ticker,
                    confidence=confidence,
                    price=price,
                    current_volume=snapshot.volumes[ticker],
                    recent_prices=snapshot.recent_prices[ticker],
                    positions_map=positions_map,
                    equity=equity,
                    regime=regime,
                    reason=buy_reason,
                )
                if result.success:
                    journal.record_trade(
                        ticker,
                        "BUY",
                        result.filled_price,
                        result.qty,
                        result.requested_notional,
                        reason=buy_reason,
                        confidence=confidence,
                        order_id=result.order_id,
                        status=result.status,
                    )
                    kill_switch.record_trade()
                    positions_map = broker_adapter.get_positions_map()
                else:
                    details = dict(details)
                    details["confidence"] = confidence
                    journal.record_rejection(
                        ticker,
                        "BUY",
                        result.reason or "buy_failed",
                        details,
                    )

        account = broker_adapter.get_account()
        equity = float(account.get("portfolio_value") or account.get("equity") or 0.0)
        positions_map = broker_adapter.get_positions_map()
        positions_count = len([position for position in positions_map.values() if float(position.get("qty", 0.0)) > 0])
        total_return = (equity - initial_balance) / max(initial_balance, 1e-8)
        drawdown = (kill_switch.peak_equity - equity) / max(kill_switch.peak_equity, 1e-8)
        exposure = _compute_exposure(positions_map, equity)
        journal.record_equity(
            equity,
            balance=float(account.get("cash", equity)),
            n_positions=positions_count,
            drawdown=drawdown,
            exposure=exposure,
            source="post_iteration",
        )
        logger.info(
            "  Equity: $%s (%+.2f%%) | DD: %.1f%% | Positions: %s | Regime: %s",
            f"{equity:,.2f}",
            total_return * 100,
            drawdown * 100,
            positions_count,
            "risk_on" if regime.risk_on else "risk_off",
        )

        if running and (max_iterations is None or iteration < max_iterations):
            logger.info("  Prochain check dans %s min...", live_settings.interval_minutes)
            time.sleep(max(live_settings.interval_minutes, 0) * 60)

    logger.info("\n%s", "=" * 70)
    logger.info("FIN DU PAPER TRADING")
    logger.info("%s", "=" * 70)

    summary = journal.get_summary(initial_balance)
    logger.info("  Trades      : %s", summary["n_trades"])
    logger.info("  Rejections  : %s", summary["n_rejections"])
    logger.info("  Return      : %+.2f%%", summary.get("total_return", 0.0) * 100)
    logger.info("  Equity fin  : $%s", f"{summary.get('final_equity', initial_balance):,.2f}")
    logger.info("  Duree       : %.1fh", summary.get("duration_hours", 0.0))
    if kill_switch.triggered:
        logger.warning("  Kill Switch : %s", kill_switch.trigger_reason)

    report = {
        "session_id": journal.session_id,
        "session_dir": str(journal.session_dir),
        "model_path": str(model_path) if model_path is not None else None,
        "resolved_models": [str(path) for path in model_paths],
        "mode": mode,
        "strategy_family": inference_runtime.family,
        "config_path": config_path,
        "start_time": start_time.isoformat(),
        "end_time": datetime.now().isoformat(),
        "tickers": tickers,
        "initial_balance": initial_balance,
        "summary": summary,
        "kill_switch": {
            "triggered": kill_switch.triggered,
            "reason": kill_switch.trigger_reason,
            "alerts": kill_switch.alerts,
        },
        "live_settings": vars(live_settings),
        "vecnorm_path": str(vecnorm_path) if vecnorm_path else None,
    }
    journal.finalize(report)
    if inference_runtime.policy is not None:
        inference_runtime.policy.close()
    return report


def main():
    parser = argparse.ArgumentParser(description="Paper Trader - eToro-first live simulation")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Chemin du modele ou dossier (requis pour les familles PPO live)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="simulate",
        choices=["simulate", "alpaca", "etoro"],
        help="Mode broker a utiliser",
    )
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config YAML")
    parser.add_argument("--api-key", type=str, default=None, help="Alpaca API key (compat)")
    parser.add_argument("--api-secret", type=str, default=None, help="Alpaca API secret (compat)")
    parser.add_argument(
        "--balance",
        type=float,
        default=DEFAULT_INITIAL_BALANCE,
        help=f"Balance initiale (defaut: ${DEFAULT_INITIAL_BALANCE:,.0f})",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help=f"Intervalle entre checks (minutes, defaut config/live ou {TRADE_INTERVAL_MINUTES})",
    )
    parser.add_argument("--max-hours", type=float, default=24, help="Duree max du paper trading")
    parser.add_argument(
        "--buy-pct",
        type=float,
        default=None,
        help="Override du pourcentage d'achat par trade",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Nombre max d'iterations (utile pour tests et smoke runs)",
    )
    args = parser.parse_args()

    run_paper_trading(
        model_path=args.model,
        mode=args.mode,
        api_key=args.api_key,
        api_secret=args.api_secret,
        initial_balance=args.balance,
        interval_minutes=args.interval,
        max_hours=args.max_hours,
        buy_pct=args.buy_pct,
        config_path=args.config,
        max_iterations=args.max_iterations,
    )


if __name__ == "__main__":
    main()
