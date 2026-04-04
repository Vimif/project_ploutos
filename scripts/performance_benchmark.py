#!/usr/bin/env python3
"""Benchmark critical Ploutos runtime stages for performance investigations."""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import platform
import pstats
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_BENCHMARK_TICKERS = [
    "NVDA",
    "MSFT",
    "AAPL",
    "AMZN",
    "META",
    "TSLA",
    "SPY",
    "QQQ",
    "XLK",
    "XLF",
]


@dataclass
class BenchmarkSettings:
    """Resolved runtime settings for the performance benchmark."""

    config_path: Optional[str]
    output_path: Optional[str]
    model_path: Optional[str]
    tickers: list[str]
    periods: int
    iterations: int
    warmup_iterations: int
    initial_balance: float
    max_features_per_ticker: int
    interval: str
    interval_minutes: int
    ensemble_size: int
    include_macro: bool
    random_seed: int
    profile_stage: Optional[str] = None
    profile_output: Optional[str] = None


class SyntheticEnsemblePredictor:
    """Lightweight predictor used when no trained ensemble is provided."""

    def __init__(
        self,
        observation_size: int,
        n_assets: int,
        ensemble_size: int,
        seed: int,
    ):
        rng = np.random.default_rng(seed)
        self.weights = [
            rng.normal(loc=0.0, scale=0.05, size=(n_assets, observation_size)).astype(np.float32)
            for _ in range(max(ensemble_size, 1))
        ]
        self.bias = rng.normal(loc=0.15, scale=0.1, size=n_assets).astype(np.float32)
        if n_assets > 1:
            self.bias[1::3] -= 0.35

    def predict_with_asset_confidences(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        del deterministic

        obs = np.asarray(observation, dtype=np.float32)
        all_actions = []
        for weights in self.weights:
            scores = weights @ obs + self.bias
            actions = np.where(scores > 0.25, 1, np.where(scores < -0.35, 2, 0))
            all_actions.append(actions.astype(np.int64))

        stacked = np.asarray(all_actions, dtype=np.int64)
        final_actions = np.zeros(stacked.shape[1], dtype=np.int64)
        confidences = np.zeros(stacked.shape[1], dtype=np.float32)
        for asset_idx in range(stacked.shape[1]):
            values, counts = np.unique(stacked[:, asset_idx], return_counts=True)
            best_idx = int(np.argmax(counts))
            final_actions[asset_idx] = int(values[best_idx])
            confidences[asset_idx] = float(counts[best_idx] / len(self.weights))
        return final_actions, confidences


def load_runtime_config(config_path: Optional[str]) -> dict:
    """Load YAML config if present."""

    if not config_path:
        return {}

    path = Path(config_path)
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_settings(args: argparse.Namespace, config: dict) -> BenchmarkSettings:
    """Merge CLI overrides with runtime config defaults."""

    data_cfg = config.get("data", {})
    env_cfg = config.get("environment", {})
    live_cfg = config.get("live", {})

    tickers = (
        [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
        if args.tickers
        else list(data_cfg.get("tickers") or DEFAULT_BENCHMARK_TICKERS)
    )

    interval = str(data_cfg.get("interval", "1h"))
    interval_minutes = int(
        args.interval_minutes or live_cfg.get("interval_minutes") or _interval_to_minutes(interval)
    )

    return BenchmarkSettings(
        config_path=args.config,
        output_path=args.output,
        model_path=args.model,
        tickers=tickers,
        periods=int(args.periods),
        iterations=int(args.iterations),
        warmup_iterations=int(args.warmup),
        initial_balance=float(env_cfg.get("initial_balance", args.initial_balance)),
        max_features_per_ticker=int(
            env_cfg.get("max_features_per_ticker", args.max_features_per_ticker)
        ),
        interval=interval,
        interval_minutes=interval_minutes,
        ensemble_size=int(live_cfg.get("ensemble_size", args.ensemble_size)),
        include_macro=not bool(args.no_macro),
        random_seed=int(args.seed),
        profile_stage=args.profile_stage,
        profile_output=args.profile_output,
    )


def _interval_to_minutes(interval: str) -> int:
    normalized = interval.strip().lower()
    mapping = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
    }
    return mapping.get(normalized, 60)


def _interval_to_pandas_freq(interval: str) -> str:
    normalized = interval.strip().lower()
    mapping = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1D",
    }
    return mapping.get(normalized, "1h")


def generate_synthetic_market_data(settings: BenchmarkSettings) -> dict[str, pd.DataFrame]:
    """Create realistic synthetic OHLCV history for the benchmark."""

    rng = np.random.default_rng(settings.random_seed)
    freq = _interval_to_pandas_freq(settings.interval)
    index = pd.date_range(
        end=pd.Timestamp.utcnow().floor("h"),
        periods=settings.periods,
        freq=freq,
    )

    data: dict[str, pd.DataFrame] = {}
    for idx, ticker in enumerate(settings.tickers):
        base_price = 90 + idx * 18
        noise = rng.normal(loc=0.0006, scale=0.012, size=settings.periods)
        close = base_price * np.exp(np.cumsum(noise))
        open_ = close * (1 + rng.normal(loc=0.0, scale=0.0015, size=settings.periods))
        high = np.maximum(open_, close) * (1 + rng.uniform(0.0001, 0.003, size=settings.periods))
        low = np.minimum(open_, close) * (1 - rng.uniform(0.0001, 0.003, size=settings.periods))
        volume = rng.integers(700_000, 3_500_000, size=settings.periods).astype(float)
        data[ticker] = pd.DataFrame(
            {
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            },
            index=index,
        )
    return data


def generate_synthetic_macro_data(settings: BenchmarkSettings) -> Optional[pd.DataFrame]:
    """Create a simple macro frame aligned with the synthetic market data."""

    if not settings.include_macro:
        return None

    rng = np.random.default_rng(settings.random_seed + 11)
    freq = _interval_to_pandas_freq(settings.interval)
    index = pd.date_range(
        end=pd.Timestamp.utcnow().floor("h"),
        periods=settings.periods,
        freq=freq,
    )
    return pd.DataFrame(
        {
            "vix_close": np.clip(20 + rng.normal(0, 2.5, size=settings.periods), 12, 35),
            "tnx_close": np.clip(4 + rng.normal(0, 0.15, size=settings.periods), 2, 6),
            "dxy_close": np.clip(103 + rng.normal(0, 0.6, size=settings.periods), 97, 108),
        },
        index=index,
    )


def _detect_hardware() -> dict:
    """Read repo hardware helper lazily so tests can stub it."""

    try:
        from config.hardware import detect_hardware
    except ImportError:
        return {}
    return detect_hardware()


def _import_project_components() -> dict[str, Any]:
    """Load project runtime components lazily."""

    try:
        from core.live_observation import LiveObservationEngine
        from trading.live_execution import (
            LiveBrokerAdapter,
            LiveExecutionConfig,
            LiveExecutionEngine,
            SimulatedBroker,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Unable to import live benchmarking components. Install the project "
            f"dependencies before running this benchmark. Missing dependency: {exc}"
        ) from exc

    return {
        "LiveObservationEngine": LiveObservationEngine,
        "LiveBrokerAdapter": LiveBrokerAdapter,
        "LiveExecutionConfig": LiveExecutionConfig,
        "LiveExecutionEngine": LiveExecutionEngine,
        "SimulatedBroker": SimulatedBroker,
    }


def _load_real_predictor(
    model_path: str,
    config: dict,
    live_config_class,
) -> tuple[Any, list[str], int, dict[str, Any]]:
    """Load the real ensemble when a trained model path is provided."""

    from scripts.paper_trade import load_predictor_bundle

    live_settings = live_config_class.from_dict(config.get("live"))
    predictor, tickers, obs_size, model_config, _vecnorm_path, model_paths = load_predictor_bundle(
        model_path,
        live_settings,
        config,
    )
    return predictor, tickers, obs_size, {
        "resolved_models": [str(path) for path in model_paths],
        "model_config": model_config,
    }


def _make_positions_map(
    settings: BenchmarkSettings,
    market_data: dict[str, pd.DataFrame],
) -> dict[str, dict]:
    """Create a small portfolio state for snapshot construction."""

    positions: dict[str, dict] = {}
    for ticker in settings.tickers[: min(2, len(settings.tickers))]:
        price = float(market_data[ticker]["Close"].iloc[-1])
        qty = max((settings.initial_balance * 0.05) / max(price, 1e-6), 1.0)
        positions[ticker] = {
            "symbol": ticker,
            "qty": qty,
            "avg_entry_price": price * 0.97,
            "current_price": price,
            "market_value": qty * price,
            "cost_basis": qty * price * 0.97,
            "unrealized_pl": qty * price * 0.03,
            "unrealized_plpc": 0.03,
        }
    return positions


def measure_stage(
    name: str,
    iterations: int,
    warmup_iterations: int,
    func: Callable[[], Any],
) -> tuple[dict[str, Any], Any]:
    """Measure latency and peak allocation for one stage."""

    last_result = None
    for _ in range(max(warmup_iterations, 0)):
        last_result = func()

    durations_ms: list[float] = []
    tracemalloc.start()
    try:
        for _ in range(max(iterations, 1)):
            start = time.perf_counter()
            last_result = func()
            durations_ms.append((time.perf_counter() - start) * 1000)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    values = np.asarray(durations_ms, dtype=np.float64)
    stats = {
        "name": name,
        "iterations": int(len(values)),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "total_ms": float(values.sum()),
        "throughput_per_sec": float(1000.0 / values.mean()) if values.mean() > 0 else 0.0,
        "peak_memory_kib": float(peak / 1024),
    }
    return stats, last_result


def _profile_stage(func: Callable[[], Any], output_path: Path) -> dict[str, Any]:
    """Capture cProfile stats for a single benchmark stage."""

    profiler = cProfile.Profile()
    profiler.enable()
    func()
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
    stats.print_stats(20)
    profiler.dump_stats(str(output_path))

    return {
        "path": str(output_path),
        "top_functions": stream.getvalue(),
    }


def build_recommendations(stage_stats: dict[str, dict[str, Any]]) -> list[str]:
    """Turn the benchmark output into concrete next actions."""

    recommendations: list[str] = []
    cycle_mean = float(stage_stats["end_to_end_cycle"]["mean_ms"])
    feature_mean = float(stage_stats["feature_compute"]["mean_ms"])
    snapshot_mean = float(stage_stats["snapshot_build"]["mean_ms"])
    inference_mean = float(stage_stats["inference"]["mean_ms"])
    buy_mean = float(stage_stats["buy_path"]["mean_ms"])

    if cycle_mean > 0 and feature_mean / cycle_mean >= 0.55:
        recommendations.append(
            "Feature engineering dominates the live cycle. Prioritize incremental feature "
            "updates, cached rolling windows, or a shorter live history window."
        )
    if snapshot_mean > inference_mean * 2:
        recommendations.append(
            "Observation building is materially slower than inference. Reuse prepared arrays "
            "and avoid rebuilding unchanged portfolio or macro state on every tick."
        )
    if inference_mean >= max(feature_mean * 0.6, 1.0):
        recommendations.append(
            "Inference is a meaningful share of latency. Benchmark again with a real model and "
            "consider reducing ensemble size or moving inference to a faster device."
        )
    if buy_mean >= max(cycle_mean * 0.25, 0.5):
        recommendations.append(
            "Order path checks are heavier than expected. Inspect cost estimation and duplicate "
            "order checks before adding more live filters."
        )
    if not recommendations:
        recommendations.append(
            "No single stage dominates the cycle. Focus next on external I/O latency such as "
            "data fetching, broker round-trips, and retry behavior."
        )

    return recommendations


def run_benchmark(settings: BenchmarkSettings, config: dict) -> dict[str, Any]:
    """Execute the synthetic performance benchmark and return a structured report."""

    components = _import_project_components()
    market_data = generate_synthetic_market_data(settings)
    macro_data = generate_synthetic_macro_data(settings)
    positions_map = _make_positions_map(settings, market_data)

    live_config = components["LiveExecutionConfig"].from_dict(config.get("live"))
    live_config.ensemble_size = settings.ensemble_size
    live_config.interval_minutes = settings.interval_minutes

    observation_engine = components["LiveObservationEngine"](
        tickers=settings.tickers,
        initial_balance=settings.initial_balance,
        max_features_per_ticker=settings.max_features_per_ticker,
    )

    broker_adapter = components["LiveBrokerAdapter"](
        components["SimulatedBroker"](settings.initial_balance),
        broker_name="simulate",
        fill_timeout=getattr(live_config, "order_fill_timeout_seconds", 30),
        poll_interval=getattr(live_config, "order_poll_interval_seconds", 1.0),
    )
    execution_engine = components["LiveExecutionEngine"](broker_adapter, live_config)

    account = broker_adapter.get_account()
    balance = float(account.get("cash", settings.initial_balance))
    equity = float(account.get("portfolio_value", settings.initial_balance))

    feature_stats, _ = measure_stage(
        "feature_compute",
        settings.iterations,
        settings.warmup_iterations,
        lambda: observation_engine._compute_features(market_data),
    )
    snapshot_stats, snapshot = measure_stage(
        "snapshot_build",
        settings.iterations,
        settings.warmup_iterations,
        lambda: observation_engine.build_snapshot(
            market_data,
            positions_map,
            balance=balance,
            equity=equity,
            macro_data=macro_data,
        ),
    )

    predictor_mode = "synthetic"
    predictor_details: dict[str, Any] = {}
    predictor = SyntheticEnsemblePredictor(
        observation_size=int(snapshot.observation.shape[0]),
        n_assets=len(settings.tickers),
        ensemble_size=settings.ensemble_size,
        seed=settings.random_seed + 29,
    )

    if settings.model_path:
        try:
            real_predictor, model_tickers, model_obs_size, details = _load_real_predictor(
                settings.model_path,
                config,
                components["LiveExecutionConfig"],
            )
            if model_tickers == settings.tickers and model_obs_size == int(snapshot.observation.shape[0]):
                predictor = real_predictor
                predictor_mode = "model"
                predictor_details = details
            else:
                predictor_details = {
                    "fallback_reason": "model_shape_mismatch",
                    "expected_tickers": model_tickers,
                    "expected_observation_size": model_obs_size,
                    "actual_observation_size": int(snapshot.observation.shape[0]),
                    **details,
                }
        except Exception as exc:
            predictor_details = {
                "fallback_reason": "model_load_failed",
                "error": str(exc),
            }

    inference_stats, inference_result = measure_stage(
        "inference",
        settings.iterations,
        settings.warmup_iterations,
        lambda: predictor.predict_with_asset_confidences(
            snapshot.observation,
            deterministic=True,
        ),
    )

    regime_stats, regime = measure_stage(
        "regime_filter",
        settings.iterations,
        settings.warmup_iterations,
        lambda: execution_engine.evaluate_market_regime(market_data, macro_data),
    )

    first_symbol = settings.tickers[0]

    def _run_buy_path():
        local_adapter = components["LiveBrokerAdapter"](
            components["SimulatedBroker"](settings.initial_balance),
            broker_name="simulate",
            fill_timeout=getattr(live_config, "order_fill_timeout_seconds", 30),
            poll_interval=getattr(live_config, "order_poll_interval_seconds", 1.0),
        )
        local_engine = components["LiveExecutionEngine"](local_adapter, live_config)
        local_regime = local_engine.evaluate_market_regime(market_data, macro_data)
        return local_engine.execute_buy(
            first_symbol,
            confidence=1.0,
            price=float(snapshot.prices[first_symbol]),
            current_volume=float(snapshot.volumes[first_symbol]),
            recent_prices=snapshot.recent_prices[first_symbol],
            positions_map=local_adapter.get_positions_map(),
            equity=settings.initial_balance,
            regime=local_regime,
            reason="benchmark_buy",
        )

    buy_stats, _ = measure_stage(
        "buy_path",
        settings.iterations,
        settings.warmup_iterations,
        _run_buy_path,
    )

    def _run_cycle():
        local_observation_engine = components["LiveObservationEngine"](
            tickers=settings.tickers,
            initial_balance=settings.initial_balance,
            max_features_per_ticker=settings.max_features_per_ticker,
        )
        local_adapter = components["LiveBrokerAdapter"](
            components["SimulatedBroker"](settings.initial_balance),
            broker_name="simulate",
            fill_timeout=getattr(live_config, "order_fill_timeout_seconds", 30),
            poll_interval=getattr(live_config, "order_poll_interval_seconds", 1.0),
        )
        local_execution_engine = components["LiveExecutionEngine"](local_adapter, live_config)
        local_snapshot = local_observation_engine.build_snapshot(
            market_data,
            positions_map,
            balance=settings.initial_balance,
            equity=settings.initial_balance,
            macro_data=macro_data,
        )
        actions, confidences = predictor.predict_with_asset_confidences(
            local_snapshot.observation,
            deterministic=True,
        )
        local_regime = local_execution_engine.evaluate_market_regime(market_data, macro_data)
        for idx, symbol in enumerate(settings.tickers):
            action = int(actions[idx]) if idx < len(actions) else 0
            if action != 1:
                continue
            local_execution_engine.execute_buy(
                symbol,
                confidence=float(confidences[idx]),
                price=float(local_snapshot.prices[symbol]),
                current_volume=float(local_snapshot.volumes[symbol]),
                recent_prices=local_snapshot.recent_prices[symbol],
                positions_map=local_adapter.get_positions_map(),
                equity=settings.initial_balance,
                regime=local_regime,
                reason="benchmark_cycle",
            )
        return actions, confidences

    cycle_stats, cycle_result = measure_stage(
        "end_to_end_cycle",
        settings.iterations,
        settings.warmup_iterations,
        _run_cycle,
    )

    profile = None
    if settings.profile_stage:
        stage_map = {
            "feature_compute": lambda: observation_engine._compute_features(market_data),
            "snapshot_build": lambda: observation_engine.build_snapshot(
                market_data,
                positions_map,
                balance=balance,
                equity=equity,
                macro_data=macro_data,
            ),
            "inference": lambda: predictor.predict_with_asset_confidences(
                snapshot.observation,
                deterministic=True,
            ),
            "buy_path": _run_buy_path,
            "end_to_end_cycle": _run_cycle,
        }
        profile_func = stage_map[settings.profile_stage]
        output_path = Path(
            settings.profile_output
            or f"logs/performance/profile_{settings.profile_stage}_{datetime.now():%Y%m%d_%H%M%S}.prof"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        profile = _profile_stage(profile_func, output_path)

    stage_stats = {
        "feature_compute": feature_stats,
        "snapshot_build": snapshot_stats,
        "inference": inference_stats,
        "regime_filter": regime_stats,
        "buy_path": buy_stats,
        "end_to_end_cycle": cycle_stats,
    }

    cycle_budget_ms = settings.interval_minutes * 60 * 1000
    budget_share = cycle_stats["mean_ms"] / cycle_budget_ms if cycle_budget_ms > 0 else 0.0

    report = {
        "created_at": datetime.now().isoformat(),
        "settings": asdict(settings),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "hardware": _detect_hardware(),
        },
        "predictor_mode": predictor_mode,
        "predictor_details": predictor_details,
        "observation_size": int(snapshot.observation.shape[0]),
        "cycle_budget": {
            "interval_minutes": settings.interval_minutes,
            "mean_cycle_ms": cycle_stats["mean_ms"],
            "budget_share": float(budget_share),
        },
        "market_snapshot": {
            "n_tickers": len(settings.tickers),
            "periods": settings.periods,
            "feature_columns": len(getattr(snapshot, "feature_columns", [])),
            "macro_columns": len(getattr(snapshot, "macro_columns", [])),
        },
        "stages": stage_stats,
        "sample_outputs": {
            "actions": [int(value) for value in np.asarray(cycle_result[0]).tolist()],
            "confidences": [float(value) for value in np.asarray(cycle_result[1]).tolist()],
            "regime": {
                "risk_on": bool(getattr(regime, "risk_on", False)),
                "reason": str(getattr(regime, "reason", "")),
            },
            "inference_confidences": [
                float(value) for value in np.asarray(inference_result[1]).tolist()
            ],
        },
        "recommendations": build_recommendations(stage_stats),
    }
    if profile is not None:
        report["profile"] = profile

    return report


def save_report(report: dict[str, Any], output_path: Optional[str]) -> Path:
    """Persist the benchmark report as JSON."""

    if output_path:
        path = Path(output_path)
    else:
        path = Path("logs/performance") / f"benchmark_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return path


def print_summary(report: dict[str, Any], output_path: Path) -> None:
    """Emit a compact terminal summary."""

    print("=" * 72)
    print("PLOUTOS PERFORMANCE BENCHMARK")
    print("=" * 72)
    print(
        f"Tickers={report['market_snapshot']['n_tickers']} | "
        f"Periods={report['market_snapshot']['periods']} | "
        f"Obs={report['observation_size']} | "
        f"Predictor={report['predictor_mode']}"
    )
    print(
        f"Mean cycle={report['cycle_budget']['mean_cycle_ms']:.2f} ms | "
        f"Budget share={report['cycle_budget']['budget_share']:.6f}"
    )
    print("")
    for name, stats in report["stages"].items():
        print(
            f"{name:>16}: mean={stats['mean_ms']:.2f} ms | "
            f"p95={stats['p95_ms']:.2f} ms | "
            f"peak_mem={stats['peak_memory_kib']:.1f} KiB"
        )
    print("")
    print("Recommendations:")
    for recommendation in report["recommendations"]:
        print(f"- {recommendation}")
    print("")
    print(f"Report saved to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    """Create the benchmark CLI."""

    parser = argparse.ArgumentParser(
        description="Benchmark the main Ploutos runtime stages with synthetic market data."
    )
    parser.add_argument("--config", type=str, default="config/config.yaml", help="YAML config path")
    parser.add_argument("--model", type=str, default=None, help="Optional trained model path")
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated ticker override")
    parser.add_argument("--periods", type=int, default=720, help="Bars per synthetic ticker")
    parser.add_argument("--iterations", type=int, default=5, help="Measured iterations per stage")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations per stage")
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=100_000.0,
        help="Fallback balance when config does not define one",
    )
    parser.add_argument(
        "--max-features-per-ticker",
        type=int,
        default=30,
        help="Fallback max feature count per ticker",
    )
    parser.add_argument("--ensemble-size", type=int, default=3, help="Synthetic ensemble size")
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=None,
        help="Override the cycle budget interval in minutes",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--no-macro", action="store_true", help="Disable synthetic macro inputs")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument(
        "--profile-stage",
        type=str,
        default=None,
        choices=["feature_compute", "snapshot_build", "inference", "buy_path", "end_to_end_cycle"],
        help="Optional stage to profile with cProfile",
    )
    parser.add_argument(
        "--profile-output",
        type=str,
        default=None,
        help="Optional .prof output path",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_runtime_config(args.config)
    settings = resolve_settings(args, config)
    try:
        report = run_benchmark(settings, config)
    except RuntimeError as exc:
        parser.exit(1, f"Benchmark setup failed: {exc}\n")
    output_path = save_report(report, settings.output_path)
    print_summary(report, output_path)


if __name__ == "__main__":
    main()
