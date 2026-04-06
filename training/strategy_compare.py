"""Common bake-off engine for comparing trading strategy families."""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from config.schema import validate_config
from core.evidence_hardening import (
    aggregate_evidence_status,
    evaluate_backtest_artifact,
    evaluate_robustness_artifact,
    evaluate_walk_forward_artifact,
    reconcile_equity,
)
from core.promotion_gate import (
    evaluate_robustness_promotion,
    evaluate_walk_forward_promotion,
    promotion_thresholds_from_config,
)
from training.strategy_policies import (
    OHLCV_COLUMNS,
    SUPPORTED_STRATEGY_FAMILIES,
    StrategyContext,
    _estimate_round_trip_cost,
    build_strategy_policy,
)

DEFAULT_COMPARE_OUTPUT_DIR = Path("logs") / "strategy_compare"


def load_compare_config(config_path: str) -> dict:
    """Load and validate the compare configuration."""

    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    warnings = validate_config(config)
    for warning in warnings:
        print(f"[strategy_compare] warning: {warning}")
    return config


def resolve_strategy_settings(config: dict) -> dict:
    """Resolve strategy-bakeoff defaults from YAML config."""

    strategy_cfg = dict(config.get("strategy", {}))
    raw_seed_offsets = strategy_cfg.get("seed_offsets", [0])
    seed_offsets: list[int] = []
    for offset in raw_seed_offsets:
        normalized = int(offset)
        if normalized not in seed_offsets:
            seed_offsets.append(normalized)
    if not seed_offsets:
        seed_offsets = [0]
    return {
        "family": strategy_cfg.get("family", "ppo_ensemble"),
        "candidate_families": list(
            strategy_cfg.get(
                "candidate_families",
                ["ppo_single", "ppo_ensemble", "recurrent_ppo", "supervised_ranker", "rule_momentum_regime"],
            )
        ),
        "phase2_interval": strategy_cfg.get("phase2_interval", "4h"),
        "phase2_top_k": int(strategy_cfg.get("phase2_top_k", 2)),
        "monte_carlo_sims": int(strategy_cfg.get("monte_carlo_sims", 20)),
        "monte_carlo_noise_std": float(strategy_cfg.get("monte_carlo_noise_std", 0.005)),
        "extreme_return_threshold": float(strategy_cfg.get("extreme_return_threshold", 5.0)),
        "seed_offsets": seed_offsets,
    }


def _runtime_imports() -> dict:
    from core.data_fetcher import download_data
    from core.environment import TradingEnv
    from core.features import FeatureEngineer
    from core.macro_data import MacroDataFetcher

    return {
        "download_data": download_data,
        "TradingEnv": TradingEnv,
        "FeatureEngineer": FeatureEngineer,
        "MacroDataFetcher": MacroDataFetcher,
    }


def _annualization_factor(interval: str) -> float:
    factors = {
        "1h": 252 * 6.5,
        "4h": 252 * (6.5 / 4),
        "1d": 252,
        "1wk": 52,
        "1mo": 12,
    }
    return float(factors.get(interval, 252))


def generate_walk_forward_splits(
    data: dict[str, pd.DataFrame],
    *,
    train_years: int,
    test_months: int,
    step_months: int,
    embargo_months: int = 1,
) -> list[dict]:
    """Generate walk-forward splits on raw market data."""

    ref_ticker = next(iter(data))
    ref_df = data[ref_ticker]
    start_date = ref_df.index[0]
    end_date = ref_df.index[-1]

    splits: list[dict] = []
    train_end = start_date + pd.DateOffset(years=train_years)

    while True:
        test_start = train_end + pd.DateOffset(months=embargo_months)
        test_end = test_start + pd.DateOffset(months=test_months)
        if test_end > end_date:
            break

        train_slice: dict[str, pd.DataFrame] = {}
        test_slice: dict[str, pd.DataFrame] = {}
        for ticker, df in data.items():
            train_mask = (df.index >= start_date) & (df.index < train_end)
            test_mask = (df.index >= test_start) & (df.index < test_end)
            current_train = df.loc[train_mask]
            current_test = df.loc[test_mask]
            if len(current_train) < 100 or len(current_test) < 50:
                continue
            train_slice[ticker] = current_train.copy()
            test_slice[ticker] = current_test.copy()

        if train_slice and test_slice and len(train_slice) == len(data):
            splits.append(
                {
                    "train": train_slice,
                    "test": test_slice,
                    "train_start": str(start_date.date()),
                    "train_end": str(train_end.date()),
                    "test_start": str(test_start.date()),
                    "test_end": str(test_end.date()),
                }
            )

        train_end += pd.DateOffset(months=step_months)

    return splits


def _select_locked_feature_columns(
    train_frames: dict[str, pd.DataFrame],
    max_features_per_ticker: int,
) -> list[str]:
    ref_ticker = next(iter(train_frames))
    ref_df = train_frames[ref_ticker]
    feature_columns = [
        col
        for col in ref_df.columns
        if col not in OHLCV_COLUMNS and pd.api.types.is_numeric_dtype(ref_df[col])
    ]
    if max_features_per_ticker > 0 and len(feature_columns) > max_features_per_ticker:
        variances = ref_df[feature_columns].var().fillna(0.0)
        feature_columns = variances.nlargest(max_features_per_ticker).index.tolist()
    return feature_columns


def _lock_feature_frame(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    base_columns = [col for col in df.columns if col in OHLCV_COLUMNS]
    ordered_columns = base_columns + [col for col in feature_columns if col in df.columns]
    locked = df.loc[:, ordered_columns].copy()
    return locked.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _precompute_fold_features(
    split: dict,
    *,
    feature_engineer,
    max_features_per_ticker: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], list[str]]:
    fold_train: dict[str, pd.DataFrame] = {}
    fold_test: dict[str, pd.DataFrame] = {}

    for ticker in split["train"]:
        train_df = split["train"][ticker]
        test_df = split["test"][ticker]

        train_index = train_df.index
        feat_train = feature_engineer.calculate_all_features(train_df.copy())
        if len(feat_train) == len(train_index):
            feat_train.index = train_index
        fold_train[ticker] = feat_train

        test_index = test_df.index
        feat_test = feature_engineer.calculate_all_features(test_df.copy())
        if len(feat_test) == len(test_index):
            feat_test.index = test_index
        fold_test[ticker] = feat_test

    feature_columns = _select_locked_feature_columns(fold_train, max_features_per_ticker)
    locked_train = {
        ticker: _lock_feature_frame(df, feature_columns) for ticker, df in fold_train.items()
    }
    locked_test = {
        ticker: _lock_feature_frame(df, feature_columns) for ticker, df in fold_test.items()
    }
    return locked_train, locked_test, feature_columns


def load_market_bundle(
    config: dict,
    *,
    interval: str | None = None,
    max_workers: int = 3,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None]:
    """Load ticker OHLCV data plus aligned macro data."""

    runtime = _runtime_imports()
    resolved_interval = interval or config.get("data", {}).get("interval", "1h")
    data = runtime["download_data"](
        tickers=config["data"]["tickers"],
        period=config["data"].get("period", "5y"),
        interval=resolved_interval,
        max_workers=max_workers,
        dataset_path=config["data"].get("dataset_path"),
    )
    if not data:
        raise RuntimeError("No market data available for strategy compare")

    ref_ticker = next(iter(data))
    ref_df = data[ref_ticker]
    macro_fetcher = runtime["MacroDataFetcher"]()
    try:
        macro_data = macro_fetcher.fetch_all(
            start_date=str(ref_df.index[0].date()),
            end_date=str(ref_df.index[-1].date()),
            interval=resolved_interval,
        )
    except Exception:
        macro_data = pd.DataFrame()
    if macro_data is not None and macro_data.empty:
        macro_data = None
    return data, macro_data


def build_protocol_snapshot(
    config: dict,
    *,
    interval: str,
    splits: list[dict],
    families: list[str],
    strategy_settings: dict,
) -> dict:
    """Create an auditable snapshot of the evaluation protocol."""

    live_cfg = config.get("live", {})
    env_cfg = config.get("environment", {})
    return {
        "tickers": list(config.get("data", {}).get("tickers", [])),
        "interval": interval,
        "families": list(families),
        "n_splits": len(splits),
        "train_years": int(config.get("walk_forward", {}).get("train_years", 3)),
        "test_months": int(config.get("walk_forward", {}).get("test_months", 6)),
        "step_months": int(config.get("walk_forward", {}).get("step_months", 6)),
        "buy_pct": float(env_cfg.get("buy_pct", 0.2)),
        "max_position_pct": float(env_cfg.get("max_position_pct", 0.25)),
        "stop_loss_pct": float(env_cfg.get("stop_loss_pct", 0.0)),
        "max_open_positions": int(live_cfg.get("max_open_positions", len(config.get("data", {}).get("tickers", [])))),
        "min_confidence": float(live_cfg.get("min_confidence", 0.0)),
        "max_cost_pct": float(live_cfg.get("max_cost_pct", 1.0)),
        "promotion_thresholds": promotion_thresholds_from_config(config),
        "phase2_interval": strategy_settings["phase2_interval"],
        "phase2_top_k": strategy_settings["phase2_top_k"],
        "monte_carlo_sims": strategy_settings["monte_carlo_sims"],
        "monte_carlo_noise_std": strategy_settings["monte_carlo_noise_std"],
        "seed_offsets": list(strategy_settings.get("seed_offsets", [0])),
        "n_seed_variants": len(strategy_settings.get("seed_offsets", [0])),
    }


def _macro_row(test_env, current_step: int) -> dict[str, float] | None:
    if not getattr(test_env, "macro_columns", None) or getattr(test_env, "macro_array", None) is None:
        return None
    if current_step >= len(test_env.macro_array):
        return None
    row = test_env.macro_array[current_step]
    return {
        column: float(row[idx])
        for idx, column in enumerate(test_env.macro_columns)
    }


def _regime_is_risk_on(test_env, current_step: int, config: dict) -> bool:
    live_cfg = config.get("live", {})
    fast_span = int(live_cfg.get("regime_fast_ma", 20))
    slow_span = int(live_cfg.get("regime_slow_ma", 50))
    vix_threshold = float(live_cfg.get("regime_vix_threshold", 30.0))

    spy_df = test_env.processed_data.get("SPY")
    if spy_df is None:
        return True
    close = spy_df["Close"].iloc[: current_step + 1]
    if len(close) < slow_span:
        return False
    fast_ema = float(close.ewm(span=fast_span, adjust=False).mean().iloc[-1])
    slow_ema = float(close.ewm(span=slow_span, adjust=False).mean().iloc[-1])

    macro_row = _macro_row(test_env, current_step) or {}
    vix_value = float(macro_row.get("vix", 0.0))
    return fast_ema > slow_ema and (vix_value <= 0.0 or vix_value < vix_threshold)


def _apply_common_filters(
    actions: np.ndarray,
    *,
    test_env,
    regime_risk_on: bool,
    config: dict,
) -> tuple[np.ndarray, dict[str, int]]:
    live_cfg = config.get("live", {})
    env_cfg = config.get("environment", {})
    buy_pct = float(live_cfg.get("buy_pct", env_cfg.get("buy_pct", 0.2)))
    max_position_pct = float(
        live_cfg.get("max_position_pct", env_cfg.get("max_position_pct", 0.25))
    )
    max_open_positions = int(live_cfg.get("max_open_positions", len(test_env.tickers)))
    order_min_notional = float(live_cfg.get("order_min_notional", 0.0))
    max_cost_pct = float(live_cfg.get("max_cost_pct", 1.0))
    estimated_cost_pct = float(_estimate_round_trip_cost(config))

    filtered = np.asarray(actions, dtype=np.int64).copy()
    rejections = {
        "regime_blocked": 0,
        "max_positions_blocked": 0,
        "position_limit_blocked": 0,
        "cost_blocked": 0,
        "notional_blocked": 0,
    }

    for idx, ticker in enumerate(test_env.tickers):
        if filtered[idx] != 1:
            continue

        if not regime_risk_on:
            filtered[idx] = 0
            rejections["regime_blocked"] += 1
            continue

        current_price = test_env._get_current_price(ticker)
        if current_price <= 0:
            filtered[idx] = 0
            rejections["cost_blocked"] += 1
            continue

        open_positions = sum(1 for qty in test_env.portfolio.values() if qty > 1e-8)
        if test_env.portfolio.get(ticker, 0.0) <= 1e-8 and open_positions >= max_open_positions:
            filtered[idx] = 0
            rejections["max_positions_blocked"] += 1
            continue

        position_value = test_env.portfolio.get(ticker, 0.0) * current_price
        position_pct = position_value / max(test_env.equity, 1e-8)
        if position_pct >= max_position_pct:
            filtered[idx] = 0
            rejections["position_limit_blocked"] += 1
            continue

        planned_notional = min(test_env.balance * buy_pct, test_env.equity * max_position_pct)
        if planned_notional < order_min_notional:
            filtered[idx] = 0
            rejections["notional_blocked"] += 1
            continue

        if estimated_cost_pct > max_cost_pct:
            filtered[idx] = 0
            rejections["cost_blocked"] += 1

    return filtered, rejections


def run_policy_backtest(
    policy,
    test_data: dict[str, pd.DataFrame],
    macro_data: pd.DataFrame | None,
    config: dict,
    *,
    interval: str,
    deterministic: bool = True,
) -> dict:
    """Backtest one fitted strategy policy on one fold."""

    del deterministic

    runtime = _runtime_imports()
    env_kwargs = {k: v for k, v in config.get("environment", {}).items()}
    env_kwargs["mode"] = "backtest"
    env_kwargs["seed"] = 42
    env_kwargs["features_precomputed"] = True
    test_env = runtime["TradingEnv"](data=test_data, macro_data=macro_data, **env_kwargs)

    obs, _ = test_env.reset()
    policy.reset_state()
    done = False
    equity_curve = [float(test_env.initial_balance)]
    ref_ticker = next(iter(test_data))
    timestamps = [test_data[ref_ticker].index[min(test_env.current_step, len(test_data[ref_ticker]) - 1)]]
    max_equity_error = 0.0
    aggregated_rejections = {
        "regime_blocked": 0,
        "max_positions_blocked": 0,
        "position_limit_blocked": 0,
        "cost_blocked": 0,
        "notional_blocked": 0,
    }

    while not done:
        current_prices = {
            ticker: float(test_env._get_current_price(ticker)) for ticker in test_env.tickers
        }
        regime_risk_on = _regime_is_risk_on(test_env, test_env.current_step, config)
        context = StrategyContext(
            observation=np.asarray(obs, dtype=np.float32),
            current_step=int(test_env.current_step),
            tickers=list(test_env.tickers),
            prices=current_prices,
            portfolio=dict(test_env.portfolio),
            entry_prices=dict(test_env.entry_prices),
            processed_data=test_env.processed_data,
            feature_columns=list(test_env.feature_columns),
            macro_columns=list(test_env.macro_columns),
            macro_row=_macro_row(test_env, test_env.current_step),
            interval=interval,
            regime_risk_on=regime_risk_on,
        )
        raw_actions = np.asarray(policy.predict_actions(context), dtype=np.int64)
        actions, rejections = _apply_common_filters(
            raw_actions,
            test_env=test_env,
            regime_risk_on=regime_risk_on,
            config=config,
        )
        for key, value in rejections.items():
            aggregated_rejections[key] += int(value)

        obs, _reward, done, _truncated, info = test_env.step(actions)
        reconciliation = reconcile_equity(
            balance=float(test_env.balance),
            positions=test_env.portfolio,
            prices=current_prices,
            reported_equity=float(test_env.equity),
        )
        max_equity_error = max(max_equity_error, float(reconciliation["error"]))
        equity_curve.append(float(info["equity"]))
        current_idx = min(test_env.current_step, len(test_data[ref_ticker]) - 1)
        timestamps.append(test_data[ref_ticker].index[current_idx])

    equity_series = pd.Series(equity_curve, index=pd.Index(timestamps[: len(equity_curve)]))
    returns = equity_series.pct_change().dropna()
    total_return = (equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0]
    max_drawdown = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max()
    sharpe_ratio = 0.0
    if len(returns) > 1 and returns.std() > 0:
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(_annualization_factor(interval))

    daily_equity = equity_series.groupby(pd.to_datetime(equity_series.index).normalize()).last()
    daily_returns = daily_equity.pct_change().dropna()
    max_daily_loss = abs(float(daily_returns.min())) if not daily_returns.empty and daily_returns.min() < 0 else 0.0
    closed_trades = info.get("winning_trades", 0) + info.get("losing_trades", 0)
    win_rate = info.get("winning_trades", 0) / closed_trades if closed_trades > 0 else 0.0

    metrics = {
        "total_return": float(total_return),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown),
        "max_daily_loss": float(max_daily_loss),
        "total_trades": int(info.get("total_trades", 0)),
        "winning_trades": int(info.get("winning_trades", 0)),
        "losing_trades": int(info.get("losing_trades", 0)),
        "win_rate": float(win_rate),
        "final_equity": float(equity_series.iloc[-1]),
        "n_steps": int(len(equity_curve)),
        "rejections": aggregated_rejections,
        "accounting": {"max_equity_error": float(max_equity_error)},
    }
    return metrics


def _add_price_noise(
    data: dict[str, pd.DataFrame],
    *,
    noise_std: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    noisy_data: dict[str, pd.DataFrame] = {}
    for ticker, df in data.items():
        noisy_df = df.copy()
        for column in ("Open", "High", "Low", "Close"):
            if column in noisy_df.columns:
                noise = rng.normal(0.0, noise_std, size=len(noisy_df))
                noisy_df[column] = noisy_df[column] * (1 + noise)
        if "High" in noisy_df.columns and {"Open", "Close"}.issubset(noisy_df.columns):
            noisy_df["High"] = noisy_df[["Open", "High", "Close"]].max(axis=1)
        if "Low" in noisy_df.columns and {"Open", "Close"}.issubset(noisy_df.columns):
            noisy_df["Low"] = noisy_df[["Open", "Low", "Close"]].min(axis=1)
        noisy_data[ticker] = noisy_df.clip(lower=0.01)
    return noisy_data


def _simulate_crash(
    data: dict[str, pd.DataFrame],
    *,
    crash_pct: float = -0.20,
) -> dict[str, pd.DataFrame]:
    crashed: dict[str, pd.DataFrame] = {}
    for ticker, df in data.items():
        crashed_df = df.copy()
        crash_start = max(len(crashed_df) // 2, 0)
        crash_end = min(crash_start + 6, len(crashed_df))
        for column in ("Open", "High", "Low", "Close"):
            if column not in crashed_df.columns:
                continue
            for offset, row_idx in enumerate(range(crash_start, crash_end)):
                progress = (offset + 1) / max(crash_end - crash_start, 1)
                crashed_df.iloc[row_idx, crashed_df.columns.get_loc(column)] *= 1 + (crash_pct * progress)
            crashed_df.iloc[crash_end:, crashed_df.columns.get_loc(column)] *= 1 + crash_pct
        crashed[ticker] = crashed_df.clip(lower=0.01)
    return crashed


def run_policy_monte_carlo(
    policy,
    test_data: dict[str, pd.DataFrame],
    macro_data: pd.DataFrame | None,
    config: dict,
    *,
    interval: str,
    n_sims: int,
    noise_std: float,
    seed: int,
) -> dict:
    """Run Monte Carlo price-noise simulations for one fitted policy."""

    simulations = []
    for sim_idx in range(n_sims):
        noisy_test = _add_price_noise(test_data, noise_std=noise_std, seed=seed + sim_idx)
        metrics = run_policy_backtest(policy, noisy_test, macro_data, config, interval=interval)
        simulations.append(metrics)

    returns = [float(item["total_return"]) for item in simulations]
    sharpes = [float(item["sharpe_ratio"]) for item in simulations]
    drawdowns = [float(item["max_drawdown"]) for item in simulations]
    return {
        "n_sims": int(n_sims),
        "noise_std": float(noise_std),
        "deterministic": True,
        "avg_return": float(np.mean(returns)) if returns else 0.0,
        "median_return": float(np.median(returns)) if returns else 0.0,
        "std_return": float(np.std(returns)) if returns else 0.0,
        "min_return": float(min(returns)) if returns else 0.0,
        "max_return": float(max(returns)) if returns else 0.0,
        "p5_return": float(np.percentile(returns, 5)) if returns else 0.0,
        "p95_return": float(np.percentile(returns, 95)) if returns else 0.0,
        "avg_sharpe": float(np.mean(sharpes)) if sharpes else 0.0,
        "avg_max_drawdown": float(np.mean(drawdowns)) if drawdowns else 0.0,
        "loss_rate": float(sum(1 for value in returns if value <= 0) / len(returns)) if returns else 1.0,
        "is_overfit": False,
    }


def run_policy_stress_test(
    policy,
    test_data: dict[str, pd.DataFrame],
    macro_data: pd.DataFrame | None,
    config: dict,
    *,
    interval: str,
    crash_pct: float = -0.20,
) -> dict:
    """Run a crash stress test for one fitted policy."""

    baseline = run_policy_backtest(policy, test_data, macro_data, config, interval=interval)
    crashed = _simulate_crash(test_data, crash_pct=crash_pct)
    crash_metrics = run_policy_backtest(policy, crashed, macro_data, config, interval=interval)
    return {
        "crash_pct": float(crash_pct),
        "baseline_return": float(baseline["total_return"]),
        "crash_return": float(crash_metrics["total_return"]),
        "return_impact": float(crash_metrics["total_return"] - baseline["total_return"]),
        "baseline_max_drawdown": float(baseline["max_drawdown"]),
        "crash_max_drawdown": float(crash_metrics["max_drawdown"]),
        "survives": bool(crash_metrics["max_drawdown"] < 0.50),
        "acceptable_drawdown": bool(crash_metrics["max_drawdown"] < 0.25),
    }


def aggregate_walk_forward_metrics(
    folds: list[dict],
    *,
    config: dict,
    interval: str,
    extreme_return_threshold: float,
) -> dict:
    """Aggregate fold metrics into walk-forward summary + evidence."""

    returns = [float(fold["total_return"]) for fold in folds]
    sharpes = [float(fold["sharpe_ratio"]) for fold in folds]
    drawdowns = [float(fold["max_drawdown"]) for fold in folds]
    max_daily_losses = [float(fold.get("max_daily_loss", 0.0)) for fold in folds]
    win_rates = [float(fold.get("win_rate", 0.0)) for fold in folds]

    thresholds = promotion_thresholds_from_config(config)
    evidence = evaluate_walk_forward_artifact(
        folds,
        interval=interval,
        initial_balance=float(config.get("environment", {}).get("initial_balance", 0.0)),
        extreme_return_threshold=extreme_return_threshold,
    )
    promotion_gate = evaluate_walk_forward_promotion(
        returns=returns,
        sharpes=sharpes,
        drawdowns=drawdowns,
        thresholds=thresholds,
    )
    cumulative_return = float(np.prod([1 + value for value in returns]) - 1) if returns else 0.0
    return {
        "n_folds": int(len(folds)),
        "avg_return": float(np.mean(returns)) if returns else 0.0,
        "avg_sharpe": float(np.mean(sharpes)) if sharpes else 0.0,
        "avg_max_drawdown": float(np.mean(drawdowns)) if drawdowns else 0.0,
        "avg_win_rate": float(np.mean(win_rates)) if win_rates else 0.0,
        "worst_fold_return": float(min(returns)) if returns else 0.0,
        "best_fold_return": float(max(returns)) if returns else 0.0,
        "cumulative_return": cumulative_return,
        "win_fold_ratio": float(sum(1 for value in returns if value > 0) / len(returns)) if returns else 0.0,
        "avg_max_daily_loss": float(np.mean(max_daily_losses)) if max_daily_losses else 0.0,
        "worst_max_daily_loss": float(max(max_daily_losses)) if max_daily_losses else 0.0,
        "promotion_gate": promotion_gate,
        "evidence": evidence,
        "folds": folds,
    }


def aggregate_robustness_metrics(
    monte_carlo_reports: list[dict],
    stress_reports: list[dict],
    *,
    config: dict,
) -> dict:
    """Aggregate robustness reports across folds."""

    thresholds = promotion_thresholds_from_config(config)
    if monte_carlo_reports:
        avg_mc = {
            "avg_return": float(np.mean([item["avg_return"] for item in monte_carlo_reports])),
            "avg_sharpe": float(np.mean([item["avg_sharpe"] for item in monte_carlo_reports])),
            "avg_max_drawdown": float(np.mean([item["avg_max_drawdown"] for item in monte_carlo_reports])),
            "loss_rate": float(np.mean([item["loss_rate"] for item in monte_carlo_reports])),
            "noise_std": float(np.mean([item["noise_std"] for item in monte_carlo_reports])),
            "std_return": float(np.mean([item["std_return"] for item in monte_carlo_reports])),
            "min_return": float(min(item["min_return"] for item in monte_carlo_reports)),
            "max_return": float(max(item["max_return"] for item in monte_carlo_reports)),
            "p5_return": float(np.mean([item["p5_return"] for item in monte_carlo_reports])),
            "p95_return": float(np.mean([item["p95_return"] for item in monte_carlo_reports])),
            "median_return": float(np.mean([item["median_return"] for item in monte_carlo_reports])),
            "deterministic": all(bool(item.get("deterministic", True)) for item in monte_carlo_reports),
            "is_overfit": any(bool(item.get("is_overfit", False)) for item in monte_carlo_reports),
        }
    else:
        avg_mc = {
            "avg_return": 0.0,
            "avg_sharpe": 0.0,
            "avg_max_drawdown": 0.0,
            "loss_rate": 1.0,
            "noise_std": 0.0,
            "std_return": 0.0,
            "min_return": 0.0,
            "max_return": 0.0,
            "p5_return": 0.0,
            "p95_return": 0.0,
            "median_return": 0.0,
            "deterministic": True,
            "is_overfit": False,
        }

    evidence = evaluate_robustness_artifact({"monte_carlo": avg_mc})
    promotion_gate = evaluate_robustness_promotion(avg_mc, thresholds=thresholds)
    return {
        "monte_carlo": avg_mc,
        "stress_test": {
            "crash_max_drawdown": float(np.mean([item["crash_max_drawdown"] for item in stress_reports])) if stress_reports else 0.0,
            "worst_crash_max_drawdown": float(max(item["crash_max_drawdown"] for item in stress_reports)) if stress_reports else 0.0,
            "survival_ratio": float(np.mean([1.0 if item["survives"] else 0.0 for item in stress_reports])) if stress_reports else 0.0,
            "acceptable_drawdown_ratio": float(np.mean([1.0 if item["acceptable_drawdown"] else 0.0 for item in stress_reports])) if stress_reports else 0.0,
        },
        "promotion_gate": promotion_gate,
        "evidence": evidence,
    }


def _score_forward(value: float, bad: float, good: float) -> float:
    if value <= bad:
        return 0.0
    if value >= good:
        return 1.0
    return float((value - bad) / max(good - bad, 1e-8))


def _score_reverse(value: float, good: float, bad: float) -> float:
    if value <= good:
        return 1.0
    if value >= bad:
        return 0.0
    return float(1 - ((value - good) / max(bad - good, 1e-8)))


def calculate_selection_score(candidate: dict, thresholds: dict) -> float:
    """Composite 0-100 selection score oriented around profit and risk."""

    walk_forward = candidate["walk_forward"]
    robustness = candidate["robustness"]
    monte_carlo = robustness["monte_carlo"]
    stress_test = robustness["stress_test"]

    score = 0.0
    score += 25 * _score_forward(
        walk_forward["cumulative_return"],
        bad=-0.10,
        good=max(0.12, thresholds["cumulative_return_min"] + 0.12),
    )
    score += 20 * _score_forward(
        walk_forward["avg_sharpe"],
        bad=0.0,
        good=max(1.2, thresholds["sharpe_min"]),
    )
    score += 15 * _score_reverse(
        walk_forward["avg_max_drawdown"],
        good=min(0.08, thresholds["max_drawdown"]),
        bad=max(0.20, thresholds["max_drawdown"] * 1.75),
    )
    score += 10 * _score_forward(
        walk_forward["win_fold_ratio"],
        bad=0.40,
        good=max(0.75, thresholds["win_fold_ratio_min"]),
    )
    score += 15 * _score_reverse(
        monte_carlo["loss_rate"],
        good=min(0.10, thresholds["loss_rate_max"]),
        bad=max(0.50, thresholds["loss_rate_max"] * 2),
    )
    score += 15 * _score_reverse(
        stress_test["worst_crash_max_drawdown"],
        good=min(0.10, thresholds["max_drawdown"]),
        bad=max(0.35, thresholds["max_drawdown"] * 2.5),
    )
    return round(float(score), 2)


def _candidate_verdict(candidate: dict, thresholds: dict) -> str:
    walk_forward = candidate["walk_forward"]
    robustness = candidate["robustness"]
    evidence_status = candidate["evidence_status"]
    worst_daily_loss = float(walk_forward.get("worst_max_daily_loss", 1.0))
    demo_ready = (
        evidence_status != "suspect"
        and walk_forward["promotion_gate"]["passed"]
        and robustness["promotion_gate"]["passed"]
        and worst_daily_loss <= thresholds["max_daily_loss"]
    )
    if demo_ready:
        return "promotable_demo"
    if evidence_status == "suspect":
        return "suspect_artifact"
    if walk_forward["cumulative_return"] <= thresholds["cumulative_return_min"]:
        return "improve_edge"
    if walk_forward["avg_max_drawdown"] > thresholds["max_drawdown"] or worst_daily_loss > thresholds["max_daily_loss"]:
        return "improve_risk_profile"
    if robustness["monte_carlo"]["loss_rate"] > thresholds["loss_rate_max"]:
        return "improve_robustness"
    return "needs_iteration"


def evaluate_candidate_family(
    family: str,
    *,
    config: dict,
    splits: list[dict],
    macro_data: pd.DataFrame | None,
    interval: str,
    monte_carlo_sims: int,
    monte_carlo_noise_std: float,
    extreme_return_threshold: float,
    seed_offsets: list[int] | None = None,
) -> dict:
    """Train, evaluate, and score one candidate family across all folds."""

    runtime = _runtime_imports()
    feature_engineer = runtime["FeatureEngineer"]()
    thresholds = promotion_thresholds_from_config(config)
    max_features = int(config.get("environment", {}).get("max_features_per_ticker", 0))
    resolved_seed_offsets = [int(offset) for offset in (seed_offsets or [0])]

    fold_metrics: list[dict] = []
    monte_carlo_reports: list[dict] = []
    stress_reports: list[dict] = []
    policy_metadata: list[dict] = []

    for fold_idx, split in enumerate(splits):
        train_data, test_data, feature_columns = _precompute_fold_features(
            split,
            feature_engineer=feature_engineer,
            max_features_per_ticker=max_features,
        )
        for seed_offset in resolved_seed_offsets:
            policy_seed = 42 + (fold_idx * 1000) + int(seed_offset)
            fold_policy = build_strategy_policy(family, copy.deepcopy(config), seed=policy_seed)
            fold_policy.fit(train_data, macro_data)
            metrics = run_policy_backtest(
                fold_policy,
                test_data,
                macro_data,
                config,
                interval=interval,
            )
            metrics["fold_idx"] = int(fold_idx)
            metrics["seed_offset"] = int(seed_offset)
            metrics["seed"] = int(policy_seed)
            metrics["train_period"] = f"{split['train_start']}->{split['train_end']}"
            metrics["test_period"] = f"{split['test_start']}->{split['test_end']}"
            metrics["feature_count"] = int(len(feature_columns))
            metrics["evidence"] = evaluate_backtest_artifact(
                metrics,
                interval=interval,
                test_period=metrics["test_period"],
                accounting=metrics.get("accounting"),
                initial_balance=float(config.get("environment", {}).get("initial_balance", 0.0)),
                extreme_return_threshold=extreme_return_threshold,
            )
            fold_metrics.append(metrics)

            metadata = fold_policy.artifact_metadata()
            metadata["fold_idx"] = int(fold_idx)
            metadata["seed_offset"] = int(seed_offset)
            metadata["seed"] = int(policy_seed)
            policy_metadata.append(metadata)

            monte_carlo_reports.append(
                run_policy_monte_carlo(
                    fold_policy,
                    test_data,
                    macro_data,
                    config,
                    interval=interval,
                    n_sims=monte_carlo_sims,
                    noise_std=monte_carlo_noise_std,
                    seed=(fold_idx * 10_000) + (int(seed_offset) * 100),
                )
            )
            stress_reports.append(
                run_policy_stress_test(
                    fold_policy,
                    test_data,
                    macro_data,
                    config,
                    interval=interval,
                )
            )
            fold_policy.close()

    walk_forward = aggregate_walk_forward_metrics(
        fold_metrics,
        config=config,
        interval=interval,
        extreme_return_threshold=extreme_return_threshold,
    )
    robustness = aggregate_robustness_metrics(monte_carlo_reports, stress_reports, config=config)
    evidence_status = aggregate_evidence_status(
        [walk_forward["evidence"]["status"], robustness["evidence"]["status"]]
    )

    candidate = {
        "family": family,
        "interval": interval,
        "walk_forward": walk_forward,
        "robustness": robustness,
        "policy_metadata": policy_metadata,
        "evidence_status": evidence_status,
        "seed_offsets": resolved_seed_offsets,
    }
    candidate["selection_score"] = calculate_selection_score(candidate, thresholds)
    candidate["verdict"] = _candidate_verdict(candidate, thresholds)
    return candidate


def _select_phase2_candidates(phase1_candidates: list[dict], top_k: int) -> list[str]:
    ranked = sorted(
        phase1_candidates,
        key=lambda item: (
            item["evidence_status"] == "suspect",
            -item["selection_score"],
        ),
    )
    return [candidate["family"] for candidate in ranked[:top_k]]


def _beats_baseline(candidate: dict, baseline: dict) -> bool:
    return (
        candidate["walk_forward"]["cumulative_return"] > baseline["walk_forward"]["cumulative_return"]
        and candidate["walk_forward"]["avg_sharpe"] >= baseline["walk_forward"]["avg_sharpe"]
        and candidate["walk_forward"]["avg_max_drawdown"] <= baseline["walk_forward"]["avg_max_drawdown"]
        and candidate["walk_forward"]["win_fold_ratio"] >= baseline["walk_forward"]["win_fold_ratio"]
        and candidate["robustness"]["monte_carlo"]["loss_rate"] <= baseline["robustness"]["monte_carlo"]["loss_rate"]
        and candidate["robustness"]["stress_test"]["worst_crash_max_drawdown"]
        <= baseline["robustness"]["stress_test"]["worst_crash_max_drawdown"]
        and candidate["evidence_status"] != "suspect"
    )


def compare_strategy_families(
    *,
    config_path: str = "config/config.yaml",
    output_dir: str | None = None,
    candidate_families: list[str] | None = None,
    phase2_top_k: int | None = None,
) -> dict:
    """Run the full model-family bake-off and return a leaderboard."""

    config = load_compare_config(config_path)
    strategy_settings = resolve_strategy_settings(config)
    families = list(candidate_families or strategy_settings["candidate_families"])
    invalid_families = [family for family in families if family not in SUPPORTED_STRATEGY_FAMILIES]
    if invalid_families:
        raise ValueError(f"Unsupported candidate families: {invalid_families}")

    phase2_k = int(phase2_top_k or strategy_settings["phase2_top_k"])
    current_interval = str(config.get("data", {}).get("interval", "1h"))

    data, macro_data = load_market_bundle(config, interval=current_interval)
    wf_cfg = config.get("walk_forward", {})
    splits = generate_walk_forward_splits(
        data,
        train_years=int(wf_cfg.get("train_years", 3)),
        test_months=int(wf_cfg.get("test_months", 6)),
        step_months=int(wf_cfg.get("step_months", 6)),
    )
    if not splits:
        raise RuntimeError("No walk-forward splits available for strategy comparison")

    protocol = build_protocol_snapshot(
        config,
        interval=current_interval,
        splits=splits,
        families=families,
        strategy_settings=strategy_settings,
    )
    phase1 = [
        evaluate_candidate_family(
            family,
            config=config,
            splits=splits,
            macro_data=macro_data,
            interval=current_interval,
            monte_carlo_sims=strategy_settings["monte_carlo_sims"],
            monte_carlo_noise_std=strategy_settings["monte_carlo_noise_std"],
            extreme_return_threshold=strategy_settings["extreme_return_threshold"],
            seed_offsets=strategy_settings["seed_offsets"],
        )
        for family in families
    ]
    phase1 = sorted(phase1, key=lambda item: item["selection_score"], reverse=True)

    phase2_families = _select_phase2_candidates(phase1, phase2_k)
    phase2_interval = strategy_settings["phase2_interval"]
    phase2_data, phase2_macro = load_market_bundle(config, interval=phase2_interval)
    phase2_config = copy.deepcopy(config)
    phase2_config.setdefault("data", {})["interval"] = phase2_interval
    phase2_splits = generate_walk_forward_splits(
        phase2_data,
        train_years=int(wf_cfg.get("train_years", 3)),
        test_months=int(wf_cfg.get("test_months", 6)),
        step_months=int(wf_cfg.get("step_months", 6)),
    )
    phase2 = []
    if phase2_splits:
        phase2 = [
            evaluate_candidate_family(
                family,
                config=phase2_config,
                splits=phase2_splits,
                macro_data=phase2_macro,
                interval=phase2_interval,
                monte_carlo_sims=strategy_settings["monte_carlo_sims"],
                monte_carlo_noise_std=strategy_settings["monte_carlo_noise_std"],
                extreme_return_threshold=min(strategy_settings["extreme_return_threshold"], 3.0),
                seed_offsets=strategy_settings["seed_offsets"],
            )
            for family in phase2_families
        ]
        phase2 = sorted(phase2, key=lambda item: item["selection_score"], reverse=True)

    threshold_cfg = promotion_thresholds_from_config(config)
    baseline = next((candidate for candidate in phase1 if candidate["family"] == "ppo_ensemble"), None)
    winning_pool = phase2 or phase1
    winner = winning_pool[0] if winning_pool else None
    winner_family = winner["family"] if winner else None
    winner_beats_baseline = bool(winner and baseline and _beats_baseline(winner, baseline))

    leaderboard = {
        "created_at": datetime.now().isoformat(),
        "config_path": str(config_path),
        "protocol": protocol,
        "phase_1": phase1,
        "phase_2": phase2,
        "selection": {
            "winner_family": winner_family,
            "winner_interval": winner["interval"] if winner else None,
            "winner_selection_score": winner["selection_score"] if winner else None,
            "winner_verdict": winner["verdict"] if winner else None,
            "winner_beats_ppo_ensemble": winner_beats_baseline,
            "baseline_family": baseline["family"] if baseline else None,
            "baseline_selection_score": baseline["selection_score"] if baseline else None,
            "recommended_next_iteration": None if winner and winner["verdict"] == "promotable_demo" else "position_sizing_universe_timeframe_cost_sensitivity",
            "promotion_thresholds": threshold_cfg,
        },
    }

    if output_dir:
        save_leaderboard(leaderboard, output_dir)
    return leaderboard


def save_leaderboard(leaderboard: dict, output_dir: str | Path) -> Path:
    """Persist the strategy leaderboard JSON and return its path."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / "strategy_leaderboard.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(leaderboard, handle, indent=2, default=str)
    return output_path
