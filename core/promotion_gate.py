"""Promotion gate helpers for training and robustness reports."""

from __future__ import annotations

from typing import Iterable

DEFAULT_PROMOTION_THRESHOLDS = {
    "max_drawdown": 0.12,
    "max_daily_loss": 0.03,
    "sharpe_min": 0.8,
    "win_fold_ratio_min": 0.60,
    "cumulative_return_min": 0.0,
    "loss_rate_max": 0.20,
}


def promotion_thresholds_from_config(config: dict | None = None) -> dict:
    """Build promotion thresholds from config with sensible defaults."""

    live_cfg = (config or {}).get("live", {})
    thresholds = dict(DEFAULT_PROMOTION_THRESHOLDS)
    thresholds["max_drawdown"] = float(live_cfg.get("max_drawdown", thresholds["max_drawdown"]))
    thresholds["max_daily_loss"] = float(
        live_cfg.get("max_daily_loss", thresholds["max_daily_loss"])
    )
    thresholds["sharpe_min"] = float(live_cfg.get("promotion_sharpe_min", thresholds["sharpe_min"]))
    thresholds["win_fold_ratio_min"] = float(
        live_cfg.get("promotion_win_fold_ratio_min", thresholds["win_fold_ratio_min"])
    )
    thresholds["cumulative_return_min"] = float(
        live_cfg.get(
            "promotion_cumulative_return_min",
            thresholds["cumulative_return_min"],
        )
    )
    thresholds["loss_rate_max"] = float(
        live_cfg.get("promotion_loss_rate_max", thresholds["loss_rate_max"])
    )
    return thresholds


def evaluate_walk_forward_promotion(
    returns: Iterable[float],
    sharpes: Iterable[float],
    drawdowns: Iterable[float],
    thresholds: dict | None = None,
) -> dict:
    """Evaluate promotion readiness from walk-forward metrics."""

    thresholds = thresholds or dict(DEFAULT_PROMOTION_THRESHOLDS)
    returns = list(returns)
    sharpes = list(sharpes)
    drawdowns = list(drawdowns)

    if not returns:
        return {
            "passed": False,
            "reason": "no_folds",
            "thresholds": thresholds,
            "metrics": {},
            "checks": {},
            "pending_checks": ["demo_soak_sync", "daily_loss_limit"],
        }

    avg_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0.0
    avg_max_drawdown = sum(drawdowns) / len(drawdowns) if drawdowns else 0.0
    cumulative_return = 1.0
    for ret in returns:
        cumulative_return *= 1 + ret
    cumulative_return -= 1
    win_fold_ratio = sum(1 for ret in returns if ret > 0) / len(returns)

    metrics = {
        "avg_sharpe": avg_sharpe,
        "avg_max_drawdown": avg_max_drawdown,
        "cumulative_return": cumulative_return,
        "win_fold_ratio": win_fold_ratio,
    }
    checks = {
        "max_drawdown": avg_max_drawdown <= thresholds["max_drawdown"],
        "avg_sharpe": avg_sharpe >= thresholds["sharpe_min"],
        "win_fold_ratio": win_fold_ratio >= thresholds["win_fold_ratio_min"],
        "cumulative_return": cumulative_return > thresholds["cumulative_return_min"],
    }
    return {
        "passed": all(checks.values()),
        "thresholds": thresholds,
        "metrics": metrics,
        "checks": checks,
        "pending_checks": ["demo_soak_sync", "daily_loss_limit"],
    }


def evaluate_robustness_promotion(report: dict, thresholds: dict | None = None) -> dict:
    """Evaluate whether robustness outputs are compatible with promotion."""

    thresholds = thresholds or dict(DEFAULT_PROMOTION_THRESHOLDS)
    avg_return = float(report.get("avg_return", 0.0))
    avg_sharpe = float(report.get("avg_sharpe", 0.0))
    avg_max_drawdown = float(report.get("avg_max_drawdown", 0.0))
    loss_rate = float(report.get("loss_rate", 1.0))

    metrics = {
        "avg_return": avg_return,
        "avg_sharpe": avg_sharpe,
        "avg_max_drawdown": avg_max_drawdown,
        "loss_rate": loss_rate,
    }
    checks = {
        "avg_return": avg_return > thresholds["cumulative_return_min"],
        "avg_sharpe": avg_sharpe >= thresholds["sharpe_min"],
        "avg_max_drawdown": avg_max_drawdown <= thresholds["max_drawdown"],
        "loss_rate": loss_rate <= thresholds["loss_rate_max"],
    }
    return {
        "passed": all(checks.values()) and not report.get("is_overfit", False),
        "thresholds": thresholds,
        "metrics": metrics,
        "checks": checks,
    }
