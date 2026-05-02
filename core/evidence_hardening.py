"""Evidence hardening helpers for model and artifact audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

DEFAULT_EXTREME_RETURN_THRESHOLD = {
    "1h": 5.0,
    "4h": 3.0,
    "1d": 2.0,
}
DEFAULT_MONTE_CARLO_MIN_STD = 1e-6
EVIDENCE_STATUSES = ("validated", "warning", "suspect")


@dataclass
class EvidenceResult:
    """Structured audit result for one artifact."""

    status: str
    checks: dict[str, bool]
    reasons: list[str]
    warnings: list[str]
    details: dict

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "checks": dict(self.checks),
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "details": dict(self.details),
        }


def aggregate_evidence_status(statuses: list[str]) -> str:
    """Aggregate multiple evidence statuses into one."""

    normalized = [status for status in statuses if status in EVIDENCE_STATUSES]
    if not normalized:
        return "warning"
    if "suspect" in normalized:
        return "suspect"
    if "warning" in normalized:
        return "warning"
    return "validated"


def reconcile_equity(
    balance: float,
    positions: dict[str, float],
    prices: dict[str, float],
    reported_equity: float,
) -> dict:
    """Reconcile cash + positions against the reported equity."""

    marked_to_market = sum(
        float(positions.get(symbol, 0.0)) * float(prices.get(symbol, 0.0)) for symbol in positions
    )
    reconciled_equity = float(balance) + float(marked_to_market)
    error = abs(float(reported_equity) - reconciled_equity)
    return {
        "reported_equity": float(reported_equity),
        "reconciled_equity": float(reconciled_equity),
        "marked_to_market": float(marked_to_market),
        "error": float(error),
    }


def _parse_test_period_days(test_period: Optional[str]) -> Optional[int]:
    if not test_period or "->" not in test_period:
        return None
    start_raw, end_raw = test_period.split("->", 1)
    start = pd.Timestamp(start_raw)
    end = pd.Timestamp(end_raw)
    return max(int((end - start).days), 0)


def evaluate_backtest_artifact(
    metrics: dict,
    *,
    interval: str = "1h",
    test_period: Optional[str] = None,
    accounting: Optional[dict] = None,
    initial_balance: Optional[float] = None,
    extreme_return_threshold: Optional[float] = None,
    equity_tolerance: float = 1e-4,
) -> dict:
    """Evaluate whether a backtest artifact looks internally credible."""

    checks: dict[str, bool] = {}
    reasons: list[str] = []
    warnings: list[str] = []
    details = {
        "interval": interval,
        "test_period": test_period,
        "test_days": _parse_test_period_days(test_period),
    }

    total_trades = int(metrics.get("total_trades", 0))
    winning_trades = int(metrics.get("winning_trades", 0))
    losing_trades = int(metrics.get("losing_trades", 0))
    total_return = float(metrics.get("total_return", 0.0))
    final_equity = float(metrics.get("final_equity", 0.0))

    closed_trades = winning_trades + losing_trades
    checks["closed_trades_le_total_trades"] = closed_trades <= total_trades
    if not checks["closed_trades_le_total_trades"]:
        reasons.append("winning_trades + losing_trades exceeds total_trades")

    if initial_balance is not None and initial_balance > 0:
        expected_final_equity = float(initial_balance) * (1 + total_return)
        final_equity_error = abs(expected_final_equity - final_equity)
        details["final_equity_error"] = final_equity_error
        checks["final_equity_matches_return"] = final_equity_error <= max(
            equity_tolerance,
            initial_balance * 1e-6,
        )
        if not checks["final_equity_matches_return"]:
            reasons.append("final_equity is inconsistent with initial_balance and total_return")
    else:
        checks["final_equity_matches_return"] = True

    accounting_error = None
    if accounting:
        accounting_error = float(accounting.get("max_equity_error", 0.0))
        details["max_equity_error"] = accounting_error
        checks["equity_reconciled"] = accounting_error <= equity_tolerance
        if not checks["equity_reconciled"]:
            reasons.append("cash + positions does not reconcile with reported equity")
    else:
        checks["equity_reconciled"] = False
        warnings.append("accounting reconciliation was not provided")

    threshold = extreme_return_threshold
    if threshold is None:
        threshold = DEFAULT_EXTREME_RETURN_THRESHOLD.get(interval, 5.0)
    details["extreme_return_threshold"] = float(threshold)

    test_days = details["test_days"]
    extreme_return = (
        interval == "1h"
        and test_days is not None
        and test_days <= 220
        and abs(total_return) > threshold
    )
    checks["extreme_return_requires_reconciliation"] = not (
        extreme_return and not checks["equity_reconciled"]
    )
    if extreme_return and not checks["extreme_return_requires_reconciliation"]:
        reasons.append("extreme 1h fold return was not reconciled")
    elif extreme_return:
        warnings.append("extreme 1h fold return observed but accounting reconciliation passed")

    if closed_trades == 0 and abs(total_return) > 0.20:
        checks["large_return_without_closed_trades"] = False
        reasons.append("large return reported without any closed trades")
    else:
        checks["large_return_without_closed_trades"] = True

    if reasons:
        status = "suspect"
    elif warnings:
        status = "warning"
    else:
        status = "validated"

    return EvidenceResult(
        status=status,
        checks=checks,
        reasons=reasons,
        warnings=warnings,
        details=details,
    ).to_dict()


def evaluate_walk_forward_artifact(
    folds: list[dict],
    *,
    interval: str = "1h",
    initial_balance: Optional[float] = None,
    extreme_return_threshold: Optional[float] = None,
) -> dict:
    """Aggregate evidence checks across walk-forward folds."""

    fold_results = []
    statuses = []
    for fold in folds:
        result = evaluate_backtest_artifact(
            fold,
            interval=interval,
            test_period=fold.get("test_period"),
            accounting=fold.get("accounting"),
            initial_balance=initial_balance,
            extreme_return_threshold=extreme_return_threshold,
        )
        fold_results.append(
            {
                "fold_idx": fold.get("fold_idx"),
                "status": result["status"],
                "reasons": result["reasons"],
                "warnings": result["warnings"],
            }
        )
        statuses.append(result["status"])

    overall_status = aggregate_evidence_status(statuses)
    return {
        "status": overall_status,
        "folds": fold_results,
        "suspect_folds": [fold["fold_idx"] for fold in fold_results if fold["status"] == "suspect"],
        "warning_folds": [fold["fold_idx"] for fold in fold_results if fold["status"] == "warning"],
    }


def evaluate_robustness_artifact(
    report: dict,
    *,
    min_std: float = DEFAULT_MONTE_CARLO_MIN_STD,
) -> dict:
    """Evaluate whether a robustness report looks internally credible."""

    checks: dict[str, bool] = {}
    reasons: list[str] = []
    warnings: list[str] = []

    monte_carlo = report.get("monte_carlo", report)
    if not monte_carlo:
        return EvidenceResult(
            status="warning",
            checks={"monte_carlo_present": False},
            reasons=[],
            warnings=["robustness report does not contain monte_carlo metrics"],
            details={},
        ).to_dict()

    deterministic = bool(monte_carlo.get("deterministic", True))
    noise_std = float(monte_carlo.get("noise_std", 0.0))
    std_return = float(monte_carlo.get("std_return", 0.0))
    min_return = float(monte_carlo.get("min_return", 0.0))
    max_return = float(monte_carlo.get("max_return", 0.0))
    p5_return = float(monte_carlo.get("p5_return", 0.0))
    p95_return = float(monte_carlo.get("p95_return", 0.0))
    median_return = float(monte_carlo.get("median_return", 0.0))

    checks["return_range_ordered"] = min_return <= median_return <= max_return
    if not checks["return_range_ordered"]:
        reasons.append("monte_carlo return range is internally inconsistent")

    checks["percentiles_ordered"] = p5_return <= median_return <= p95_return
    if not checks["percentiles_ordered"]:
        reasons.append("monte_carlo percentiles are internally inconsistent")

    if noise_std > 0 and not deterministic:
        checks["non_zero_dispersion_under_noise"] = std_return > min_std
        if not checks["non_zero_dispersion_under_noise"]:
            reasons.append(
                "monte_carlo dispersion is near zero despite noise and stochastic policy"
            )
    else:
        checks["non_zero_dispersion_under_noise"] = True
        if noise_std > 0 and std_return <= min_std:
            warnings.append("monte_carlo dispersion is near zero; inspect policy determinism")

    status = "suspect" if reasons else "warning" if warnings else "validated"
    return EvidenceResult(
        status=status,
        checks=checks,
        reasons=reasons,
        warnings=warnings,
        details={
            "deterministic": deterministic,
            "noise_std": noise_std,
            "std_return": std_return,
            "min_std": min_std,
        },
    ).to_dict()
