#!/usr/bin/env python3
"""Audit profitability and robustness from walk-forward and robustness outputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.promotion_gate import (
    evaluate_robustness_promotion,
    evaluate_walk_forward_promotion,
    promotion_thresholds_from_config,
)


def load_runtime_config(config_path: Optional[str]) -> dict:
    """Load YAML config when available."""

    if not config_path:
        return {}

    path = Path(config_path)
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def resolve_walk_forward_path(path_str: str) -> Path:
    """Resolve a walk-forward results file from a file or directory."""

    path = Path(path_str)
    if path.is_file():
        return path

    candidate = path / "walk_forward_results.json"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"walk_forward_results.json not found in {path}")


def resolve_robustness_paths(path_str: Optional[str]) -> list[Path]:
    """Resolve one or many robustness report files."""

    if not path_str:
        return []

    path = Path(path_str)
    if path.is_file():
        return [path]

    candidates = []
    direct = path / "robustness_report.json"
    if direct.exists():
        candidates.append(direct)

    candidates.extend(sorted(path.glob("fold_*/robustness_report.json")))
    if not candidates:
        candidates.extend(sorted(path.glob("**/robustness_report.json")))

    unique: list[Path] = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(candidate)
    return unique


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_median(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _downside_std(values: list[float]) -> float:
    negatives = [value for value in values if value < 0]
    return _safe_std(negatives)


def aggregate_walk_forward(results: dict, thresholds: dict) -> dict:
    """Aggregate profitability metrics from walk-forward outputs."""

    folds = list(results.get("folds") or [])
    returns = [float(fold.get("total_return", 0.0)) for fold in folds]
    sharpes = [float(fold.get("sharpe_ratio", 0.0)) for fold in folds]
    drawdowns = [float(fold.get("max_drawdown", 0.0)) for fold in folds]
    win_rates = [float(fold.get("win_rate", 0.0)) for fold in folds]

    if not folds:
        returns = [float(results.get("avg_return", 0.0))]
        sharpes = [float(results.get("avg_sharpe", 0.0))]
        drawdowns = [float(results.get("avg_max_drawdown", 0.0))]
        if "avg_win_rate" in results:
            win_rates = [float(results.get("avg_win_rate", 0.0))]

    gate = evaluate_walk_forward_promotion(
        returns=returns,
        sharpes=sharpes,
        drawdowns=drawdowns,
        thresholds=thresholds,
    )

    cumulative_return = float(results.get("cumulative_return", 0.0))
    if folds:
        cumulative_return = float(np.prod([1 + value for value in returns]) - 1)

    avg_max_drawdown = float(results.get("avg_max_drawdown", _safe_mean(drawdowns)))
    return_to_drawdown = (
        cumulative_return / avg_max_drawdown
        if avg_max_drawdown > 0
        else math.inf if cumulative_return > 0 else 0.0
    )

    return {
        "source": "walk_forward",
        "n_folds": len(folds) or int(results.get("n_folds", 1)),
        "avg_return": float(results.get("avg_return", _safe_mean(returns))),
        "median_return": _safe_median(returns),
        "std_return": _safe_std(returns),
        "downside_std_return": _downside_std(returns),
        "best_fold_return": max(returns) if returns else 0.0,
        "worst_fold_return": min(returns) if returns else 0.0,
        "avg_sharpe": float(results.get("avg_sharpe", _safe_mean(sharpes))),
        "avg_max_drawdown": avg_max_drawdown,
        "avg_win_rate": float(results.get("avg_win_rate", _safe_mean(win_rates))),
        "cumulative_return": cumulative_return,
        "win_fold_ratio": float(
            results.get(
                "win_fold_ratio",
                sum(1 for value in returns if value > 0) / len(returns) if returns else 0.0,
            )
        ),
        "positive_fold_count": sum(1 for value in returns if value > 0),
        "negative_fold_count": sum(1 for value in returns if value < 0),
        "return_to_drawdown": float(return_to_drawdown),
        "promotion_gate": gate,
        "folds": folds,
    }


def aggregate_robustness(reports: list[dict], thresholds: dict) -> Optional[dict]:
    """Aggregate one or many robustness reports into a profitability view."""

    if not reports:
        return None

    monte_carlo = [report.get("monte_carlo") for report in reports if report.get("monte_carlo")]
    stress_tests = [report.get("stress_test") for report in reports if report.get("stress_test")]

    robustness_gate = None
    if monte_carlo:
        avg_mc = {
            "avg_return": _safe_mean([float(item.get("avg_return", 0.0)) for item in monte_carlo]),
            "avg_sharpe": _safe_mean([float(item.get("avg_sharpe", 0.0)) for item in monte_carlo]),
            "avg_max_drawdown": _safe_mean(
                [float(item.get("avg_max_drawdown", 0.0)) for item in monte_carlo]
            ),
            "loss_rate": _safe_mean([float(item.get("loss_rate", 1.0)) for item in monte_carlo]),
            "is_overfit": any(bool(item.get("is_overfit", False)) for item in monte_carlo),
        }
        robustness_gate = evaluate_robustness_promotion(avg_mc, thresholds=thresholds)

    return {
        "source": "robustness",
        "n_reports": len(reports),
        "n_monte_carlo_reports": len(monte_carlo),
        "n_stress_reports": len(stress_tests),
        "avg_loss_rate": _safe_mean([float(item.get("loss_rate", 1.0)) for item in monte_carlo]),
        "worst_loss_rate": max(
            [float(item.get("loss_rate", 1.0)) for item in monte_carlo],
            default=0.0,
        ),
        "avg_mc_return": _safe_mean([float(item.get("avg_return", 0.0)) for item in monte_carlo]),
        "p5_mc_return": min(
            [float(item.get("p5_return", 0.0)) for item in monte_carlo],
            default=0.0,
        ),
        "avg_mc_sharpe": _safe_mean([float(item.get("avg_sharpe", 0.0)) for item in monte_carlo]),
        "avg_mc_drawdown": _safe_mean(
            [float(item.get("avg_max_drawdown", 0.0)) for item in monte_carlo]
        ),
        "overfit_ratio": _safe_mean(
            [1.0 if item.get("is_overfit", False) else 0.0 for item in monte_carlo]
        ),
        "crash_survival_ratio": _safe_mean(
            [1.0 if item.get("survives", False) else 0.0 for item in stress_tests]
        ),
        "acceptable_crash_drawdown_ratio": _safe_mean(
            [1.0 if item.get("acceptable_drawdown", False) else 0.0 for item in stress_tests]
        ),
        "worst_crash_drawdown": max(
            [float(item.get("crash_max_drawdown", 0.0)) for item in stress_tests],
            default=0.0,
        ),
        "promotion_gate": robustness_gate,
    }


def _score_forward(value: float, bad: float, good: float) -> float:
    if value <= bad:
        return 0.0
    if value >= good:
        return 1.0
    return float((value - bad) / (good - bad))


def _score_reverse(value: float, good: float, bad: float) -> float:
    if value <= good:
        return 1.0
    if value >= bad:
        return 0.0
    return float(1 - ((value - good) / (bad - good)))


def calculate_profitability_score(
    walk_forward: dict,
    robustness: Optional[dict],
    thresholds: dict,
) -> float:
    """Map profitability and robustness to a 0-100 score."""

    parts = {
        "cumulative_return": 25
        * _score_forward(
            walk_forward["cumulative_return"],
            bad=-0.10,
            good=max(0.12, thresholds["cumulative_return_min"] + 0.12),
        ),
        "avg_sharpe": 20
        * _score_forward(
            walk_forward["avg_sharpe"],
            bad=0.0,
            good=max(1.2, thresholds["sharpe_min"]),
        ),
        "win_fold_ratio": 15
        * _score_forward(
            walk_forward["win_fold_ratio"],
            bad=0.40,
            good=max(0.75, thresholds["win_fold_ratio_min"]),
        ),
        "avg_max_drawdown": 15
        * _score_reverse(
            walk_forward["avg_max_drawdown"],
            good=min(0.08, thresholds["max_drawdown"]),
            bad=max(0.20, thresholds["max_drawdown"] * 1.5),
        ),
        "return_to_drawdown": 10
        * _score_forward(
            walk_forward["return_to_drawdown"],
            bad=0.0,
            good=1.0,
        ),
    }

    if robustness:
        parts["loss_rate"] = 10 * _score_reverse(
            robustness["avg_loss_rate"],
            good=0.05,
            bad=max(0.35, thresholds["loss_rate_max"] * 1.5),
        )
        parts["crash_drawdown"] = 5 * _score_forward(
            robustness["acceptable_crash_drawdown_ratio"],
            bad=0.0,
            good=1.0,
        )
    else:
        parts["loss_rate"] = 0.0
        parts["crash_drawdown"] = 0.0

    return float(sum(parts.values()))


def build_findings(
    walk_forward: dict,
    robustness: Optional[dict],
    thresholds: dict,
) -> list[dict]:
    """Produce prioritized profitability findings."""

    findings = []

    if walk_forward["cumulative_return"] <= thresholds["cumulative_return_min"]:
        findings.append(
            {
                "severity": "critical",
                "code": "negative_cumulative_return",
                "message": "Le walk-forward ne montre pas encore d'edge rentable net.",
            }
        )
    if walk_forward["avg_sharpe"] < thresholds["sharpe_min"]:
        findings.append(
            {
                "severity": "high",
                "code": "low_sharpe",
                "message": "Le rendement est trop faible par rapport a la volatilite prise.",
            }
        )
    if walk_forward["avg_max_drawdown"] > thresholds["max_drawdown"]:
        findings.append(
            {
                "severity": "high",
                "code": "drawdown_too_high",
                "message": "Le drawdown moyen depasse le niveau de promotion vise.",
            }
        )
    if walk_forward["win_fold_ratio"] < thresholds["win_fold_ratio_min"]:
        findings.append(
            {
                "severity": "high",
                "code": "inconsistent_folds",
                "message": "La profitabilite n'est pas assez stable d'un fold a l'autre.",
            }
        )
    if walk_forward["std_return"] > max(abs(walk_forward["avg_return"]) * 1.5, 0.08):
        findings.append(
            {
                "severity": "medium",
                "code": "high_return_dispersion",
                "message": "Les retours par fold sont trop disperses, ce qui fragilise la confiance.",
            }
        )
    if robustness is None:
        findings.append(
            {
                "severity": "medium",
                "code": "missing_robustness",
                "message": "Aucun rapport de robustesse n'a ete fourni, la rentabilite n'est pas encore stressee.",
            }
        )
        return findings

    if robustness["avg_loss_rate"] > thresholds["loss_rate_max"]:
        findings.append(
            {
                "severity": "critical",
                "code": "overfit_loss_rate",
                "message": "Les simulations Monte Carlo perdent trop souvent, ce qui suggere de l'overfit.",
            }
        )
    if robustness["overfit_ratio"] > 0:
        findings.append(
            {
                "severity": "high",
                "code": "overfit_detected",
                "message": "Au moins un rapport de robustesse signale un comportement sur-optimise.",
            }
        )
    if robustness["acceptable_crash_drawdown_ratio"] < 1.0:
        findings.append(
            {
                "severity": "high",
                "code": "crash_resilience_gap",
                "message": "La strategie encaisse mal les scenarios de krach simules.",
            }
        )
    if robustness["p5_mc_return"] < -0.10:
        findings.append(
            {
                "severity": "medium",
                "code": "weak_left_tail",
                "message": "La queue gauche des simulations reste trop penalisante pour un deploiement serein.",
            }
        )
    return findings


def build_recommendations(
    walk_forward: dict,
    robustness: Optional[dict],
    config: dict,
    thresholds: dict,
) -> list[str]:
    """Generate concrete next experiments based on profitability gaps."""

    live_cfg = config.get("live", {})
    recommendations = []

    if walk_forward["cumulative_return"] <= thresholds["cumulative_return_min"]:
        new_conf = min(float(live_cfg.get("min_confidence", 0.67)) + 0.05, 0.85)
        recommendations.append(
            "Tester un filtre d'entree plus selectif pour remonter l'edge net, "
            f"par exemple `live.min_confidence={new_conf:.2f}`."
        )
    if walk_forward["win_fold_ratio"] < thresholds["win_fold_ratio_min"]:
        recommendations.append(
            "Comparer plusieurs jeux de splits walk-forward et retirer les actifs qui cassent "
            "la regularite inter-fold au lieu d'optimiser sur le fold gagnant."
        )
    if walk_forward["avg_max_drawdown"] > thresholds["max_drawdown"]:
        buy_pct = float(live_cfg.get("buy_pct", 0.15))
        max_positions = int(live_cfg.get("max_open_positions", 5))
        recommendations.append(
            "Reduire le risque unitaire pour ameliorer le couple profit/drawdown, "
            f"par exemple `buy_pct={max(buy_pct * 0.75, 0.05):.3f}` "
            f"et `max_open_positions={max(max_positions - 1, 1)}`."
        )
    if walk_forward["avg_sharpe"] < thresholds["sharpe_min"]:
        recommendations.append(
            "Lancer un A/B test de reward et de couts de transaction pour verifier si la "
            "strategie gagne avant frais mais s'erode apres friction."
        )
    if robustness and robustness["avg_loss_rate"] > thresholds["loss_rate_max"]:
        recommendations.append(
            "Allonger la fenetre d'entrainement walk-forward ou reduire le nombre de features "
            "afin de limiter l'overfit mesure par Monte Carlo."
        )
    if robustness and robustness["acceptable_crash_drawdown_ratio"] < 1.0:
        recommendations.append(
            "Durcir le filtre de regime et les exits de risque avant de chercher plus de rendement, "
            "car la strategie reste trop vulnerable en stress market."
        )
    if not recommendations:
        recommendations.append(
            "La profitabilite est coherente avec les seuils actuels. La prochaine etape logique "
            "est un soak test demo plus long avec suivi de drift et de desynchronisation."
        )
    return recommendations


def determine_verdict(walk_forward: dict, robustness: Optional[dict]) -> str:
    """Map the audit to an operational verdict."""

    wf_passed = bool(walk_forward["promotion_gate"]["passed"])
    rb_passed = robustness is not None and robustness.get("promotion_gate", {}).get("passed", False)

    if wf_passed and rb_passed:
        return "ready_for_demo_soak"
    if walk_forward["cumulative_return"] <= 0 or walk_forward["win_fold_ratio"] < 0.5:
        return "improve_edge"
    if walk_forward["avg_max_drawdown"] > 0.12 or (
        robustness and robustness["acceptable_crash_drawdown_ratio"] < 1.0
    ):
        return "improve_risk_profile"
    if robustness is None:
        return "needs_robustness_validation"
    return "needs_iteration"


def build_profitability_report(
    walk_forward_results: dict,
    robustness_reports: list[dict],
    config: dict,
) -> dict:
    """Create a complete profitability audit from project artifacts."""

    thresholds = promotion_thresholds_from_config(config)
    walk_forward = aggregate_walk_forward(walk_forward_results, thresholds)
    robustness = aggregate_robustness(robustness_reports, thresholds)
    score = calculate_profitability_score(walk_forward, robustness, thresholds)

    return {
        "created_at": datetime.now().isoformat(),
        "thresholds": thresholds,
        "profitability_score": score,
        "verdict": determine_verdict(walk_forward, robustness),
        "walk_forward": walk_forward,
        "robustness": robustness,
        "findings": build_findings(walk_forward, robustness, thresholds),
        "recommendations": build_recommendations(walk_forward, robustness, config, thresholds),
    }


def save_report(report: dict, output_path: Optional[str]) -> Path:
    """Persist the audit as JSON."""

    path = (
        Path(output_path)
        if output_path
        else Path("logs/profitability") / f"audit_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return path


def print_summary(report: dict, output_path: Path) -> None:
    """Print a compact profitability summary."""

    walk = report["walk_forward"]
    robustness = report.get("robustness")

    print("=" * 72)
    print("PLOUTOS PROFITABILITY AUDIT")
    print("=" * 72)
    print(
        f"Verdict={report['verdict']} | Score={report['profitability_score']:.1f}/100 | "
        f"Folds={walk['n_folds']}"
    )
    print(
        f"Cumulative={walk['cumulative_return']:+.2%} | "
        f"AvgSharpe={walk['avg_sharpe']:.2f} | "
        f"AvgMaxDD={walk['avg_max_drawdown']:.2%} | "
        f"WinFolds={walk['win_fold_ratio']:.1%}"
    )
    if robustness:
        print(
            f"MC LossRate={robustness['avg_loss_rate']:.1%} | "
            f"MC AvgReturn={robustness['avg_mc_return']:+.2%} | "
            f"Crash OK={robustness['acceptable_crash_drawdown_ratio']:.1%}"
        )
    print("")
    print("Findings:")
    for finding in report["findings"]:
        print(f"- [{finding['severity']}] {finding['message']}")
    print("")
    print("Recommendations:")
    for recommendation in report["recommendations"]:
        print(f"- {recommendation}")
    print("")
    print(f"Report saved to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit profitability from walk-forward and robustness outputs."
    )
    parser.add_argument(
        "--walk-forward",
        type=str,
        required=True,
        help="Path to walk_forward_results.json or its parent directory",
    )
    parser.add_argument(
        "--robustness",
        type=str,
        default=None,
        help="Optional robustness_report.json file or directory",
    )
    parser.add_argument("--config", type=str, default="config/config.yaml", help="YAML config path")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_runtime_config(args.config)
    walk_forward_root = Path(args.walk_forward)
    walk_forward_path = resolve_walk_forward_path(args.walk_forward)
    default_robustness_root = (
        walk_forward_root if walk_forward_root.is_dir() else walk_forward_root.parent
    )
    robustness_paths = resolve_robustness_paths(args.robustness or str(default_robustness_root))

    walk_forward_results = _load_json(walk_forward_path)
    robustness_reports = [_load_json(path) for path in robustness_paths]
    report = build_profitability_report(walk_forward_results, robustness_reports, config)
    output_path = save_report(report, args.output)
    print_summary(report, output_path)


if __name__ == "__main__":
    main()
