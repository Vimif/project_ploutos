from __future__ import annotations

import json

from scripts import profitability_audit


def make_walk_forward_results():
    return {
        "n_folds": 3,
        "avg_return": -0.01,
        "avg_sharpe": 0.35,
        "avg_max_drawdown": 0.16,
        "avg_win_rate": 0.46,
        "cumulative_return": -0.05,
        "win_fold_ratio": 1 / 3,
        "folds": [
            {"total_return": -0.04, "sharpe_ratio": 0.10, "max_drawdown": 0.18, "win_rate": 0.40},
            {"total_return": 0.02, "sharpe_ratio": 0.75, "max_drawdown": 0.10, "win_rate": 0.52},
            {"total_return": -0.03, "sharpe_ratio": 0.20, "max_drawdown": 0.20, "win_rate": 0.45},
        ],
    }


def make_robustness_report(loss_rate=0.35, acceptable_drawdown=False):
    return {
        "monte_carlo": {
            "avg_return": -0.01,
            "avg_sharpe": 0.25,
            "avg_max_drawdown": 0.17,
            "loss_rate": loss_rate,
            "p5_return": -0.14,
            "is_overfit": loss_rate > 0.20,
        },
        "stress_test": {
            "survives": True,
            "acceptable_drawdown": acceptable_drawdown,
            "crash_max_drawdown": 0.29,
        },
    }


def test_build_profitability_report_flags_edge_and_risk_failures():
    report = profitability_audit.build_profitability_report(
        walk_forward_results=make_walk_forward_results(),
        robustness_reports=[make_robustness_report()],
        config={"live": {"min_confidence": 0.67, "buy_pct": 0.15, "max_open_positions": 5}},
    )

    codes = {finding["code"] for finding in report["findings"]}

    assert report["verdict"] == "improve_edge"
    assert report["profitability_score"] < 50
    assert "negative_cumulative_return" in codes
    assert "drawdown_too_high" in codes
    assert "overfit_loss_rate" in codes


def test_cli_helpers_resolve_reports_and_save_output(tmp_path):
    walk_dir = tmp_path / "walk"
    fold_dir = walk_dir / "fold_00"
    walk_dir.mkdir()
    fold_dir.mkdir()

    walk_path = walk_dir / "walk_forward_results.json"
    robustness_path = fold_dir / "robustness_report.json"

    with open(walk_path, "w", encoding="utf-8") as handle:
        json.dump(make_walk_forward_results(), handle)
    with open(robustness_path, "w", encoding="utf-8") as handle:
        json.dump(make_robustness_report(loss_rate=0.10, acceptable_drawdown=True), handle)

    resolved_walk = profitability_audit.resolve_walk_forward_path(str(walk_dir))
    resolved_robustness = profitability_audit.resolve_robustness_paths(str(walk_dir))
    report = profitability_audit.build_profitability_report(
        walk_forward_results=profitability_audit._load_json(resolved_walk),
        robustness_reports=[profitability_audit._load_json(path) for path in resolved_robustness],
        config={},
    )
    output_path = profitability_audit.save_report(report, str(tmp_path / "audit.json"))

    assert resolved_walk == walk_path
    assert resolved_robustness == [robustness_path]
    assert output_path.exists()
