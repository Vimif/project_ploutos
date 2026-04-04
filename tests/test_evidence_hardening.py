from core.evidence_hardening import (
    aggregate_evidence_status,
    evaluate_backtest_artifact,
    evaluate_robustness_artifact,
    evaluate_walk_forward_artifact,
)


def test_backtest_artifact_flags_extreme_return_without_reconciliation():
    result = evaluate_backtest_artifact(
        {
            "total_return": 8.0,
            "final_equity": 900_000,
            "total_trades": 2,
            "winning_trades": 1,
            "losing_trades": 0,
        },
        interval="1h",
        test_period="2025-01-01->2025-06-01",
        initial_balance=100_000,
    )

    assert result["status"] == "suspect"
    assert "extreme 1h fold return was not reconciled" in result["reasons"]


def test_walk_forward_and_robustness_evidence_aggregate_cleanly():
    walk_forward = evaluate_walk_forward_artifact(
        [
            {
                "fold_idx": 0,
                "test_period": "2025-01-01->2025-06-01",
                "total_return": 0.12,
                "final_equity": 112_000,
                "total_trades": 10,
                "winning_trades": 6,
                "losing_trades": 4,
                "accounting": {"max_equity_error": 0.0},
            }
        ],
        interval="1h",
        initial_balance=100_000,
    )
    robustness = evaluate_robustness_artifact(
        {
            "monte_carlo": {
                "noise_std": 0.005,
                "deterministic": True,
                "std_return": 0.01,
                "min_return": -0.05,
                "median_return": 0.03,
                "max_return": 0.08,
                "p5_return": -0.03,
                "p95_return": 0.06,
            }
        }
    )

    assert walk_forward["status"] == "validated"
    assert robustness["status"] == "validated"
    assert aggregate_evidence_status([walk_forward["status"], robustness["status"]]) == "validated"
