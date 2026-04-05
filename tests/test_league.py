from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from training import league


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_config(path: Path, batch_output_root: Path) -> None:
    config = {
        "training": {
            "total_timesteps": 128,
            "n_envs": 1,
            "batch_size": 64,
            "n_steps": 64,
            "n_epochs": 1,
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
        },
        "environment": {
            "initial_balance": 100_000,
            "commission": 0.0,
            "reward_scaling": 1.0,
            "max_steps": 500,
            "buy_pct": 0.10,
            "max_position_pct": 0.10,
            "max_trades_per_day": 10,
            "min_holding_period": 1,
            "warmup_steps": 5,
            "steps_per_trading_week": 78,
            "drawdown_threshold": 0.1,
            "max_features_per_ticker": 5,
            "stop_loss_pct": 0.05,
        },
        "data": {
            "tickers": ["SPY", "QQQ"],
            "period": "3y",
            "interval": "4h",
            "dataset_path": "./data/dataset_v8/",
        },
        "walk_forward": {"train_years": 1, "test_months": 6, "step_months": 6},
        "live": {
            "min_confidence": 0.67,
            "buy_pct": 0.10,
            "max_position_pct": 0.10,
            "max_open_positions": 4,
            "max_cost_pct": 0.004,
            "regime_fast_ma": 20,
            "regime_slow_ma": 50,
            "regime_vix_threshold": 30.0,
            "max_drawdown": 0.12,
            "max_daily_loss": 0.03,
            "promotion_sharpe_min": 0.8,
            "promotion_win_fold_ratio_min": 0.60,
            "promotion_cumulative_return_min": 0.0,
            "promotion_loss_rate_max": 0.20,
        },
        "strategy": {
            "family": "rule_momentum_regime",
            "candidate_families": ["supervised_ranker", "ppo_ensemble", "rule_momentum_regime"],
            "phase2_interval": "4h",
            "phase2_top_k": 2,
            "monte_carlo_sims": 3,
            "monte_carlo_noise_std": 0.005,
            "rule_fast_ma": 20,
            "rule_slow_ma": 50,
        },
        "league": {
            "snapshot_id": "test_snapshot",
            "candidate_families": ["supervised_ranker", "ppo_ensemble", "rule_momentum_regime"],
            "baseline_family": "rule_momentum_regime",
            "gold_holdout_months": 6,
            "batch_output_root": str(batch_output_root),
            "cadence": "batch_complete_only",
            "learning_granularity": "batch_decision",
            "learning_action_mode": "advisory",
        },
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def _make_market():
    dates = pd.date_range("2022-01-01", periods=3000, freq="4h")
    return {
        "SPY": pd.DataFrame({"Close": range(3000)}, index=dates),
        "QQQ": pd.DataFrame({"Close": range(3000)}, index=dates),
    }


def _candidate_payload(family: str, interval: str, split_count: int) -> dict:
    challenge = split_count > 1
    score_map = {
        ("supervised_ranker", True): 84.0,
        ("ppo_ensemble", True): 66.0,
        ("rule_momentum_regime", True): 78.0,
        ("supervised_ranker", False): 82.0,
        ("ppo_ensemble", False): 60.0,
        ("rule_momentum_regime", False): 75.0,
    }
    return_map = {
        ("supervised_ranker", True): 0.18,
        ("ppo_ensemble", True): 0.04,
        ("rule_momentum_regime", True): 0.12,
        ("supervised_ranker", False): 0.15,
        ("ppo_ensemble", False): 0.02,
        ("rule_momentum_regime", False): 0.11,
    }
    sharpe_map = {
        ("supervised_ranker", True): 0.95,
        ("ppo_ensemble", True): 0.42,
        ("rule_momentum_regime", True): 0.81,
        ("supervised_ranker", False): 0.91,
        ("ppo_ensemble", False): 0.35,
        ("rule_momentum_regime", False): 0.78,
    }
    drawdown_map = {
        ("supervised_ranker", True): 0.08,
        ("ppo_ensemble", True): 0.14,
        ("rule_momentum_regime", True): 0.09,
        ("supervised_ranker", False): 0.09,
        ("ppo_ensemble", False): 0.16,
        ("rule_momentum_regime", False): 0.10,
    }
    loss_rate_map = {
        ("supervised_ranker", True): 0.18,
        ("ppo_ensemble", True): 0.32,
        ("rule_momentum_regime", True): 0.20,
        ("supervised_ranker", False): 0.18,
        ("ppo_ensemble", False): 0.34,
        ("rule_momentum_regime", False): 0.22,
    }
    verdict_map = {
        ("supervised_ranker", True): "needs_iteration",
        ("ppo_ensemble", True): "improve_robustness",
        ("rule_momentum_regime", True): "needs_iteration",
        ("supervised_ranker", False): "needs_iteration",
        ("ppo_ensemble", False): "improve_edge",
        ("rule_momentum_regime", False): "needs_iteration",
    }

    total_return = return_map[(family, challenge)]
    sharpe = sharpe_map[(family, challenge)]
    max_drawdown = drawdown_map[(family, challenge)]
    loss_rate = loss_rate_map[(family, challenge)]
    return {
        "family": family,
        "interval": interval,
        "walk_forward": {
            "n_folds": split_count,
            "avg_return": total_return / max(split_count, 1),
            "avg_sharpe": sharpe,
            "avg_max_drawdown": max_drawdown,
            "avg_win_rate": 0.6,
            "cumulative_return": total_return,
            "win_fold_ratio": 1.0,
            "worst_max_daily_loss": 0.02,
            "promotion_gate": {"passed": family == "supervised_ranker"},
            "evidence": {"status": "validated"},
            "folds": [
                {
                    "total_return": total_return / max(split_count, 1),
                    "sharpe_ratio": sharpe,
                    "max_drawdown": max_drawdown,
                    "win_rate": 0.6,
                }
                for _ in range(split_count)
            ],
        },
        "robustness": {
            "monte_carlo": {
                "avg_return": total_return / 2,
                "avg_sharpe": sharpe - 0.1,
                "avg_max_drawdown": max_drawdown,
                "loss_rate": loss_rate,
            },
            "stress_test": {"worst_crash_max_drawdown": max_drawdown + 0.02},
            "promotion_gate": {"passed": family == "supervised_ranker"},
            "evidence": {"status": "validated"},
        },
        "evidence_status": "validated",
        "selection_score": score_map[(family, challenge)],
        "verdict": verdict_map[(family, challenge)],
    }


def test_run_league_batch_writes_canonical_outputs_and_learning(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    paper_dir = tmp_path / "paper"
    batch_root = tmp_path / "league"
    _write_config(config_path, batch_root)
    _write_json(
        batch_root / "20260331_120000" / "project_learning.json",
        {
            "decision_records": [
                {"decision_key": "ppo_ensemble@4h", "verdict": "bad_decision"},
            ]
        },
    )
    _write_json(
        paper_dir / "20260404_120000" / "session_meta.json",
        {
            "session_id": "20260404_120000",
            "mode": "etoro",
            "strategy_family": "supervised_ranker",
        },
    )
    _write_json(
        paper_dir / "20260404_120000" / "report.json",
        {
            "session_id": "20260404_120000",
            "mode": "etoro",
            "strategy_family": "supervised_ranker",
            "summary": {"n_trades": 3, "n_rejections": 1},
        },
    )

    market = _make_market()

    monkeypatch.setattr(league, "load_market_bundle", lambda config, interval=None: (market, None))
    monkeypatch.setattr(
        league,
        "generate_walk_forward_splits",
        lambda data, **kwargs: [
            {
                "train": data,
                "test": data,
                "train_start": "2022-01-01",
                "train_end": "2023-12-31",
                "test_start": "2024-01-01",
                "test_end": "2024-06-30",
            },
            {
                "train": data,
                "test": data,
                "train_start": "2022-07-01",
                "train_end": "2024-06-30",
                "test_start": "2024-07-01",
                "test_end": "2024-12-31",
            },
        ],
    )
    monkeypatch.setattr(
        league,
        "evaluate_candidate_family",
        lambda family, *, config, splits, macro_data, interval, monte_carlo_sims, monte_carlo_noise_std, extreme_return_threshold, seed_offsets: _candidate_payload(
            family,
            interval,
            len(splits),
        ),
    )
    monkeypatch.setattr(
        league,
        "_load_latest_demo_session",
        lambda paper_trading_dir: {
            "session_id": "20260404_120000",
            "session_dir": str(paper_dir / "20260404_120000"),
            "report": {
                "session_id": "20260404_120000",
                "mode": "etoro",
                "strategy_family": "supervised_ranker",
                "summary": {"n_trades": 3, "n_rejections": 1},
            },
            "meta": {
                "session_id": "20260404_120000",
                "mode": "etoro",
                "strategy_family": "supervised_ranker",
            },
        },
    )

    result = league.run_league_batch(
        config_path=str(config_path),
        output_dir=str(batch_root / "20260404_150000"),
    )

    assert result["audit_verdict"] in {
        "needs_iteration",
        "improve_robustness",
        "promotable_demo",
        "ready_for_demo_soak",
    }
    assert Path(result["outputs"]["league_leaderboard"]).exists()
    assert Path(result["outputs"]["league_audit"]).exists()
    assert Path(result["outputs"]["demo_followup"]).exists()
    assert Path(result["outputs"]["decision_review"]).exists()
    assert Path(result["outputs"]["project_learning"]).exists()

    leaderboard = json.loads(Path(result["outputs"]["league_leaderboard"]).read_text(encoding="utf-8"))
    project_learning = json.loads(Path(result["outputs"]["project_learning"]).read_text(encoding="utf-8"))
    demo_followup = json.loads(Path(result["outputs"]["demo_followup"]).read_text(encoding="utf-8"))

    assert leaderboard["selection"]["gold_winner_family"] == "supervised_ranker"
    assert leaderboard["snapshot_id"] == "test_snapshot"
    assert project_learning["history_alerts"]
    assert "supervised_ranker@4h" in project_learning["patterns"]["good_decisions"]
    assert "ppo_ensemble@4h" in project_learning["patterns"]["bad_decisions"]
    assert demo_followup["status"] == "aligned_demo_candidate"


def test_record_verdict_does_not_reward_negative_edge_against_weak_baseline():
    baseline = {
        "family": "rule_momentum_regime",
        "interval": "4h",
        "walk_forward": {
            "cumulative_return": -0.02,
            "avg_sharpe": -0.50,
            "avg_max_drawdown": 0.04,
            "win_fold_ratio": 0.0,
        },
        "robustness": {
            "monte_carlo": {"loss_rate": 1.0},
            "stress_test": {"worst_crash_max_drawdown": 0.05},
        },
        "evidence_status": "validated",
        "verdict": "improve_edge",
    }
    candidate = {
        "family": "supervised_ranker",
        "interval": "4h",
        "walk_forward": {
            "cumulative_return": -0.001,
            "avg_sharpe": -0.10,
            "avg_max_drawdown": 0.02,
            "win_fold_ratio": 0.0,
        },
        "robustness": {
            "monte_carlo": {"loss_rate": 0.0},
            "stress_test": {"worst_crash_max_drawdown": 0.02},
        },
        "evidence_status": "validated",
        "verdict": "improve_edge",
    }

    assert league._record_verdict(candidate, baseline) == "bad_decision"


def test_record_verdict_marks_robustness_gap_as_inconclusive():
    baseline = {
        "family": "rule_momentum_regime",
        "interval": "4h",
        "walk_forward": {
            "cumulative_return": -0.01,
            "avg_sharpe": -0.40,
            "avg_max_drawdown": 0.04,
            "win_fold_ratio": 0.0,
        },
        "robustness": {
            "monte_carlo": {"loss_rate": 1.0},
            "stress_test": {"worst_crash_max_drawdown": 0.06},
        },
        "evidence_status": "validated",
        "verdict": "improve_edge",
    }
    candidate = {
        "family": "ppo_ensemble",
        "interval": "4h",
        "walk_forward": {
            "cumulative_return": 0.03,
            "avg_sharpe": 0.60,
            "avg_max_drawdown": 0.02,
            "win_fold_ratio": 1.0,
        },
        "robustness": {
            "monte_carlo": {"loss_rate": 0.30},
            "stress_test": {"worst_crash_max_drawdown": 0.03},
        },
        "evidence_status": "validated",
        "verdict": "improve_robustness",
    }

    assert league._record_verdict(candidate, baseline) == "inconclusive"
