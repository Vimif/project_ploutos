from pathlib import Path

import pandas as pd
import yaml

from training import strategy_compare


def _write_config(path: Path) -> None:
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
            "buy_pct": 0.15,
            "max_position_pct": 0.15,
            "max_trades_per_day": 10,
            "min_holding_period": 1,
            "warmup_steps": 5,
            "steps_per_trading_week": 78,
            "drawdown_threshold": 0.1,
            "max_features_per_ticker": 5,
            "stop_loss_pct": 0.06,
        },
        "data": {
            "tickers": ["SPY", "QQQ"],
            "period": "2y",
            "interval": "1h",
            "dataset_path": "./data/dataset_v8/",
        },
        "walk_forward": {"train_years": 1, "test_months": 6, "step_months": 6},
        "live": {
            "min_confidence": 0.67,
            "max_position_pct": 0.15,
            "max_open_positions": 5,
            "max_cost_pct": 0.005,
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
            "family": "ppo_ensemble",
            "candidate_families": [
                "ppo_single",
                "ppo_ensemble",
                "supervised_ranker",
            ],
            "phase2_interval": "4h",
            "phase2_top_k": 2,
            "seed_offsets": [0, 100],
            "monte_carlo_sims": 3,
            "monte_carlo_noise_std": 0.005,
            "rule_fast_ma": 20,
            "rule_slow_ma": 50,
        },
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def test_compare_strategy_families_builds_leaderboard_with_phase2(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)

    dates = pd.date_range("2024-01-01", periods=500, freq="h")
    market = {
        "SPY": pd.DataFrame({"Close": range(500)}, index=dates),
        "QQQ": pd.DataFrame({"Close": range(500)}, index=dates),
    }
    splits = [
        {
            "train": market,
            "test": market,
            "train_start": "2024-01-01",
            "train_end": "2024-12-31",
            "test_start": "2025-01-01",
            "test_end": "2025-06-30",
        }
    ]
    calls = []

    def fake_load_market_bundle(config, interval=None, max_workers=3):
        del config, max_workers
        return market, None

    def fake_generate_walk_forward_splits(data, **kwargs):
        del data, kwargs
        return splits

    def fake_evaluate_candidate_family(
        family,
        *,
        config,
        splits,
        macro_data,
        interval,
        monte_carlo_sims,
        monte_carlo_noise_std,
        extreme_return_threshold,
        seed_offsets,
    ):
        del (
            config,
            splits,
            macro_data,
            monte_carlo_sims,
            monte_carlo_noise_std,
            extreme_return_threshold,
        )
        calls.append((family, interval, tuple(seed_offsets)))
        score_map = {
            ("ppo_single", "1h"): 60.0,
            ("ppo_ensemble", "1h"): 55.0,
            ("supervised_ranker", "1h"): 70.0,
            ("supervised_ranker", "4h"): 72.0,
            ("ppo_single", "4h"): 62.0,
        }
        cumulative_return = 0.15 if family == "supervised_ranker" else 0.08
        avg_sharpe = 1.0 if family == "supervised_ranker" else 0.75
        avg_drawdown = 0.10 if family == "supervised_ranker" else 0.11
        loss_rate = 0.10 if family == "supervised_ranker" else 0.18
        return {
            "family": family,
            "interval": interval,
            "walk_forward": {
                "cumulative_return": cumulative_return,
                "avg_sharpe": avg_sharpe,
                "avg_max_drawdown": avg_drawdown,
                "win_fold_ratio": 1.0,
                "worst_max_daily_loss": 0.02,
                "promotion_gate": {"passed": family == "supervised_ranker"},
                "evidence": {"status": "validated"},
            },
            "robustness": {
                "monte_carlo": {"loss_rate": loss_rate},
                "stress_test": {"worst_crash_max_drawdown": avg_drawdown},
                "promotion_gate": {"passed": family == "supervised_ranker"},
                "evidence": {"status": "validated"},
            },
            "evidence_status": "validated",
            "selection_score": score_map[(family, interval)],
            "verdict": "promotable_demo" if family == "supervised_ranker" else "needs_iteration",
        }

    monkeypatch.setattr(strategy_compare, "load_market_bundle", fake_load_market_bundle)
    monkeypatch.setattr(
        strategy_compare, "generate_walk_forward_splits", fake_generate_walk_forward_splits
    )
    monkeypatch.setattr(
        strategy_compare, "evaluate_candidate_family", fake_evaluate_candidate_family
    )

    leaderboard = strategy_compare.compare_strategy_families(
        config_path=str(config_path),
        output_dir=str(tmp_path / "leaderboard"),
    )

    assert leaderboard["selection"]["winner_family"] == "supervised_ranker"
    assert leaderboard["selection"]["winner_beats_ppo_ensemble"] is True
    assert [entry["family"] for entry in leaderboard["phase_2"]] == [
        "supervised_ranker",
        "ppo_single",
    ]
    assert ("ppo_ensemble", "4h", (0, 100)) not in calls
    assert leaderboard["protocol"]["seed_offsets"] == [0, 100]
    assert (tmp_path / "leaderboard" / "strategy_leaderboard.json").exists()


def test_resolve_strategy_settings_normalizes_seed_offsets():
    settings = strategy_compare.resolve_strategy_settings(
        {"strategy": {"seed_offsets": [0, 100, 0, -100]}}
    )

    assert settings["seed_offsets"] == [0, 100, -100]
