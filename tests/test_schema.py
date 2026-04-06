"""Tests for strict YAML config validation."""

from pathlib import Path

import pytest
import yaml

from config.schema import validate_config
from core.exceptions import ConfigValidationError


def make_valid_config():
    return {
        "training": {
            "total_timesteps": 1_000,
            "n_envs": 2,
            "batch_size": 64,
            "n_steps": 64,
            "n_epochs": 2,
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
        },
        "environment": {
            "initial_balance": 10_000,
            "commission": 0.001,
            "reward_scaling": 1.0,
            "max_steps": 500,
            "buy_pct": 0.2,
            "max_position_pct": 0.25,
            "max_trades_per_day": 10,
            "min_holding_period": 1,
            "warmup_steps": 10,
            "steps_per_trading_week": 78,
            "drawdown_threshold": 0.1,
        },
        "data": {
            "tickers": ["TEST"],
            "period": "2y",
            "interval": "1h",
        },
        "walk_forward": {
            "train_years": 1,
            "test_months": 3,
            "step_months": 3,
        },
    }


class TestValidateConfig:
    def test_valid_minimal_config_passes(self):
        assert validate_config(make_valid_config()) == []

    def test_repo_config_yaml_is_valid(self):
        config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        assert validate_config(config) == []

    def test_optional_sections_are_allowed(self):
        config = make_valid_config()
        config["optuna"] = {"timesteps_per_trial": 1_000}
        config["wandb"] = {"enabled": False, "project": "ploutos", "entity": None}
        config["live"] = {"ensemble_size": 3, "regime_fast_ma": 20, "regime_slow_ma": 50}
        config["strategy"] = {
            "family": "ppo_ensemble",
            "candidate_families": ["ppo_single", "ppo_ensemble", "supervised_ranker"],
            "seed_offsets": [0, 100],
            "rule_fast_ma": 20,
            "rule_slow_ma": 50,
        }
        config["league"] = {
            "snapshot_id": "test_snapshot",
            "candidate_families": ["supervised_ranker", "ppo_ensemble", "rule_momentum_regime"],
            "baseline_family": "rule_momentum_regime",
            "gold_holdout_months": 6,
            "batch_output_root": "logs/league_batches",
            "cadence": "batch_complete_only",
            "learning_granularity": "batch_decision",
            "learning_action_mode": "advisory",
        }

        assert validate_config(config) == []

    def test_unknown_top_level_section_raises(self):
        config = make_valid_config()
        config["legacy"] = {"enabled": True}

        with pytest.raises(ConfigValidationError, match="Unknown top-level section"):
            validate_config(config)

    def test_unknown_key_raises(self):
        config = make_valid_config()
        config["training"]["xml_output"] = False

        with pytest.raises(ConfigValidationError, match="training'.*xml_output|xml_output"):
            validate_config(config)

    def test_missing_required_section_raises(self):
        config = make_valid_config()
        del config["walk_forward"]

        with pytest.raises(ConfigValidationError, match="Missing required section"):
            validate_config(config)

    def test_invalid_batch_geometry_raises(self):
        config = make_valid_config()
        config["training"]["n_envs"] = 1
        config["training"]["n_steps"] = 64
        config["training"]["batch_size"] = 128

        with pytest.raises(ConfigValidationError, match="n_envs \\* n_steps"):
            validate_config(config)

    def test_tuple_type_error_message_is_readable(self):
        config = make_valid_config()
        config["environment"]["initial_balance"] = "a lot"

        with pytest.raises(ConfigValidationError, match="expected int \\| float"):
            validate_config(config)

    def test_bool_is_not_accepted_for_int_fields(self):
        config = make_valid_config()
        config["training"]["n_envs"] = True

        with pytest.raises(ConfigValidationError, match="training.n_envs|got bool"):
            validate_config(config)

    def test_strategy_family_must_be_supported(self):
        config = make_valid_config()
        config["strategy"] = {"family": "transformer_magic"}

        with pytest.raises(ConfigValidationError, match="strategy.family"):
            validate_config(config)

    def test_strategy_rule_fast_ma_must_be_lower_than_slow_ma(self):
        config = make_valid_config()
        config["strategy"] = {"family": "rule_momentum_regime", "rule_fast_ma": 50, "rule_slow_ma": 20}

        with pytest.raises(ConfigValidationError, match="strategy.rule_fast_ma"):
            validate_config(config)

    def test_strategy_seed_offsets_must_be_integers(self):
        config = make_valid_config()
        config["strategy"] = {"family": "ppo_ensemble", "seed_offsets": [0, "bad"]}

        with pytest.raises(ConfigValidationError, match="strategy.seed_offsets"):
            validate_config(config)

    def test_league_baseline_family_must_be_supported(self):
        config = make_valid_config()
        config["league"] = {"baseline_family": "alien_alpha"}

        with pytest.raises(ConfigValidationError, match="league.baseline_family"):
            validate_config(config)
