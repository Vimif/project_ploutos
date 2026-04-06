"""Schema validation for YAML training configs.

The goal is to fail on real configuration mistakes instead of silently
continuing with typos or impossible PPO settings.
"""

from __future__ import annotations

from typing import Any

from core.exceptions import ConfigValidationError
from training.strategy_policies import SUPPORTED_STRATEGY_FAMILIES

FieldType = type | tuple[type, ...]
FieldSpec = tuple[FieldType, float | int | None, float | int | None]

SCHEMA: dict[str, dict[str, FieldSpec]] = {
    "training": {
        "total_timesteps": (int, 128, 1_000_000_000),
        "n_envs": (int, 1, 512),
        "batch_size": (int, 32, 131_072),
        "n_steps": (int, 64, 16_384),
        "n_epochs": (int, 1, 100),
        "learning_rate": (float, 1e-6, 1e-1),
        "gamma": (float, 0.9, 1.0),
        "gae_lambda": (float, 0.8, 1.0),
        "clip_range": (float, 0.05, 0.5),
        "ent_coef": (float, 0.0, 0.5),
        "max_grad_norm": (float, 0.1, 10.0),
        "vf_coef": (float, 0.01, 10.0),
        "target_kl": (float, 0.001, 1.0),
        "use_shared_memory": (bool, None, None),
    },
    "environment": {
        "initial_balance": ((int, float), 1_000, 100_000_000),
        "max_steps": (int, 100, 100_000),
        "buy_pct": (float, 0.01, 1.0),
        "max_position_pct": (float, 0.01, 1.0),
        "max_trades_per_day": (int, 1, 1_000),
        "min_holding_period": (int, 0, 100),
        "reward_scaling": (float, 0.01, 100.0),
        "warmup_steps": (int, 0, 1_000),
        "steps_per_trading_week": (int, 1, 500),
        "drawdown_threshold": (float, 0.01, 1.0),
        "commission": (float, 0.0, 1.0),
        "sec_fee": (float, 0.0, 1.0),
        "finra_taf": (float, 0.0, 1.0),
        "slippage_model": (str, None, None),
        "spread_bps": ((int, float), 0.0, 100.0),
        "market_impact_factor": (float, 0.0, 1.0),
        "use_sharpe_penalty": (bool, None, None),
        "use_drawdown_penalty": (bool, None, None),
        "reward_trade_success": (float, 0.0, 10.0),
        "penalty_overtrading": (float, 0.0, 10.0),
        "drawdown_penalty_factor": (float, 0.0, 100.0),
        "reward_buy_executed": (float, -10.0, 10.0),
        "reward_overtrading": (float, -10.0, 10.0),
        "reward_invalid_trade": (float, -10.0, 10.0),
        "reward_bad_price": (float, -10.0, 10.0),
        "reward_good_return_bonus": (float, -10.0, 10.0),
        "reward_high_winrate_bonus": (float, -10.0, 10.0),
        "good_return_threshold": (float, 0.0, 1.0),
        "high_winrate_threshold": (float, 0.0, 1.0),
        "max_features_per_ticker": (int, 0, 1_000),
        "stop_loss_pct": (float, 0.0, 1.0),
    },
    "data": {
        "tickers": (list, None, None),
        "period": (str, None, None),
        "interval": (str, None, None),
        "dataset_path": (str, None, None),
    },
    "walk_forward": {
        "train_years": (int, 1, 30),
        "test_months": (int, 1, 60),
        "step_months": (int, 1, 60),
        "embargo_months": (int, 0, 24),
    },
    "network": {
        "net_arch": (list, None, None),
        "activation_fn": (str, None, None),
        "lstm_hidden_size": (int, 16, 2_048),
        "n_lstm_layers": (int, 1, 8),
    },
    "checkpoint": {
        "save_freq": (int, 100, 10_000_000),
        "save_path": (str, None, None),
    },
    "eval": {
        "eval_freq": (int, 100, 10_000_000),
        "n_eval_episodes": (int, 1, 100),
        "best_model_save_path": (str, None, None),
    },
    "optuna": {
        "timesteps_per_trial": (int, 128, 1_000_000_000),
    },
    "wandb": {
        "enabled": (bool, None, None),
        "project": (str, None, None),
        "entity": ((str, type(None)), None, None),
    },
    "live": {
        "ensemble_size": (int, 1, 32),
        "min_confidence": (float, 0.0, 1.0),
        "buy_pct": (float, 0.01, 1.0),
        "max_position_pct": (float, 0.01, 1.0),
        "max_open_positions": (int, 1, 100),
        "stop_loss_pct": (float, 0.0, 1.0),
        "take_profit_pct": (float, 0.0, 10.0),
        "max_cost_pct": (float, 0.0, 1.0),
        "max_drawdown": (float, 0.0, 1.0),
        "max_daily_loss": (float, 0.0, 1.0),
        "inactivity_hours": ((int, float), 0.0, 168.0),
        "interval_minutes": (int, 1, 1_440),
        "regime_fast_ma": (int, 2, 500),
        "regime_slow_ma": (int, 2, 500),
        "regime_vix_threshold": ((int, float), 1.0, 100.0),
        "dedupe_window_seconds": (int, 0, 86_400),
        "history_days": (int, 5, 365),
        "order_fill_timeout_seconds": (int, 1, 600),
        "order_poll_interval_seconds": ((int, float), 0.1, 60.0),
        "promotion_sharpe_min": ((int, float), -10.0, 10.0),
        "promotion_win_fold_ratio_min": (float, 0.0, 1.0),
        "promotion_cumulative_return_min": ((int, float), -1.0, 10.0),
        "promotion_loss_rate_max": (float, 0.0, 1.0),
        "order_min_notional": ((int, float), 0.0, 1_000_000.0),
    },
    "strategy": {
        "family": (str, None, None),
        "candidate_families": (list, None, None),
        "phase2_interval": (str, None, None),
        "phase2_top_k": (int, 1, 10),
        "seed_offsets": (list, None, None),
        "monte_carlo_sims": (int, 1, 1_000),
        "monte_carlo_noise_std": (float, 0.0, 1.0),
        "supervised_forward_bars": (int, 1, 128),
        "supervised_buy_threshold": ((int, float), -1.0, 10.0),
        "supervised_sell_threshold": ((int, float), -1.0, 10.0),
        "rule_fast_ma": (int, 2, 500),
        "rule_slow_ma": (int, 2, 500),
        "rule_momentum_lookback": (int, 1, 252),
        "ensemble_size": (int, 1, 32),
        "extreme_return_threshold": ((int, float), 0.0, 1_000_000.0),
    },
    "league": {
        "snapshot_id": (str, None, None),
        "candidate_families": (list, None, None),
        "baseline_family": (str, None, None),
        "gold_holdout_months": (int, 1, 60),
        "batch_output_root": (str, None, None),
        "cadence": (str, None, None),
        "learning_granularity": (str, None, None),
        "learning_action_mode": (str, None, None),
    },
}

REQUIRED_SECTIONS = {"training", "environment", "data", "walk_forward"}


def _type_label(expected_type: FieldType) -> str:
    if isinstance(expected_type, tuple):
        return " | ".join(t.__name__ for t in expected_type)
    return expected_type.__name__


def _is_valid_type(value: Any, expected_type: FieldType) -> bool:
    if isinstance(expected_type, tuple):
        return any(_is_valid_type(value, member_type) for member_type in expected_type)
    if expected_type is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, expected_type)


def _validate_bounds(section_name: str, key: str, value: Any, min_val: Any, max_val: Any) -> None:
    if isinstance(value, bool):
        return
    if min_val is not None and value < min_val:
        raise ConfigValidationError(f"'{section_name}.{key}': {value} < minimum {min_val}")
    if max_val is not None and value > max_val:
        raise ConfigValidationError(f"'{section_name}.{key}': {value} > maximum {max_val}")


def _validate_section(
    section_name: str, section: dict[str, Any], fields: dict[str, FieldSpec]
) -> None:
    known_keys = set(fields)
    unknown_keys = sorted(set(section) - known_keys)
    if unknown_keys:
        pretty_keys = ", ".join(unknown_keys)
        raise ConfigValidationError(
            f"Section '{section_name}' contains unknown key(s): {pretty_keys}. "
            f"Valid keys: {sorted(known_keys)}"
        )

    for key, (expected_type, min_val, max_val) in fields.items():
        if key not in section:
            continue

        value = section[key]
        if not _is_valid_type(value, expected_type):
            raise ConfigValidationError(
                f"'{section_name}.{key}': expected {_type_label(expected_type)}, "
                f"got {type(value).__name__} ({value!r})"
            )
        _validate_bounds(section_name, key, value, min_val, max_val)


def _validate_cross_field_constraints(config: dict[str, Any]) -> None:
    training = config.get("training", {})
    n_envs = training.get("n_envs", 1)
    n_steps = training.get("n_steps", 2_048)
    batch_size = training.get("batch_size", 64)

    if n_envs * n_steps < batch_size:
        raise ConfigValidationError(
            f"n_envs * n_steps ({n_envs * n_steps}) < batch_size ({batch_size}). "
            "PPO requires n_envs * n_steps >= batch_size."
        )

    live = config.get("live", {})
    fast_ma = live.get("regime_fast_ma")
    slow_ma = live.get("regime_slow_ma")
    if fast_ma is not None and slow_ma is not None and fast_ma >= slow_ma:
        raise ConfigValidationError(
            "live.regime_fast_ma must be strictly lower than live.regime_slow_ma."
        )

    strategy = config.get("strategy", {})
    family = strategy.get("family")
    if family is not None and family not in SUPPORTED_STRATEGY_FAMILIES:
        raise ConfigValidationError(
            "strategy.family must be one of "
            f"{list(SUPPORTED_STRATEGY_FAMILIES)}."
        )

    candidate_families = strategy.get("candidate_families", [])
    invalid_families = [
        candidate_family
        for candidate_family in candidate_families
        if candidate_family not in SUPPORTED_STRATEGY_FAMILIES
    ]
    if invalid_families:
        raise ConfigValidationError(
            "strategy.candidate_families contains unsupported value(s): "
            f"{invalid_families}. Supported families: {list(SUPPORTED_STRATEGY_FAMILIES)}"
        )

    seed_offsets = strategy.get("seed_offsets", [0])
    if not seed_offsets:
        raise ConfigValidationError("strategy.seed_offsets must contain at least one integer offset.")
    invalid_seed_offsets = [
        offset for offset in seed_offsets if not isinstance(offset, int) or isinstance(offset, bool)
    ]
    if invalid_seed_offsets:
        raise ConfigValidationError(
            "strategy.seed_offsets must contain only integers. "
            f"Invalid values: {invalid_seed_offsets}"
        )

    rule_fast_ma = strategy.get("rule_fast_ma")
    rule_slow_ma = strategy.get("rule_slow_ma")
    if rule_fast_ma is not None and rule_slow_ma is not None and rule_fast_ma >= rule_slow_ma:
        raise ConfigValidationError(
            "strategy.rule_fast_ma must be strictly lower than strategy.rule_slow_ma."
        )

    league = config.get("league", {})
    league_families = league.get("candidate_families", [])
    invalid_league_families = [
        candidate_family
        for candidate_family in league_families
        if candidate_family not in SUPPORTED_STRATEGY_FAMILIES
    ]
    if invalid_league_families:
        raise ConfigValidationError(
            "league.candidate_families contains unsupported value(s): "
            f"{invalid_league_families}. Supported families: {list(SUPPORTED_STRATEGY_FAMILIES)}"
        )

    baseline_family = league.get("baseline_family")
    if baseline_family is not None and baseline_family not in SUPPORTED_STRATEGY_FAMILIES:
        raise ConfigValidationError(
            "league.baseline_family must be one of "
            f"{list(SUPPORTED_STRATEGY_FAMILIES)}."
        )


def validate_config(config: dict) -> list[str]:
    """Validate a YAML config and return non-fatal warnings.

    The current policy is intentionally strict: typos, unknown sections and
    impossible PPO settings raise ``ConfigValidationError``.
    """
    if not isinstance(config, dict):
        raise ConfigValidationError(
            f"Top-level config must be a mapping, got {type(config).__name__}"
        )

    unknown_sections = sorted(set(config) - set(SCHEMA))
    if unknown_sections:
        raise ConfigValidationError(
            f"Unknown top-level section(s): {', '.join(unknown_sections)}. "
            f"Valid sections: {sorted(SCHEMA)}"
        )

    missing_sections = sorted(section for section in REQUIRED_SECTIONS if section not in config)
    if missing_sections:
        raise ConfigValidationError(f"Missing required section(s): {', '.join(missing_sections)}")

    for section_name, fields in SCHEMA.items():
        section = config.get(section_name)
        if section is None:
            continue
        if not isinstance(section, dict):
            raise ConfigValidationError(
                f"Section '{section_name}' must be a mapping, got {type(section).__name__}"
            )
        _validate_section(section_name, section, fields)

    _validate_cross_field_constraints(config)
    return []
