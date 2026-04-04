"""Schema validation for YAML training configs.

The goal is to fail on real configuration mistakes instead of silently
continuing with typos or impossible PPO settings.
"""

from __future__ import annotations

from typing import Any

from core.exceptions import ConfigValidationError

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
