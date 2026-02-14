# config/schema.py
"""Validation légère des configs YAML de training.

Détecte les typos et valeurs aberrantes sans dépendance externe.

Usage:
    from config.schema import validate_config
    config = yaml.safe_load(open('config/config.yaml'))
    validate_config(config)  # Lève ConfigValidationError si invalide
"""

import logging

from core.exceptions import ConfigValidationError

logger = logging.getLogger(__name__)

# Clés attendues par section avec (type, min, max) ou (type, None, None)
SCHEMA = {
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
    },
    "environment": {
        "initial_balance": ((int, float), 1000, 100_000_000),
        "max_steps": (int, 100, 100_000),
        "buy_pct": (float, 0.01, 1.0),
        "max_position_pct": (float, 0.01, 1.0),
        "max_trades_per_day": (int, 1, 1000),
        "min_holding_period": (int, 0, 100),
        "reward_scaling": (float, 0.01, 100.0),
        "warmup_steps": (int, 0, 1000),
        "steps_per_trading_week": (int, 1, 500),
        "drawdown_threshold": (float, 0.01, 1.0),
    },
    "data": {
        "tickers": (list, None, None),
        "period": (str, None, None),
        "interval": (str, None, None),
    },
    "walk_forward": {
        "train_years": (int, 1, 30),
        "test_months": (int, 1, 60),
        "step_months": (int, 1, 60),
    },
    "network": {
        "net_arch": (list, None, None),
        "activation_fn": (str, None, None),
        "lstm_hidden_size": (int, 16, 2048),
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
}


def validate_config(config: dict) -> list:
    """Valide la config et retourne les warnings.

    Args:
        config: Dict chargé depuis YAML.

    Returns:
        Liste de warnings (str). Vide si tout est OK.

    Raises:
        ValueError: Si une valeur est hors limites.
    """
    warnings = []

    for section_name, fields in SCHEMA.items():
        section = config.get(section_name)
        if section is None:
            warnings.append(f"Section '{section_name}' manquante dans la config")
            continue

        # Détecter les clés inconnues (potentielles typos)
        known_keys = set(fields.keys())
        for key in section:
            if key not in known_keys:
                warnings.append(
                    f"Clé inconnue '{section_name}.{key}' "
                    f"(typo ? clés valides: {sorted(known_keys)})"
                )

        # Valider les types et bornes
        for key, (expected_type, min_val, max_val) in fields.items():
            if key not in section:
                continue

            value = section[key]

            # Vérifier le type
            if not isinstance(value, expected_type):
                # Accepter int là où float est attendu
                if expected_type is float and isinstance(value, int):
                    pass
                else:
                    raise ConfigValidationError(
                        f"'{section_name}.{key}': attendu {expected_type.__name__}, "
                        f"obtenu {type(value).__name__} ({value})"
                    )

            # Vérifier les bornes
            if min_val is not None and value < min_val:
                raise ConfigValidationError(f"'{section_name}.{key}': {value} < minimum {min_val}")
            if max_val is not None and value > max_val:
                raise ConfigValidationError(f"'{section_name}.{key}': {value} > maximum {max_val}")

    # Cross-field constraints
    training = config.get("training", {})
    n_envs = training.get("n_envs", 1)
    n_steps = training.get("n_steps", 2048)
    batch_size = training.get("batch_size", 64)

    if n_envs * n_steps < batch_size:
        warnings.append(
            f"n_envs * n_steps ({n_envs * n_steps}) < batch_size ({batch_size}). "
            f"PPO requires n_envs * n_steps >= batch_size."
        )

    for w in warnings:
        logger.warning(f"Config warning: {w}")

    return warnings
