# training/train.py
"""Walk-Forward Analysis V9 - Gold Standard pour validation temporelle.

Principe:
    Train 2010-2015 -> Test 2016
    Train 2010-2016 -> Test 2017
    ...
    Train 2010-2023 -> Test 2024

Chaque fenetre entraine un modele frais, puis le teste sur la periode
suivante (jamais vue). Le resultat est une courbe de performance
realiste qui simule le trading reel annee apres annee.

Usage:
    python training/train.py --config config/config.yaml
    python training/train.py --config config/config.yaml --ensemble 3
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from config.hardware import auto_scale_config, compute_optimal_params, detect_hardware
from config.schema import validate_config
from core.data_fetcher import download_data
from core.environment import TradingEnv
from core.evidence_hardening import (
    evaluate_backtest_artifact,
    evaluate_walk_forward_artifact,
    reconcile_equity,
)
from core.features import FeatureEngineer  # Turbo Init
from core.macro_data import MacroDataFetcher
from core.model_support import predict_with_optional_recurrence
from core.promotion_gate import (
    evaluate_walk_forward_promotion,
    promotion_thresholds_from_config,
)
from core.shared_memory_manager import SharedDataManager  # V9 Shared Memory
from core.utils import setup_logging

logger = setup_logging(__name__, "train.log")

# Essayer d'importer RecurrentPPO (optionnel)
try:
    from sb3_contrib import RecurrentPPO

    HAS_RECURRENT = True
except ImportError:
    HAS_RECURRENT = False
    logger.warning("sb3-contrib non disponible, RecurrentPPO desactive")


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        logger.error(f"Config {config_path} non trouve")
        return None
    with open(config_path) as f:
        config = yaml.safe_load(f)
    warnings = validate_config(config)
    for w in warnings:
        logger.warning(w)
    return config


def generate_walk_forward_splits(
    data: dict[str, pd.DataFrame],
    train_years: int = 5,
    test_months: int = 12,
    step_months: int = 12,
    embargo_months: int = 1,  # Anti-Leak Embargo (buffer pour les indicateurs)
) -> list[dict]:
    """Genere les fenetres walk-forward.

    Args:
        data: Dict {ticker: DataFrame} avec DatetimeIndex.
        train_years: Nombre minimum d'annees d'entrainement.
        test_months: Duree de la fenetre de test en mois.
        step_months: Pas entre chaque fenetre (en mois).
        embargo_months: Mois a sauter entre train et test (Data Leakage protection).

    Returns:
        Liste de dicts avec 'train' et 'test' (chacun Dict[str, DataFrame]).
    """
    ref_ticker = list(data.keys())[0]
    ref_df = data[ref_ticker]

    start_date = ref_df.index[0]
    end_date = ref_df.index[-1]

    splits = []
    train_end = start_date + pd.DateOffset(years=train_years)

    while True:
        # Periode d'Embargo (zone tampon)
        test_start = train_end + pd.DateOffset(months=embargo_months)
        test_end = test_start + pd.DateOffset(months=test_months)

        if test_end > end_date:
            break

        train_data = {}
        test_data = {}

        for ticker, df in data.items():
            train_mask = (df.index >= start_date) & (df.index < train_end)
            test_mask = (df.index >= test_start) & (df.index < test_end)  # Start APRES l'embargo

            train_slice = df.loc[train_mask]
            test_slice = df.loc[test_mask]

            if len(train_slice) < 100 or len(test_slice) < 50:
                continue

            train_data[ticker] = train_slice.copy()
            test_data[ticker] = test_slice.copy()

        if train_data and test_data and len(train_data) == len(data):
            splits.append(
                {
                    "train": train_data,
                    "test": test_data,
                    "train_start": str(start_date.date()),
                    "train_end": str(train_end.date()),
                    "test_start": str(test_start.date()),
                    "test_end": str(test_end.date()),
                }
            )
            logger.info(
                f"  Split {len(splits)}: "
                f"Train {start_date.date()}->{train_end.date()} "
                f"| Embargo ({embargo_months}m) "
                f"| Test {test_start.date()}->{test_end.date()}"
            )

        train_end += pd.DateOffset(months=step_months)

    logger.info(f"Total: {len(splits)} walk-forward splits with {embargo_months}m embargo")
    return splits


def make_env(data, macro_data, config, mode="train", features_precomputed=False):
    def _init():
        env_kwargs = {k: v for k, v in config.get("environment", {}).items()}
        env_kwargs["mode"] = mode
        env_kwargs["features_precomputed"] = features_precomputed  # Turbo Init
        env = TradingEnv(data=data, macro_data=macro_data, **env_kwargs)
        env = Monitor(env)
        return env

    return _init


def train_single_fold(
    train_data: dict[str, pd.DataFrame],
    test_data: dict[str, pd.DataFrame],
    macro_data: pd.DataFrame | None,
    config: dict,
    fold_idx: int,
    output_dir: str,
    use_recurrent: bool = False,
    seed: int | None = None,
    split_info: dict[str, str] | None = None,
) -> dict:
    """Entraine et evalue sur un seul fold walk-forward.

    Returns:
        Dict avec metriques du fold.
    """
    fold_dir = os.path.join(output_dir, f"fold_{fold_idx:02d}")
    os.makedirs(fold_dir, exist_ok=True)

    n_envs = config.get("training", {}).get("n_envs", 4)
    use_shared_memory = config.get("training", {}).get("use_shared_memory", False)
    shm_manager = None

    # [FAST] V9 Shared Memory: Optimisation RAM
    if use_shared_memory:
        try:
            logger.info(
                f"  [FAST] V9: Loading TRAIN data into Shared Memory for {n_envs} workers..."
            )
            shm_manager = SharedDataManager()
            # On remplace le gros dict de DF par un petit dict de Metadata
            train_data = shm_manager.put_data(train_data)
        except Exception as e:
            logger.error(f"Failed to init Shared Memory: {e}")
            if shm_manager:
                shm_manager.cleanup()
            raise e

    try:
        # Environnements d'entrainement
        if use_recurrent:
            # RecurrentPPO necessite DummyVecEnv (pas SubprocVecEnv)
            envs = DummyVecEnv(
                [
                    make_env(
                        train_data, macro_data, config, mode="train", features_precomputed=True
                    )
                    for _ in range(n_envs)
                ]
            )
        else:
            envs = SubprocVecEnv(
                [
                    make_env(
                        train_data, macro_data, config, mode="train", features_precomputed=True
                    )
                    for _ in range(n_envs)
                ]
            )

        envs = VecNormalize(
            envs,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=config.get("training", {}).get("gamma", 0.99),
        )

        # Architecture reseau
        net_arch = config.get("network", {}).get("net_arch", [512, 512, 256])
        activation_name = config.get("network", {}).get("activation_fn", "tanh")
        activation_fn = torch.nn.Tanh if activation_name == "tanh" else torch.nn.ReLU

        device = "cuda" if torch.cuda.is_available() else "cpu"

        training_cfg = config.get("training", {})
        timesteps = training_cfg.get("total_timesteps", 5_000_000)

        # Creer le modele
        if use_recurrent and HAS_RECURRENT:
            policy_kwargs = {
                "net_arch": net_arch,
                "activation_fn": activation_fn,
                "lstm_hidden_size": config.get("network", {}).get("lstm_hidden_size", 256),
                "n_lstm_layers": config.get("network", {}).get("n_lstm_layers", 1),
            }
            ppo_kwargs = {
                "learning_rate": training_cfg.get("learning_rate", 0.0003),
                "n_steps": training_cfg.get("n_steps", 2048),
                "batch_size": training_cfg.get("batch_size", 128),
                "n_epochs": training_cfg.get("n_epochs", 10),
                "gamma": training_cfg.get("gamma", 0.99),
                "gae_lambda": training_cfg.get("gae_lambda", 0.95),
                "clip_range": training_cfg.get("clip_range", 0.2),
                "ent_coef": training_cfg.get("ent_coef", 0.01),
                "vf_coef": training_cfg.get("vf_coef", 0.5),
                "max_grad_norm": training_cfg.get("max_grad_norm", 0.5),
                "policy_kwargs": policy_kwargs,
                "verbose": 0,
                "device": device,
                "tensorboard_log": os.path.join(fold_dir, "tb_logs"),
            }
            if seed is not None:
                ppo_kwargs["seed"] = seed

            model = RecurrentPPO("MlpLstmPolicy", envs, **ppo_kwargs)
            logger.info(f"  Fold {fold_idx}: RecurrentPPO (LSTM) on {device}")
        else:
            policy_kwargs = {
                "net_arch": [{"pi": net_arch, "vf": net_arch}],
                "activation_fn": activation_fn,
            }
            ppo_kwargs = {
                "learning_rate": training_cfg.get("learning_rate", 0.0003),
                "n_steps": training_cfg.get("n_steps", 2048),
                "batch_size": training_cfg.get("batch_size", 2048),
                "n_epochs": training_cfg.get("n_epochs", 10),
                "gamma": training_cfg.get("gamma", 0.99),
                "gae_lambda": training_cfg.get("gae_lambda", 0.95),
                "clip_range": training_cfg.get("clip_range", 0.2),
                "ent_coef": training_cfg.get("ent_coef", 0.01),
                "vf_coef": training_cfg.get("vf_coef", 0.5),
                "max_grad_norm": training_cfg.get("max_grad_norm", 0.5),
                "target_kl": training_cfg.get("target_kl", 0.02),
                "policy_kwargs": policy_kwargs,
                "verbose": 0,
                "device": device,
                "tensorboard_log": os.path.join(fold_dir, "tb_logs"),
            }
            if seed is not None:
                ppo_kwargs["seed"] = seed

            model = PPO("MlpPolicy", envs, **ppo_kwargs)
            logger.info(f"  Fold {fold_idx}: PPO standard on {device}")

        # Callbacks
        checkpoint_cb = CheckpointCallback(
            save_freq=max(timesteps // 10, 10000),
            save_path=os.path.join(fold_dir, "checkpoints"),
            name_prefix=f"fold_{fold_idx:02d}",
        )

        # Entrainer
        logger.info(f"  Fold {fold_idx}: Training {timesteps:,} timesteps...")
        try:
            import rich  # noqa: F401
            import tqdm  # noqa: F401

            _progress_bar = True
        except ImportError:
            _progress_bar = False
        model.learn(total_timesteps=timesteps, callback=[checkpoint_cb], progress_bar=_progress_bar)

        # Sauvegarder le modele
        model_path = os.path.join(fold_dir, "model")
        model.save(model_path)
        envs.save(os.path.join(fold_dir, "vecnormalize.pkl"))

        # Sauvegarder fold metadata (pour robustness tests)
        if split_info:
            metadata_path = os.path.join(fold_dir, "fold_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(split_info, f, indent=2)
            logger.info(f"  Fold {fold_idx}: Saved metadata to {metadata_path}")

        # Evaluer sur le test set (geler les stats de normalisation)
        logger.info(f"  Fold {fold_idx}: Evaluating on test period...")
        envs.training = False
        envs.norm_reward = False
        metrics = evaluate_on_test(
            model, envs, test_data, macro_data, config, features_precomputed=True
        )

        # Sauvegarder metriques
        metrics_path = os.path.join(fold_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        envs.close()

        logger.info(
            f"  Fold {fold_idx} results: "
            f"Return={metrics['total_return']:.2%} | "
            f"Sharpe={metrics['sharpe_ratio']:.2f} | "
            f"MaxDD={metrics['max_drawdown']:.2%} | "
            f"Trades={metrics['total_trades']}"
        )

    finally:
        if shm_manager:
            logger.info("  [FAST] V9: Cleaning up Shared Memory resources")
            shm_manager.cleanup()

    return metrics


def _annualization_factor(interval: str) -> float:
    """Facteur d'annualisation du Sharpe ratio selon l'intervalle des donnees."""
    factors = {"1h": 252 * 6.5, "1d": 252, "1wk": 52, "1mo": 12}
    return np.sqrt(factors.get(interval, 252))


def evaluate_on_test(
    model, train_envs, test_data, macro_data, config, features_precomputed=False
) -> dict:
    """Evalue le modele sur la periode de test.

    Returns:
        Dict avec total_return, sharpe_ratio, max_drawdown, etc.
    """
    env_kwargs = {k: v for k, v in config.get("environment", {}).items()}
    env_kwargs["mode"] = "backtest"
    env_kwargs["seed"] = 42
    env_kwargs["features_precomputed"] = features_precomputed

    test_env = TradingEnv(data=test_data, macro_data=macro_data, **env_kwargs)

    obs, info = test_env.reset()
    done = False
    equity_curve = [test_env.initial_balance]
    timestamps = [test_data[test_env.tickers[0]].index[min(test_env.current_step, len(test_data[test_env.tickers[0]]) - 1)]]
    recurrent_state = None
    episode_start = np.array([True], dtype=bool)
    max_equity_error = 0.0

    while not done:
        # Normaliser l'observation comme pendant le training
        obs_normalized = train_envs.normalize_obs(obs.reshape(1, -1)).flatten()
        action, recurrent_state = predict_with_optional_recurrence(
            model,
            obs_normalized,
            deterministic=True,
            recurrent_state=recurrent_state,
            episode_start=episode_start,
        )
        current_prices = {
            ticker: float(test_env._get_current_price(ticker)) for ticker in test_env.tickers
        }
        obs, reward, done, truncated, info = test_env.step(action)
        episode_start = np.array([done], dtype=bool)
        equity_curve.append(info["equity"])
        reconciliation = reconcile_equity(
            balance=float(test_env.balance),
            positions=test_env.portfolio,
            prices=current_prices,
            reported_equity=float(info["equity"]),
        )
        max_equity_error = max(max_equity_error, float(reconciliation["error"]))
        current_idx = min(test_env.current_step, len(test_data[test_env.tickers[0]]) - 1)
        timestamps.append(test_data[test_env.tickers[0]].index[current_idx])

    # Calculer metriques
    equity_series = pd.Series(equity_curve, index=pd.Index(timestamps[: len(equity_curve)]))
    returns = equity_series.pct_change().dropna()

    total_return = (equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0]
    max_drawdown = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max()

    sharpe_ratio = 0.0
    interval = config.get("data", {}).get("interval", "1h")
    if len(returns) > 1 and returns.std() > 0:
        sharpe_ratio = (returns.mean() / returns.std()) * _annualization_factor(interval)
    daily_equity = equity_series.groupby(pd.to_datetime(equity_series.index).normalize()).last()
    daily_returns = daily_equity.pct_change().dropna()
    max_daily_loss = abs(float(daily_returns.min())) if not daily_returns.empty and daily_returns.min() < 0 else 0.0
    closed_trades = info.get("winning_trades", 0) + info.get("losing_trades", 0)
    win_rate = info.get("winning_trades", 0) / closed_trades if closed_trades > 0 else 0.0

    return {
        "total_return": float(total_return),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown),
        "total_trades": info.get("total_trades", 0),
        "winning_trades": info.get("winning_trades", 0),
        "losing_trades": info.get("losing_trades", 0),
        "win_rate": float(win_rate),
        "final_equity": float(equity_series.iloc[-1]),
        "max_daily_loss": float(max_daily_loss),
        "max_equity_error": float(max_equity_error),
        "n_steps": len(equity_curve),
    }


def run_walk_forward(
    config_path: str,
    use_recurrent: bool = False,
    n_ensemble: int = 1,
    auto_scale: bool = False,
    use_shared_memory: bool = False,
):
    """Pipeline complet Walk-Forward Analysis."""
    logger.info("=" * 70)
    logger.info("WALK-FORWARD ANALYSIS V9")
    logger.info("=" * 70)

    config = load_config(config_path)
    if config is None:
        return

    # Override config with CLI
    if use_shared_memory:
        config["training"]["use_shared_memory"] = True

    if auto_scale:
        hw = detect_hardware()
        params = compute_optimal_params(hw, use_recurrent=use_recurrent)
        config = auto_scale_config(config, use_recurrent=use_recurrent)
        max_workers = params["max_workers"]
    else:
        max_workers = 3

    # 1. Telecharger donnees
    logger.info("Downloading data...")
    data = download_data(
        tickers=config["data"]["tickers"],
        period=config["data"].get("period", "5y"),
        interval=config["data"].get("interval", "1h"),
        max_workers=max_workers,
        dataset_path=config["data"].get("dataset_path"),
    )

    if not data:
        logger.error("No data downloaded")
        return

    logger.info(f"{len(data)} tickers loaded")

    # Feature engineer (used per-fold to avoid look-ahead bias)
    feature_engineer = FeatureEngineer()

    # 2. Telecharger donnees macro
    logger.info("Downloading macro data (VIX, TNX, DXY)...")
    macro_fetcher = MacroDataFetcher()
    ref_ticker = list(data.keys())[0]
    ref_df = data[ref_ticker]

    try:
        macro_data = macro_fetcher.fetch_all(
            start_date=str(ref_df.index[0].date()),
            end_date=str(ref_df.index[-1].date()),
            interval=config["data"].get("interval", "1h"),
        )
    except Exception as e:
        logger.warning(f"Failed to fetch macro data (index issue?): {e}")
        macro_data = pd.DataFrame()  # Empty DF

    if macro_data.empty:
        logger.warning("No macro data available, proceeding without")
        macro_data = None

    # 3. Generer splits walk-forward
    wf_cfg = config.get("walk_forward", {})
    logger.info("Generating walk-forward splits...")
    splits = generate_walk_forward_splits(
        data,
        train_years=wf_cfg.get("train_years", 1),
        test_months=wf_cfg.get("test_months", 6),
        step_months=wf_cfg.get("step_months", 6),
        embargo_months=wf_cfg.get("embargo_months", 1),
    )

    if not splits:
        logger.error("No walk-forward splits generated (not enough data)")
        return

    # 4. Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    algo = "recurrent_ppo" if use_recurrent else "ppo"
    output_dir = f"models/walk_forward_{algo}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # 5. Entrainer chaque fold
    all_metrics = []

    for fold_idx, split in enumerate(splits):
        logger.info(f"\n{'='*50}")
        logger.info(
            f"FOLD {fold_idx + 1}/{len(splits)}: "
            f"Train {split['train_start']}->{split['train_end']} | "
            f"Test {split['test_start']}->{split['test_end']}"
        )
        logger.info(f"{'='*50}")

        # Macro data pour ce fold (si disponible)
        fold_macro = macro_data

        # Compute features PER FOLD to avoid look-ahead bias
        # Each fold gets features computed only from its own data slice
        logger.info(f"  Computing features for fold {fold_idx + 1}...")
        fold_train = {}
        fold_test = {}
        for ticker in split["train"]:
            train_df = split["train"][ticker]
            original_train_idx = train_df.index
            feat_train = feature_engineer.calculate_all_features(train_df)
            if len(feat_train) == len(original_train_idx):
                feat_train.index = original_train_idx
            fold_train[ticker] = feat_train

            test_df = split["test"][ticker]
            original_test_idx = test_df.index
            feat_test = feature_engineer.calculate_all_features(test_df)
            if len(feat_test) == len(original_test_idx):
                feat_test.index = original_test_idx
            fold_test[ticker] = feat_test

        # Fold split info for metadata
        fold_split_info = {
            "train_start": split["train_start"],
            "train_end": split["train_end"],
            "test_start": split["test_start"],
            "test_end": split["test_end"],
        }

        # Ensemble : entrainer n_ensemble modeles avec des seeds differents
        if n_ensemble > 1:
            fold_metrics_list = []
            for ens_idx in range(n_ensemble):
                metrics = train_single_fold(
                    train_data=fold_train,
                    test_data=fold_test,
                    macro_data=fold_macro,
                    config=config,
                    fold_idx=fold_idx * n_ensemble + ens_idx,
                    output_dir=output_dir,
                    use_recurrent=use_recurrent,
                    seed=None,
                    split_info=fold_split_info,
                )
                fold_metrics_list.append(metrics)

            # Moyenne des metriques de l'ensemble
            avg_metrics = {
                key: float(np.mean([m[key] for m in fold_metrics_list]))
                for key in fold_metrics_list[0]
            }
            avg_metrics["ensemble_size"] = n_ensemble
            avg_metrics["fold_idx"] = fold_idx
            avg_metrics["train_period"] = f"{split['train_start']}->{split['train_end']}"
            avg_metrics["test_period"] = f"{split['test_start']}->{split['test_end']}"
            avg_metrics["accounting"] = {
                "max_equity_error": float(avg_metrics.get("max_equity_error", 0.0))
            }
            avg_metrics["evidence"] = evaluate_backtest_artifact(
                avg_metrics,
                interval=config.get("data", {}).get("interval", "1h"),
                test_period=avg_metrics["test_period"],
                accounting=avg_metrics["accounting"],
                initial_balance=float(config.get("environment", {}).get("initial_balance", 0.0)),
            )
            all_metrics.append(avg_metrics)
        else:
            metrics = train_single_fold(
                train_data=fold_train,
                test_data=fold_test,
                macro_data=fold_macro,
                config=config,
                fold_idx=fold_idx,
                output_dir=output_dir,
                use_recurrent=use_recurrent,
                seed=None,
                split_info=fold_split_info,
            )
            metrics["fold_idx"] = fold_idx
            metrics["train_period"] = f"{split['train_start']}->{split['train_end']}"
            metrics["test_period"] = f"{split['test_start']}->{split['test_end']}"
            metrics["accounting"] = {"max_equity_error": float(metrics.get("max_equity_error", 0.0))}
            metrics["evidence"] = evaluate_backtest_artifact(
                metrics,
                interval=config.get("data", {}).get("interval", "1h"),
                test_period=metrics["test_period"],
                accounting=metrics["accounting"],
                initial_balance=float(config.get("environment", {}).get("initial_balance", 0.0)),
            )
            all_metrics.append(metrics)

    # 6. Rapport final
    logger.info("\n" + "=" * 70)
    logger.info("WALK-FORWARD RESULTS")
    logger.info("=" * 70)

    returns = [m["total_return"] for m in all_metrics]
    sharpes = [m["sharpe_ratio"] for m in all_metrics]
    drawdowns = [m["max_drawdown"] for m in all_metrics]
    win_rates = [m.get("win_rate", 0.0) for m in all_metrics]
    max_daily_losses = [m.get("max_daily_loss", 0.0) for m in all_metrics]

    for m in all_metrics:
        logger.info(
            f"  Fold {m['fold_idx']}: "
            f"Return={m['total_return']:+.2%} | "
            f"Sharpe={m['sharpe_ratio']:.2f} | "
            f"MaxDD={m['max_drawdown']:.2%} | "
            f"Period: {m['test_period']}"
        )

    logger.info("\nAGGREGATED:")
    logger.info(f"  Avg Return:  {np.mean(returns):+.2%} (std: {np.std(returns):.2%})")
    logger.info(f"  Avg Sharpe:  {np.mean(sharpes):.2f} (std: {np.std(sharpes):.2f})")
    logger.info(f"  Avg MaxDD:   {np.mean(drawdowns):.2%}")
    logger.info(f"  Avg Daily Loss: {np.mean(max_daily_losses):.2%}")
    logger.info(f"  Avg WinRate: {np.mean(win_rates):.1%}")
    logger.info(f"  Win Folds:   {sum(1 for r in returns if r > 0)}/{len(returns)}")
    logger.info(f"  Cumulative:  {np.prod([1 + r for r in returns]) - 1:+.2%}")

    promotion_gate = evaluate_walk_forward_promotion(
        returns=returns,
        sharpes=sharpes,
        drawdowns=drawdowns,
        thresholds=promotion_thresholds_from_config(config),
    )
    logger.info(
        "  Promotion:   %s",
        "PASS" if promotion_gate["passed"] else "FAIL",
    )

    # Sauvegarder resultats
    results = {
        "algorithm": algo,
        "n_folds": len(all_metrics),
        "n_ensemble": n_ensemble,
        "output_dir": output_dir,
        "avg_return": float(np.mean(returns)),
        "avg_sharpe": float(np.mean(sharpes)),
        "avg_max_drawdown": float(np.mean(drawdowns)),
        "avg_max_daily_loss": float(np.mean(max_daily_losses)),
        "avg_win_rate": float(np.mean(win_rates)),
        "cumulative_return": float(np.prod([1 + r for r in returns]) - 1),
        "win_fold_ratio": sum(1 for r in returns if r > 0) / len(returns),
        "promotion_gate": promotion_gate,
        "evidence": evaluate_walk_forward_artifact(
            all_metrics,
            interval=config.get("data", {}).get("interval", "1h"),
            initial_balance=float(config.get("environment", {}).get("initial_balance", 0.0)),
        ),
        "folds": all_metrics,
    }

    results_path = os.path.join(output_dir, "walk_forward_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved: {results_path}")
    logger.info("=" * 70)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Walk-Forward Analysis V9")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--recurrent", action="store_true", help="Use RecurrentPPO (LSTM)")
    parser.add_argument("--ensemble", type=int, default=1, help="Ensemble size (1=single model)")
    parser.add_argument(
        "--auto-scale",
        action="store_true",
        help="Auto-detect hardware and scale n_envs/batch_size/n_steps",
    )
    parser.add_argument(
        "--shared-memory",
        action="store_true",
        help="Use Shared Memory feature (V9) to reduce RAM usage",
    )

    args = parser.parse_args()

    run_walk_forward(
        config_path=args.config,
        use_recurrent=args.recurrent,
        n_ensemble=args.ensemble,
        auto_scale=args.auto_scale,
        use_shared_memory=args.shared_memory,
    )


if __name__ == "__main__":
    main()
