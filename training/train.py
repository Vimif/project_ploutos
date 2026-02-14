# training/train.py
"""Walk-Forward Analysis V9 - Gold Standard pour validation temporelle.

Principe:
    Train 2010-2015 -> Test 2016
    Train 2010-2016 -> Test 2017
    ...
    Train 2010-2023 -> Test 2024

Chaque fenêtre entraîne un modèle frais, puis le teste sur la période
suivante (jamais vue). Le résultat est une courbe de performance
réaliste qui simule le trading réel année après année.

Usage:
    python training/train.py --config config/config.yaml
    python training/train.py --config config/config.yaml --ensemble 3
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
import yaml
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor

from core.environment import TradingEnv
from core.macro_data import MacroDataFetcher
from core.data_fetcher import download_data
from core.utils import setup_logging
from config.hardware import auto_scale_config, detect_hardware, compute_optimal_params
from config.schema import validate_config
from core.features import FeatureEngineer  # Turbo Init
from core.shared_memory_manager import SharedDataManager  # V9 Shared Memory

logger = setup_logging(__name__, "train.log")

# Essayer d'importer RecurrentPPO (optionnel)
try:
    from sb3_contrib import RecurrentPPO

    HAS_RECURRENT = True
except ImportError:
    HAS_RECURRENT = False
    logger.warning("sb3-contrib non disponible, RecurrentPPO désactivé")


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        logger.error(f"Config {config_path} non trouvé")
        return None
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    warnings = validate_config(config)
    for w in warnings:
        logger.warning(w)
    return config


def generate_walk_forward_splits(
    data: Dict[str, pd.DataFrame],
    train_years: int = 5,
    test_months: int = 12,
    step_months: int = 12,
    embargo_months: int = 1,  # Anti-Leak Embargo (buffer pour les indicateurs)
) -> List[Dict]:
    """Génère les fenêtres walk-forward.

    Args:
        data: Dict {ticker: DataFrame} avec DatetimeIndex.
        train_years: Nombre minimum d'années d'entraînement.
        test_months: Durée de la fenêtre de test en mois.
        step_months: Pas entre chaque fenêtre (en mois).
        embargo_months: Mois à sauter entre train et test (Data Leakage protection).

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
        # Période d'Embargo (zone tampon)
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
    train_data: Dict[str, pd.DataFrame],
    test_data: Dict[str, pd.DataFrame],
    macro_data: Optional[pd.DataFrame],
    config: dict,
    fold_idx: int,
    output_dir: str,
    use_recurrent: bool = False,
    seed: int = 42,
) -> dict:
    """Entraîne et évalue sur un seul fold walk-forward.

    Returns:
        Dict avec métriques du fold.
    """
    fold_dir = os.path.join(output_dir, f"fold_{fold_idx:02d}")
    os.makedirs(fold_dir, exist_ok=True)

    n_envs = config.get("training", {}).get("n_envs", 4)
    use_shared_memory = config.get("training", {}).get("use_shared_memory", False)
    shm_manager = None
    
    # ⚡ V9 Shared Memory: Optimisation RAM
    if use_shared_memory:
        try:
            logger.info(f"  ⚡ V9: Loading TRAIN data into Shared Memory for {n_envs} workers...")
            shm_manager = SharedDataManager()
            # On remplace le gros dict de DF par un petit dict de Metadata
            train_data = shm_manager.put_data(train_data)
        except Exception as e:
            logger.error(f"Failed to init Shared Memory: {e}")
            if shm_manager: shm_manager.cleanup()
            raise e

    try:
        # Environnements d'entraînement
        if use_recurrent:
            # RecurrentPPO nécessite DummyVecEnv (pas SubprocVecEnv)
            envs = DummyVecEnv(
                [
                    make_env(train_data, macro_data, config, mode="train", features_precomputed=True)
                    for _ in range(n_envs)
                ]
            )
        else:
            envs = SubprocVecEnv(
                [
                    make_env(train_data, macro_data, config, mode="train", features_precomputed=True)
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

        # Architecture réseau
        net_arch = config.get("network", {}).get("net_arch", [512, 512, 256])
        activation_name = config.get("network", {}).get("activation_fn", "tanh")
        activation_fn = torch.nn.Tanh if activation_name == "tanh" else torch.nn.ReLU

        device = "cuda" if torch.cuda.is_available() else "cpu"

        training_cfg = config.get("training", {})
        timesteps = training_cfg.get("total_timesteps", 5_000_000)

        # Créer le modèle
        if use_recurrent and HAS_RECURRENT:
            policy_kwargs = {
                "net_arch": net_arch,
                "activation_fn": activation_fn,
                "lstm_hidden_size": config.get("network", {}).get("lstm_hidden_size", 256),
                "n_lstm_layers": config.get("network", {}).get("n_lstm_layers", 1),
            }
            model = RecurrentPPO(
                "MlpLstmPolicy",
                envs,
                learning_rate=training_cfg.get("learning_rate", 0.0003),
                n_steps=training_cfg.get("n_steps", 2048),
                batch_size=training_cfg.get("batch_size", 128),
                n_epochs=training_cfg.get("n_epochs", 10),
                gamma=training_cfg.get("gamma", 0.99),
                gae_lambda=training_cfg.get("gae_lambda", 0.95),
                clip_range=training_cfg.get("clip_range", 0.2),
                ent_coef=training_cfg.get("ent_coef", 0.01),
                vf_coef=training_cfg.get("vf_coef", 0.5),
                max_grad_norm=training_cfg.get("max_grad_norm", 0.5),
                policy_kwargs=policy_kwargs,
                verbose=0,
                device=device,
                seed=seed,
                tensorboard_log=os.path.join(fold_dir, "tb_logs"),
            )
            logger.info(f"  Fold {fold_idx}: RecurrentPPO (LSTM) on {device}")
        else:
            policy_kwargs = {
                "net_arch": [{"pi": net_arch, "vf": net_arch}],
                "activation_fn": activation_fn,
            }
            model = PPO(
                "MlpPolicy",
                envs,
                learning_rate=training_cfg.get("learning_rate", 0.0003),
                n_steps=training_cfg.get("n_steps", 2048),
                batch_size=training_cfg.get("batch_size", 2048),
                n_epochs=training_cfg.get("n_epochs", 10),
                gamma=training_cfg.get("gamma", 0.99),
                gae_lambda=training_cfg.get("gae_lambda", 0.95),
                clip_range=training_cfg.get("clip_range", 0.2),
                ent_coef=training_cfg.get("ent_coef", 0.01),
                vf_coef=training_cfg.get("vf_coef", 0.5),
                max_grad_norm=training_cfg.get("max_grad_norm", 0.5),
                target_kl=training_cfg.get("target_kl", 0.02),
                policy_kwargs=policy_kwargs,
                verbose=0,
                device=device,
                seed=seed,
                tensorboard_log=os.path.join(fold_dir, "tb_logs"),
            )
            logger.info(f"  Fold {fold_idx}: PPO standard on {device}")

        # Callbacks
        checkpoint_cb = CheckpointCallback(
            save_freq=max(timesteps // 10, 10000),
            save_path=os.path.join(fold_dir, "checkpoints"),
            name_prefix=f"fold_{fold_idx:02d}",
        )

        # Entraîner
        logger.info(f"  Fold {fold_idx}: Training {timesteps:,} timesteps...")
        model.learn(total_timesteps=timesteps, callback=[checkpoint_cb], progress_bar=True)

        # Sauvegarder le modèle
        model_path = os.path.join(fold_dir, "model")
        model.save(model_path)
        envs.save(os.path.join(fold_dir, "vecnormalize.pkl"))

        # Évaluer sur le test set (geler les stats de normalisation)
        logger.info(f"  Fold {fold_idx}: Evaluating on test period...")
        envs.training = False
        envs.norm_reward = False
        metrics = evaluate_on_test(
            model, envs, test_data, macro_data, config, features_precomputed=True
        )

        # Sauvegarder métriques
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
            logger.info("  ⚡ V9: Cleaning up Shared Memory resources")
            shm_manager.cleanup()

    return metrics


def _annualization_factor(interval: str) -> float:
    """Facteur d'annualisation du Sharpe ratio selon l'intervalle des données."""
    factors = {"1h": 252 * 6.5, "1d": 252, "1wk": 52, "1mo": 12}
    return np.sqrt(factors.get(interval, 252))


def evaluate_on_test(
    model, train_envs, test_data, macro_data, config, features_precomputed=False
) -> dict:
    """Évalue le modèle sur la période de test.

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

    while not done:
        # Normaliser l'observation comme pendant le training
        obs_normalized = train_envs.normalize_obs(obs.reshape(1, -1)).flatten()
        action, _ = model.predict(obs_normalized, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        equity_curve.append(info["equity"])

    # Calculer métriques
    equity_series = pd.Series(equity_curve)
    returns = equity_series.pct_change().dropna()

    total_return = (equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0]
    max_drawdown = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max()

    sharpe_ratio = 0.0
    interval = config.get("data", {}).get("interval", "1h")
    if len(returns) > 1 and returns.std() > 0:
        sharpe_ratio = (returns.mean() / returns.std()) * _annualization_factor(interval)

    return {
        "total_return": float(total_return),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown),
        "total_trades": info.get("total_trades", 0),
        "winning_trades": info.get("winning_trades", 0),
        "losing_trades": info.get("losing_trades", 0),
        "final_equity": float(equity_series.iloc[-1]),
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

    # 1. Télécharger données
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

    # 1.5 TURBO INIT: Pré-calculer les features maintenant (1 seule fois)
    logger.info("🚀 Turbo Init: Pre-computing technical features for all tickers...")
    feature_engineer = FeatureEngineer()

    # Paralléliser si possible, mais sinon simple boucle
    # (TA-Lib release le GIL, donc Threading pourrait marcher, mais on reste simple)
    processed_count = 0
    for ticker, df in data.items():
        # Sauvegarder l'index original (DatetimeIndex) pour éviter sa perte via Polars
        original_index = df.index
        
        # Calculer les features
        feat_df = feature_engineer.calculate_all_features(df)
        
        # Restaurer l'index si nécessaire
        if len(feat_df) == len(original_index):
            feat_df.index = original_index
        else:
            logger.warning(f"Feature engineering changed row count for {ticker} ({len(original_index)} -> {len(feat_df)}). Index mismatch possible.")
            
        data[ticker] = feat_df
        processed_count += 1

    logger.info(f"✅ Features computed for {processed_count} tickers. RAM usage will increase.")

    # 2. Télécharger données macro
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
        macro_data = pd.DataFrame() # Empty DF

    if macro_data.empty:
        logger.warning("No macro data available, proceeding without")
        macro_data = None

    # 3. Générer splits walk-forward
    wf_cfg = config.get("walk_forward", {})
    logger.info("Generating walk-forward splits...")
    splits = generate_walk_forward_splits(
        data,
        train_years=wf_cfg.get("train_years", 1),
        test_months=wf_cfg.get("test_months", 6),
        step_months=wf_cfg.get("step_months", 6),
    )

    if not splits:
        logger.error("No walk-forward splits generated (not enough data)")
        return

    # 4. Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    algo = "recurrent_ppo" if use_recurrent else "ppo"
    output_dir = f"models/walk_forward_{algo}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # 5. Entraîner chaque fold
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

        # Ensemble : entraîner n_ensemble modèles avec des seeds différents
        if n_ensemble > 1:
            fold_metrics_list = []
            for ens_idx in range(n_ensemble):
                seed = 42 + ens_idx * 1000
                metrics = train_single_fold(
                    train_data=split["train"],
                    test_data=split["test"],
                    macro_data=fold_macro,
                    config=config,
                    fold_idx=fold_idx * n_ensemble + ens_idx,
                    output_dir=output_dir,
                    use_recurrent=use_recurrent,
                    seed=seed,
                )
                fold_metrics_list.append(metrics)

            # Moyenne des métriques de l'ensemble
            avg_metrics = {
                key: float(np.mean([m[key] for m in fold_metrics_list]))
                for key in fold_metrics_list[0]
            }
            avg_metrics["ensemble_size"] = n_ensemble
            avg_metrics["fold_idx"] = fold_idx
            avg_metrics["train_period"] = f"{split['train_start']}->{split['train_end']}"
            avg_metrics["test_period"] = f"{split['test_start']}->{split['test_end']}"
            all_metrics.append(avg_metrics)
        else:
            metrics = train_single_fold(
                train_data=split["train"],
                test_data=split["test"],
                macro_data=fold_macro,
                config=config,
                fold_idx=fold_idx,
                output_dir=output_dir,
                use_recurrent=use_recurrent,
                seed=42,
            )
            metrics["fold_idx"] = fold_idx
            metrics["train_period"] = f"{split['train_start']}->{split['train_end']}"
            metrics["test_period"] = f"{split['test_start']}->{split['test_end']}"
            all_metrics.append(metrics)

    # 6. Rapport final
    logger.info("\n" + "=" * 70)
    logger.info("WALK-FORWARD RESULTS")
    logger.info("=" * 70)

    returns = [m["total_return"] for m in all_metrics]
    sharpes = [m["sharpe_ratio"] for m in all_metrics]
    drawdowns = [m["max_drawdown"] for m in all_metrics]

    for m in all_metrics:
        logger.info(
            f"  Fold {m['fold_idx']}: "
            f"Return={m['total_return']:+.2%} | "
            f"Sharpe={m['sharpe_ratio']:.2f} | "
            f"MaxDD={m['max_drawdown']:.2%} | "
            f"Period: {m['test_period']}"
        )

    logger.info(f"\nAGGREGATED:")
    logger.info(f"  Avg Return:  {np.mean(returns):+.2%} (std: {np.std(returns):.2%})")
    logger.info(f"  Avg Sharpe:  {np.mean(sharpes):.2f} (std: {np.std(sharpes):.2f})")
    logger.info(f"  Avg MaxDD:   {np.mean(drawdowns):.2%}")
    logger.info(f"  Win Folds:   {sum(1 for r in returns if r > 0)}/{len(returns)}")
    logger.info(f"  Cumulative:  {np.prod([1 + r for r in returns]) - 1:+.2%}")

    # Sauvegarder résultats
    results = {
        "algorithm": algo,
        "n_folds": len(all_metrics),
        "n_ensemble": n_ensemble,
        "output_dir": output_dir,
        "avg_return": float(np.mean(returns)),
        "avg_sharpe": float(np.mean(sharpes)),
        "avg_max_drawdown": float(np.mean(drawdowns)),
        "cumulative_return": float(np.prod([1 + r for r in returns]) - 1),
        "win_fold_ratio": sum(1 for r in returns if r > 0) / len(returns),
        "folds": all_metrics,
    }

    results_path = os.path.join(output_dir, "walk_forward_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved: {results_path}")
    logger.info("=" * 70)

    return results


if __name__ == "__main__":
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
