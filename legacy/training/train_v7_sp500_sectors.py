#!/usr/bin/env python3
# training/train_v7_sp500_sectors.py
"""Entrainement V7 - Top S&P 500 par secteur GICS

Selection dynamique des meilleures actions (Sharpe ratio) de chaque
secteur GICS du S&P 500, puis entrainement PPO sur ces ~22 tickers.

Usage:
    python training/train_v7_sp500_sectors.py
    python training/train_v7_sp500_sectors.py --force-rescan
    python training/train_v7_sp500_sectors.py --config config/training_config_v7_sp500.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings

warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

import json
import os
from datetime import datetime

import torch
import yaml
from core.universal_environment_v6_better_timing import UniversalTradingEnvV6BetterTiming
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from core.data_fetcher import download_data
from core.data_pipeline import DataSplitter
from core.sp500_scanner import SP500Scanner
from core.utils import setup_logging

logger = setup_logging(__name__, "training_v7_sp500.log")


# ======================================================================
# Helpers
# ======================================================================


def load_config(config_path: str) -> dict:
    """Charger la configuration YAML."""
    if not os.path.exists(config_path):
        logger.error(f"Config {config_path} non trouve")
        return None
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(data, config, rank, mode="train"):
    """Creer un environnement (pour SubprocVecEnv)."""

    def _init():
        env_kwargs = dict(config["environment"])
        env_kwargs["mode"] = mode
        env = UniversalTradingEnvV6BetterTiming(
            data=data,
            **env_kwargs,
        )
        env = Monitor(env)
        return env

    return _init


def run_sector_scan(config: dict, force_rescan: bool = False):
    """Execute le scan S&P 500 ou charge le cache.

    Returns:
        (tickers, scan_results)
    """
    scan_cfg = config.get("sector_scan", {})

    if not scan_cfg.get("enabled", False):
        tickers = config["data"].get("tickers", [])
        if not tickers:
            raise ValueError("sector_scan disabled but no tickers in config")
        return tickers, None

    scanner = SP500Scanner(
        cache_dir="data/sp500_cache",
        lookback_days=scan_cfg.get("lookback_days", 252),
    )

    cached = (
        None
        if force_rescan
        else scanner.load_cached_results(
            max_age_days=scan_cfg.get("cache_max_age_days", 30),
        )
    )

    if cached is not None:
        logger.info("Using cached scan results")
        scan_results = cached
    else:
        logger.info("Running S&P 500 sector scan...")
        logger.info(f"  stocks_per_sector: {scan_cfg.get('stocks_per_sector', 2)}")
        logger.info(f"  lookback_days: {scan_cfg.get('lookback_days', 252)}")
        scan_results = scanner.scan_sectors(
            stocks_per_sector=scan_cfg.get("stocks_per_sector", 2),
            max_workers=scan_cfg.get("parallel_workers", 5),
        )
        scanner.save_results(scan_results, scan_cfg.get("scan_results_path"))

    tickers = scanner.get_top_stocks(scan_results)
    logger.info(f"Selected {len(tickers)} stocks across {len(scan_results['sectors'])} sectors")
    for sector, stocks in scan_results["sectors"].items():
        logger.info(f"  {sector}: {', '.join(stocks)}")

    return tickers, scan_results


# ======================================================================
# Main training
# ======================================================================


def train_v7_model(config_path: str, force_rescan: bool = False):
    """Pipeline complet d'entrainement V7."""
    sys.stdout.reconfigure(encoding="utf-8")

    logger.info("=" * 70)
    logger.info("DEMARRAGE ENTRAINEMENT V7 - S&P 500 SECTORS")
    logger.info("=" * 70)

    # 1. Config
    config = load_config(config_path)
    if config is None:
        return
    logger.info(f"Config chargee: {config_path}")

    # 2. Scan S&P 500
    tickers, scan_results = run_sector_scan(config, force_rescan)
    config["data"]["tickers"] = tickers

    logger.info("\nTicker Selection:")
    logger.info(f"  Total: {len(tickers)}")
    logger.info(f"  Tickers: {', '.join(tickers)}")

    # 3. Telecharger donnees
    logger.info("\nTelechargement des donnees...")
    try:
        data = download_data(
            tickers=tickers,
            period=config["data"]["period"],
            interval=config["data"]["interval"],
        )
        if not data or len(data) == 0:
            raise ValueError("Aucune donnee recuperee")

        logger.info(f"{len(data)} tickers charges")
        for ticker, df in list(data.items())[:5]:
            logger.info(f"  {ticker}: {len(df)} bougies ({df.index[0]} -> {df.index[-1]})")
        if len(data) > 5:
            logger.info(f"  ... et {len(data) - 5} autres")

        # Extract training data date range for OOS enforcement
        all_starts = []
        all_ends = []
        for _ticker, df in data.items():
            if len(df) > 0:
                all_starts.append(str(df.index[0]))
                all_ends.append(str(df.index[-1]))
        training_data_start = min(all_starts) if all_starts else None
        training_data_end = max(all_ends) if all_ends else None
        logger.info(f"  Training data range: {training_data_start} -> {training_data_end}")

    except Exception as e:
        logger.error(f"Erreur telechargement: {e}")
        import traceback

        traceback.print_exc()
        return

    # 3b. Split temporel train/val/test
    split_cfg = config.get("data_split", {})
    train_ratio = split_cfg.get("train_ratio", 0.6)
    val_ratio = split_cfg.get("val_ratio", 0.2)
    test_ratio = split_cfg.get("test_ratio", 0.2)

    logger.info(f"\nSplit temporel: train={train_ratio} / val={val_ratio} / test={test_ratio}")
    splits = DataSplitter.split(data, train_ratio, val_ratio, test_ratio)
    DataSplitter.validate_no_overlap(splits)

    train_data = splits.train
    val_data = splits.val
    # test_data = splits.test  # reservé pour backtest OOS

    logger.info(
        f"  Train: {splits.info['train']['n_bars']} bars ({splits.info['train']['start']} -> {splits.info['train']['end']})"
    )
    logger.info(
        f"  Val:   {splits.info['val']['n_bars']} bars ({splits.info['val']['start']} -> {splits.info['val']['end']})"
    )
    logger.info(
        f"  Test:  {splits.info['test']['n_bars']} bars ({splits.info['test']['start']} -> {splits.info['test']['end']})"
    )

    # 4. Environnements paralleles
    logger.info(f"\nCreation de {config['training']['n_envs']} environnements paralleles...")
    try:
        envs = SubprocVecEnv(
            [
                make_env(train_data, config, i, mode="train")
                for i in range(config["training"]["n_envs"])
            ]
        )
        envs = VecNormalize(
            envs,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=config["training"]["gamma"],
        )
        logger.info("Environnements crees et normalises")
        logger.info(f"  Observation space: {envs.observation_space.shape}")
        logger.info(f"  Action space: {envs.action_space}")

    except Exception as e:
        logger.error(f"Erreur creation environnements: {e}")
        import traceback

        traceback.print_exc()
        return

    # 5. Modele PPO
    logger.info("\nCreation du modele PPO...")

    policy_kwargs = {
        "net_arch": [
            {
                "pi": config["network"]["net_arch"],
                "vf": config["network"]["net_arch"],
            }
        ],
        "activation_fn": (
            torch.nn.Tanh
            if config["network"].get("activation_fn", "tanh") == "tanh"
            else torch.nn.ReLU
        ),
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")

    model = PPO(
        "MlpPolicy",
        envs,
        learning_rate=config["training"]["learning_rate"],
        n_steps=config["training"]["n_steps"],
        batch_size=config["training"]["batch_size"],
        n_epochs=config["training"]["n_epochs"],
        gamma=config["training"]["gamma"],
        gae_lambda=config["training"]["gae_lambda"],
        clip_range=config["training"]["clip_range"],
        ent_coef=config["training"]["ent_coef"],
        vf_coef=config["training"]["vf_coef"],
        max_grad_norm=config["training"]["max_grad_norm"],
        target_kl=config["training"]["target_kl"],
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        tensorboard_log="./runs/v7_sp500/",
    )

    logger.info("Modele cree")
    logger.info(f"  Architecture: {config['network']['net_arch']}")
    logger.info(f"  Params: {sum(p.numel() for p in model.policy.parameters()):,}")

    # 6. Callbacks
    logger.info("\nConfiguration des callbacks...")

    os.makedirs(config["checkpoint"]["save_path"], exist_ok=True)
    os.makedirs(config["eval"]["best_model_save_path"], exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=config["checkpoint"]["save_freq"],
        save_path=config["checkpoint"]["save_path"],
        name_prefix="ploutos_v7_sp500",
    )

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        min_evals=50,
        verbose=1,
    )

    # Eval env (val data, mode eval)
    eval_envs = SubprocVecEnv([make_env(val_data, config, 0, mode="eval")])
    eval_envs = VecNormalize(
        eval_envs,
        norm_obs=True,
        norm_reward=False,  # pas de normalisation reward en eval
        clip_obs=10.0,
        gamma=config["training"]["gamma"],
    )

    eval_callback = EvalCallback(
        eval_envs,
        callback_after_eval=stop_callback,
        eval_freq=config["eval"]["eval_freq"],
        n_eval_episodes=config["eval"]["n_eval_episodes"],
        best_model_save_path=config["eval"]["best_model_save_path"],
        log_path="./logs/v7_sp500_eval/",
        deterministic=True,
        render=False,
        verbose=1,
    )

    callbacks = [checkpoint_callback, eval_callback]
    logger.info("Callbacks configures")

    # 7. ENTRAINEMENT
    logger.info("\n" + "=" * 70)
    logger.info("DEBUT DE L'ENTRAINEMENT V7 - S&P 500 SECTORS")
    logger.info("=" * 70)
    logger.info(f"Total timesteps: {config['training']['total_timesteps']:,}")
    logger.info(f"Tickers: {len(tickers)}")
    logger.info("Duree estimee: ~10-14h sur RTX 3080")
    logger.info("=" * 70 + "\n")

    try:
        model.learn(
            total_timesteps=config["training"]["total_timesteps"],
            callback=callbacks,
            progress_bar=True,
        )
        logger.info("Entrainement termine avec succes")
    except KeyboardInterrupt:
        logger.warning("Entrainement interrompu par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur pendant l'entrainement: {e}")
        import traceback

        traceback.print_exc()
        return

    # 8. Sauvegarde
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = f"models/v7_sp500/ploutos_v7_sp500_{timestamp}.zip"
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)

    model.save(final_model_path)
    logger.info(f"Modele final sauvegarde: {final_model_path}")

    # VecNormalize
    vecnorm_path = final_model_path.replace(".zip", "_vecnormalize.pkl")
    envs.save(vecnorm_path)
    logger.info(f"VecNormalize sauvegarde: {vecnorm_path}")

    # Config
    config_save_path = final_model_path.replace(".zip", "_config.json")
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info(f"Config sauvegardee: {config_save_path}")

    # Metadata V7 (tickers + secteurs + Sharpe ratios)
    metadata = {
        "version": "v7_sp500",
        "tickers": tickers,
        "n_tickers": len(tickers),
        "scan_date": scan_results["scan_date"] if scan_results else None,
        "sectors": scan_results["sectors"] if scan_results else None,
        "sharpe_ratios": scan_results["sharpe_ratios"] if scan_results else None,
        "training_date": timestamp,
        "training_data_start": training_data_start,
        "training_data_end": training_data_end,
        "observation_space_dim": envs.observation_space.shape[0],
        "total_timesteps": config["training"]["total_timesteps"],
        "network_arch": config["network"]["net_arch"],
        "data_split": splits.info,
    }
    metadata_path = final_model_path.replace(".zip", "_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Metadata V7 sauvegardee: {metadata_path}")

    # Resume
    logger.info("\n" + "=" * 70)
    logger.info("ENTRAINEMENT V7 TERMINE")
    logger.info("=" * 70)
    logger.info(f"Modele: {final_model_path}")
    logger.info(f"Metadata: {metadata_path}")
    logger.info("TensorBoard: tensorboard --logdir runs/v7_sp500/")
    logger.info(f"\nBacktest: python scripts/backtest_v6.py --model {final_model_path}")
    logger.info(f"Paper: python scripts/run_trader_v6.py --model {final_model_path} --paper")
    logger.info("=" * 70)

    return model, envs


# ======================================================================
# CLI
# ======================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entrainement V7 - S&P 500 Sectors")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config_v7_sp500.yaml",
        help="Chemin config YAML",
    )
    parser.add_argument(
        "--force-rescan",
        action="store_true",
        help="Force un nouveau scan S&P 500 (ignore le cache)",
    )
    args = parser.parse_args()

    train_v7_model(config_path=args.config, force_rescan=args.force_rescan)
