#!/usr/bin/env python3
# scripts/optimize_hyperparams.py
"""Optimisation automatique des hyperparamètres avec Optuna.

Teste différentes combinaisons de learning_rate, batch_size, gamma,
ent_coef, net_arch etc. et trouve la meilleure configuration.

Souvent +20% de performance gratuite par rapport à des valeurs manuelles.

Usage:
    pip install optuna
    python scripts/optimize_hyperparams.py --config config/training_config_v8.yaml --n-trials 50
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

try:
    import optuna
except ImportError:
    print("Optuna non installé. Exécuter: pip install optuna")
    sys.exit(1)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from core.universal_environment_v8_lstm import UniversalTradingEnvV8LSTM
from core.macro_data import MacroDataFetcher
from core.data_fetcher import download_data
from core.data_pipeline import DataSplitter
from core.utils import setup_logging

logger = setup_logging(__name__, 'optimize_hyperparams.log')


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_objective(train_data, val_data, macro_data, base_config):
    """Crée la fonction objectif Optuna."""

    def objective(trial: optuna.Trial) -> float:
        # Hyperparamètres à optimiser
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [1024, 2048, 4096, 8192])
        n_steps = trial.suggest_categorical('n_steps', [1024, 2048, 4096])
        n_epochs = trial.suggest_int('n_epochs', 5, 30)
        gamma = trial.suggest_float('gamma', 0.95, 0.999)
        gae_lambda = trial.suggest_float('gae_lambda', 0.9, 0.99)
        clip_range = trial.suggest_float('clip_range', 0.1, 0.3)
        ent_coef = trial.suggest_float('ent_coef', 0.001, 0.2, log=True)
        vf_coef = trial.suggest_float('vf_coef', 0.1, 0.9)

        # Architecture réseau
        n_layers = trial.suggest_int('n_layers', 2, 4)
        layer_size = trial.suggest_categorical('layer_size', [256, 512, 1024])
        net_arch = [layer_size] * n_layers

        # Environnement rewards
        reward_scaling = trial.suggest_float('reward_scaling', 0.5, 3.0)
        drawdown_penalty_factor = trial.suggest_float('drawdown_penalty_factor', 1.0, 5.0)

        # Créer environnement d'entraînement
        env_kwargs = {k: v for k, v in base_config.get('environment', {}).items()}
        env_kwargs['mode'] = 'train'
        env_kwargs['reward_scaling'] = reward_scaling
        env_kwargs['drawdown_penalty_factor'] = drawdown_penalty_factor

        n_envs = base_config.get('training', {}).get('n_envs', 8)

        def make_env():
            def _init():
                return Monitor(UniversalTradingEnvV8LSTM(
                    data=train_data, macro_data=macro_data, **env_kwargs
                ))
            return _init

        if n_envs > 1:
            envs = SubprocVecEnv([make_env() for _ in range(n_envs)])
        else:
            envs = DummyVecEnv([make_env()])

        envs = VecNormalize(
            envs, norm_obs=True, norm_reward=True,
            clip_obs=10.0, clip_reward=10.0, gamma=gamma,
        )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        activation_fn = torch.nn.Tanh

        policy_kwargs = {
            'net_arch': [{'pi': net_arch, 'vf': net_arch}],
            'activation_fn': activation_fn,
        }

        model = PPO(
            'MlpPolicy', envs,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=0, device=device,
        )

        # Entraîner avec budget réduit (pour aller vite)
        optim_timesteps = base_config.get('optuna', {}).get('timesteps_per_trial', 1_000_000)

        try:
            model.learn(total_timesteps=optim_timesteps)
        except Exception as e:
            logger.warning(f"Trial {trial.number} échoué: {e}")
            envs.close()
            return float('-inf')

        # Évaluer sur validation set
        env_kwargs_eval = {k: v for k, v in base_config.get('environment', {}).items()}
        env_kwargs_eval['mode'] = 'backtest'
        env_kwargs_eval['seed'] = 42
        env_kwargs_eval['reward_scaling'] = reward_scaling
        env_kwargs_eval['drawdown_penalty_factor'] = drawdown_penalty_factor

        eval_env = UniversalTradingEnvV8LSTM(
            data=val_data, macro_data=macro_data, **env_kwargs_eval
        )

        obs, _ = eval_env.reset()
        done = False
        equity_curve = [eval_env.initial_balance]

        while not done:
            obs_norm = envs.normalize_obs(obs.reshape(1, -1)).flatten()
            action, _ = model.predict(obs_norm, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            equity_curve.append(info['equity'])

        envs.close()

        # Objectif : Sharpe Ratio (pénalisé par drawdown)
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()

        total_return = (equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0]
        max_drawdown = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max()

        sharpe = 0.0
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 6.5)

        # Pénaliser les drawdowns > 15%
        score = sharpe
        if max_drawdown > 0.15:
            score -= (max_drawdown - 0.15) * 10

        logger.info(
            f"Trial {trial.number}: "
            f"Return={total_return:+.2%} | Sharpe={sharpe:.2f} | "
            f"MaxDD={max_drawdown:.2%} | Score={score:.2f}"
        )

        return score

    return objective


def optimize(config_path: str, n_trials: int = 50):
    """Lance l'optimisation Optuna."""
    logger.info("=" * 70)
    logger.info("OPTUNA HYPERPARAMETER OPTIMIZATION")
    logger.info("=" * 70)

    config = load_config(config_path)

    # Télécharger données
    logger.info("Downloading data...")
    data = download_data(
        tickers=config['data']['tickers'],
        period=config['data'].get('period', '5y'),
        interval=config['data'].get('interval', '1h'),
    )

    # Split train/val (test_ratio minimal car on n'en a pas besoin ici)
    logger.info("Splitting data...")
    splits = DataSplitter.split(data, train_ratio=0.7, val_ratio=0.25, test_ratio=0.05)
    train_data = splits.train
    val_data = splits.val

    # Macro data
    logger.info("Downloading macro data...")
    macro_fetcher = MacroDataFetcher()
    ref_df = data[list(data.keys())[0]]
    macro_data = macro_fetcher.fetch_all(
        start_date=str(ref_df.index[0].date()),
        end_date=str(ref_df.index[-1].date()),
        interval=config['data'].get('interval', '1h'),
    )
    if macro_data.empty:
        macro_data = None

    # Créer l'étude Optuna
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    study_name = f"ploutos_v8_{timestamp}"

    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )

    objective = create_objective(train_data, val_data, macro_data, config)

    logger.info(f"Starting {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Résultats
    logger.info("\n" + "=" * 70)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("=" * 70)
    logger.info(f"Best Score: {study.best_value:.4f}")
    logger.info(f"Best Params:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")

    # Sauvegarder
    output_dir = f"models/optuna_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'best_params.json'), 'w') as f:
        json.dump({
            'best_score': study.best_value,
            'best_params': study.best_params,
            'n_trials': n_trials,
        }, f, indent=2)

    logger.info(f"\nResults saved: {output_dir}/best_params.json")
    return study


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Optuna Hyperparameter Optimization')
    parser.add_argument('--config', type=str, default='config/training_config_v8.yaml')
    parser.add_argument('--n-trials', type=int, default=50)

    args = parser.parse_args()
    optimize(config_path=args.config, n_trials=args.n_trials)
