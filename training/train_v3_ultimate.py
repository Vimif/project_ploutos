#!/usr/bin/env python3
# training/train_v3_ultimate.py
"""Script d'Entraînement V3 ULTIMATE - Ploutos Trading IA

Optimisations:
- Environnement V4 ultra-réaliste
- 50+ features avancées
- Architecture neuronale profonde
- Hyperparams optimisés
- Data augmentation
- Callbacks avancés
- Weights & Biases tracking
- Multi-GPU support
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import yaml
import wandb
import torch
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement
)
from stable_baselines3.common.monitor import Monitor

from core.universal_environment_v4 import UniversalTradingEnvV4
from core.data_fetcher import download_data
from core.utils import setup_logging

logger = setup_logging(__name__, 'training_v3.log')


class WandbCallback:
    """Callback pour logger sur Weights & Biases"""
    
    def __init__(self, verbose=0):
        self.verbose = verbose
    
    def __call__(self, locals_, globals_):
        # Log métriques
        if 'infos' in locals_ and len(locals_['infos']) > 0:
            info = locals_['infos'][0]
            
            if 'equity' in info:
                wandb.log({
                    'equity': info['equity'],
                    'total_return': info.get('total_return', 0),
                    'total_trades': info.get('total_trades', 0)
                })
        
        return True


def load_config(config_path: str = 'config/training_config_v3.yaml') -> dict:
    """Charger la configuration"""
    if not os.path.exists(config_path):
        logger.warning(f"Config {config_path} non trouvé, utilisation config par défaut")
        return get_default_config()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_default_config() -> dict:
    """Configuration par défaut"""
    return {
        'training': {
            'total_timesteps': 10_000_000,
            'n_envs': 16,
            'batch_size': 8192,
            'n_steps': 4096,
            'n_epochs': 20,
            'learning_rate': 5e-5,
            'gamma': 0.995,
            'gae_lambda': 0.98,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'target_kl': 0.015
        },
        'environment': {
            'initial_balance': 100000,
            'commission': 0.0,
            'sec_fee': 0.0000221,
            'finra_taf': 0.000145,
            'max_steps': 3000,
            'buy_pct': 0.15,
            'slippage_model': 'realistic',
            'spread_bps': 2.0,
            'market_impact_factor': 0.0001,
            'max_position_pct': 0.25,
            'reward_scaling': 1.0,
            'use_sharpe_penalty': True,
            'use_drawdown_penalty': True,
            'max_trades_per_day': 4,
            'min_holding_period': 5
        },
        'data': {
            'tickers': [
                'NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN',
                'SPY', 'QQQ', 'VOO', 'XLE', 'XLF'
            ],
            'period': '5y',
            'interval': '1h'
        },
        'network': {
            'net_arch': [1024, 512, 256],
            'activation_fn': 'tanh'
        },
        'wandb': {
            'enabled': True,
            'project': 'Ploutos_Trading_V3_ULTIMATE',
            'entity': None
        }
    }


def make_env(data, config, rank):
    """Créer un environnement"""
    def _init():
        env = UniversalTradingEnvV4(
            data=data,
            **config['environment']
        )
        env = Monitor(env)
        return env
    
    return _init


def train_ultimate_model(config_path: str = None):
    """
    Entraînement V3 ULTIMATE
    
    Args:
        config_path: Chemin vers fichier config YAML
    """
    logger.info("="*70)
    logger.info("🚀 DÉMARRAGE ENTRAÎNEMENT V3 ULTIMATE")
    logger.info("="*70)
    
    # 1. Charger config
    config = load_config(config_path) if config_path else get_default_config()
    logger.info(f"✅ Configuration chargée")
    
    # 2. Setup Weights & Biases
    if config['wandb']['enabled']:
        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            config=config,
            name=f"ploutos_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        logger.info("✅ Weights & Biases initialisé")
    
    # 3. Télécharger données
    logger.info("📊 Téléchargement des données...")
    data = download_data(
        tickers=config['data']['tickers'],
        period=config['data']['period'],
        interval=config['data']['interval']
    )
    logger.info(f"✅ {len(data)} tickers chargés")
    
    # 4. Créer environnements parallèles
    logger.info(f"🏭 Création de {config['training']['n_envs']} environnements parallèles...")
    
    envs = SubprocVecEnv([
        make_env(data, config, i)
        for i in range(config['training']['n_envs'])
    ])
    
    # Normalisation
    envs = VecNormalize(
        envs,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=config['training']['gamma']
    )
    
    logger.info("✅ Environnements créés et normalisés")
    
    # 5. Créer modèle PPO
    logger.info("🧠 Création du modèle PPO...")
    
    # Architecture
    policy_kwargs = {
        'net_arch': [
            {'pi': config['network']['net_arch'], 'vf': config['network']['net_arch']}
        ],
        'activation_fn': torch.nn.Tanh if config['network']['activation_fn'] == 'tanh' else torch.nn.ReLU
    }
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"💻 Device: {device}")
    
    model = PPO(
        'MlpPolicy',
        envs,
        learning_rate=config['training']['learning_rate'],
        n_steps=config['training']['n_steps'],
        batch_size=config['training']['batch_size'],
        n_epochs=config['training']['n_epochs'],
        gamma=config['training']['gamma'],
        gae_lambda=config['training']['gae_lambda'],
        clip_range=config['training']['clip_range'],
        ent_coef=config['training']['ent_coef'],
        vf_coef=config['training']['vf_coef'],
        max_grad_norm=config['training']['max_grad_norm'],
        target_kl=config['training']['target_kl'],
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        tensorboard_log='./runs/v3_ultimate/'
    )
    
    logger.info("✅ Modèle créé")
    logger.info(f"   Architecture: {config['network']['net_arch']}")
    logger.info(f"   Params: {sum(p.numel() for p in model.policy.parameters()):,}")
    
    # 6. Callbacks
    logger.info("📎 Configuration des callbacks...")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path='./models/v3_checkpoints/',
        name_prefix='ploutos_v3'
    )
    
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        min_evals=50,
        verbose=1
    )
    
    eval_callback = EvalCallback(
        envs,
        callback_after_eval=stop_callback,
        eval_freq=10000,
        n_eval_episodes=5,
        best_model_save_path='./models/v3_best/',
        log_path='./logs/v3_eval/',
        deterministic=True,
        render=False,
        verbose=1
    )
    
    callbacks = [checkpoint_callback, eval_callback]
    
    logger.info("✅ Callbacks configurés")
    
    # 7. ENTRAÎNEMENT
    logger.info("="*70)
    logger.info("🏋️  DÉBUT DE L'ENTRAÎNEMENT")
    logger.info("="*70)
    logger.info(f"Total timesteps: {config['training']['total_timesteps']:,}")
    logger.info(f"Batch size: {config['training']['batch_size']:,}")
    logger.info(f"Updates: {config['training']['total_timesteps'] // (config['training']['n_steps'] * config['training']['n_envs']):,}")
    logger.info("="*70)
    
    try:
        model.learn(
            total_timesteps=config['training']['total_timesteps'],
            callback=callbacks,
            progress_bar=True
        )
        
        logger.info("✅ Entraînement terminé avec succès")
        
    except KeyboardInterrupt:
        logger.warning("⚠️  Entraînement interrompu par l'utilisateur")
    
    except Exception as e:
        logger.error(f"❌ Erreur pendant l'entraînement: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 8. Sauvegarder modèle final
    final_model_path = f"models/v3_ultimate/ploutos_v3_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    
    model.save(final_model_path)
    logger.info(f"✅ Modèle final sauvegardé: {final_model_path}")
    
    # Sauvegarder VecNormalize
    envs.save(final_model_path.replace('.zip', '_vecnormalize.pkl'))
    logger.info("✅ VecNormalize sauvegardé")
    
    # 9. Sauvegarder config
    import json
    config_save_path = final_model_path.replace('.zip', '_config.json')
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"✅ Config sauvegardée: {config_save_path}")
    
    # 10. Fermer W&B
    if config['wandb']['enabled']:
        wandb.finish()
    
    logger.info("="*70)
    logger.info("🎉 ENTRAÎNEMENT V3 ULTIMATE TERMINÉ")
    logger.info("="*70)
    
    return model, envs


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Entraînement V3 ULTIMATE')
    parser.add_argument('--config', type=str, default=None, help='Chemin config YAML')
    
    args = parser.parse_args()
    
    train_ultimate_model(config_path=args.config)
