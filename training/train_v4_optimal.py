#!/usr/bin/env python3
# training/train_v4_optimal.py
"""🚀 Entraînement V4 OPTIMAL - Configuration pour Performance Maximale

Améliorations vs V3:
- Entropy coef: 0.01 → 0.08 (exploration++)
- Max trades/day: 3 → 10 (liberté++)
- Buy pct: 15% → 25% (capital++)
- Reward bonus trades réussis
- Pénalités réduites
- Timesteps: 10M → 20M

Auteur: Ploutos AI Team
Date: Dec 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings('ignore', message='.*Gym has been unmaintained.*')

import os
import yaml
import torch
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement
)
from stable_baselines3.common.monitor import Monitor

# ✅ Import environnement OPTIMAL
from core.universal_environment_v4_optimal import UniversalTradingEnvV4Optimal
from core.data_fetcher import download_data
from core.utils import setup_logging

logger = setup_logging(__name__, 'training_v4_optimal.log')


def load_config(config_path: str) -> dict:
    """Charger la configuration"""
    if not os.path.exists(config_path):
        logger.error(f"Config {config_path} non trouvé")
        return None
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def make_env(data, config, rank):
    """Créer un environnement"""
    def _init():
        env = UniversalTradingEnvV4Optimal(
            data=data,
            **config['environment']
        )
        env = Monitor(env)
        return env
    
    return _init


def train_optimal_model(config_path: str):
    """
    Entraînement V4 OPTIMAL
    """
    logger.info("="*70)
    logger.info("🚀 DÉMARRAGE ENTRAÎNEMENT V4 OPTIMAL")
    logger.info("="*70)
    
    # 1. Charger config
    config = load_config(config_path)
    if config is None:
        return
    
    logger.info(f"✅ Configuration chargée")
    logger.info(f"  • Entropy coef: {config['training']['ent_coef']}")
    logger.info(f"  • Max trades/day: {config['environment']['max_trades_per_day']}")
    logger.info(f"  • Buy pct: {config['environment']['buy_pct']*100}%")
    logger.info(f"  • Total timesteps: {config['training']['total_timesteps']:,}")
    
    # 2. Télécharger données
    logger.info(f"\n📊 Téléchargement des données...")
    
    try:
        data = download_data(
            tickers=config['data']['tickers'],
            period=config['data']['period'],
            interval=config['data']['interval']
        )
        
        if not data or len(data) == 0:
            raise ValueError("❌ Aucune donnée récupérée")
        
        logger.info(f"✅ {len(data)} tickers chargés")
        
        for ticker, df in list(data.items())[:3]:
            logger.info(f"  {ticker}: {len(df)} bougies ({df.index[0]} → {df.index[-1]})")
        if len(data) > 3:
            logger.info(f"  ... et {len(data)-3} autres tickers")
        
    except Exception as e:
        logger.error(f"❌ Erreur téléchargement données: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Créer environnements parallèles
    logger.info(f"\n🏭 Création de {config['training']['n_envs']} environnements parallèles...")
    
    try:
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
        
    except Exception as e:
        logger.error(f"❌ Erreur création environnements: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. Créer modèle PPO
    logger.info(f"\n🧠 Création du modèle PPO...")
    
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
    
    if device == 'cuda':
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    
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
        tensorboard_log='./runs/v4_optimal/'
    )
    
    logger.info("✅ Modèle créé")
    logger.info(f"   Architecture: {config['network']['net_arch']}")
    logger.info(f"   Params: {sum(p.numel() for p in model.policy.parameters()):,}")
    
    # 5. Callbacks
    logger.info(f"\n📎 Configuration des callbacks...")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=config['checkpoint']['save_path'],
        name_prefix='ploutos_v4_optimal'
    )
    
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        min_evals=50,
        verbose=1
    )
    
    eval_callback = EvalCallback(
        envs,
        callback_after_eval=stop_callback,
        eval_freq=config['eval']['eval_freq'],
        n_eval_episodes=config['eval']['n_eval_episodes'],
        best_model_save_path=config['eval']['best_model_save_path'],
        log_path='./logs/v4_optimal_eval/',
        deterministic=True,
        render=False,
        verbose=1
    )
    
    callbacks = [checkpoint_callback, eval_callback]
    
    logger.info("✅ Callbacks configurés")
    
    # 6. ENTRAÎNEMENT
    logger.info("\n" + "="*70)
    logger.info("🏋️ DÉBUT DE L'ENTRAÎNEMENT V4 OPTIMAL")
    logger.info("="*70)
    logger.info(f"Total timesteps: {config['training']['total_timesteps']:,}")
    logger.info(f"Batch size: {config['training']['batch_size']:,}")
    logger.info(f"Updates: {config['training']['total_timesteps'] // (config['training']['n_steps'] * config['training']['n_envs']):,}")
    logger.info(f"Durée estimée: ~6-8h sur GPU (RTX 3080)")
    logger.info("="*70 + "\n")
    
    try:
        model.learn(
            total_timesteps=config['training']['total_timesteps'],
            callback=callbacks,
            progress_bar=True
        )
        
        logger.info("✅ Entraînement terminé avec succès")
        
    except KeyboardInterrupt:
        logger.warning("⚠️ Entraînement interrompu par l'utilisateur")
    
    except Exception as e:
        logger.error(f"❌ Erreur pendant l'entraînement: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 7. Sauvegarder modèle final
    final_model_path = f"models/v4_optimal/ploutos_v4_optimal_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    
    model.save(final_model_path)
    logger.info(f"✅ Modèle final sauvegardé: {final_model_path}")
    
    # Sauvegarder VecNormalize
    envs.save(final_model_path.replace('.zip', '_vecnormalize.pkl'))
    logger.info("✅ VecNormalize sauvegardé")
    
    # 8. Sauvegarder config
    import json
    config_save_path = final_model_path.replace('.zip', '_config.json')
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"✅ Config sauvegardée: {config_save_path}")
    
    logger.info("\n" + "="*70)
    logger.info("🎉 ENTRAÎNEMENT V4 OPTIMAL TERMINÉ")
    logger.info("="*70)
    logger.info(f"Modèle final: {final_model_path}")
    logger.info(f"TensorBoard: tensorboard --logdir runs/v4_optimal/")
    logger.info("\n✅ Pour tester: python scripts/backtest_reliability.py --model {}".format(final_model_path))
    
    return model, envs


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Entraînement V4 OPTIMAL')
    parser.add_argument('--config', type=str, default='config/training_config_v4_optimal.yaml', help='Chemin config YAML')
    
    args = parser.parse_args()
    
    train_optimal_model(config_path=args.config)
