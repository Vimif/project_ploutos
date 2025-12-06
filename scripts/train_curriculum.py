#!/usr/bin/env python3
"""
🎓 CURRICULUM LEARNING POUR PLOUTOS
Entraînement progressif : Simple → Complexe

Avec auto-optimisation rapide et transfer learning adapté

Usage:
    python3 scripts/train_curriculum.py --stage 1
    python3 scripts/train_curriculum.py --stage 2 --transfer  # ✅ Avec transfer learning
    python3 scripts/train_curriculum.py --stage 2             # Sans transfer learning
"""

import sys
sys.path.insert(0, '.')

import os
import json
import wandb
import numpy as np
import pandas as pd
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from wandb.integration.sb3 import WandbCallback

from core.data_fetcher import UniversalDataFetcher
from core.universal_environment import UniversalTradingEnv
from core.feature_adapter import FeatureAdapter
from core.trading_callback import TradingMetricsCallback  # ✅ NOUVEAU

# Params calibrés
CALIBRATED_PARAMS = {
    'stage1': {
        'name': 'Mono-Asset (SPY)',
        'tickers': ['SPY'],
        'timesteps': 3_000_000,
        'n_envs': 4,
        'learning_rate': 1e-4,
        'n_steps': 2048,
        'batch_size': 256,
        'n_epochs': 5,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.05,
        'vf_coef': 0.5,
        'max_grad_norm': 0.3,
        'policy_kwargs': {'net_arch': [512, 512, 512]},
        'target_sharpe': 1.0
    },
    'stage2': {
        'name': 'Multi-Asset ETFs',
        'tickers': ['SPY', 'QQQ', 'IWM'],
        'timesteps': 5_000_000,
        'n_envs': 8,
        'learning_rate': 5e-5,
        'n_steps': 2048,
        'batch_size': 512,
        'n_epochs': 5,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.02,
        'vf_coef': 0.5,
        'max_grad_norm': 0.3,
        'policy_kwargs': {'net_arch': [512, 512, 512]},
        'target_sharpe': 1.3
    },
    'stage3': {
        'name': 'Actions Complexes',
        'tickers': ['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN'],
        'timesteps': 10_000_000,
        'n_envs': 16,
        'learning_rate': 3e-5,
        'n_steps': 4096,
        'batch_size': 1024,
        'n_epochs': 5,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.3,
        'policy_kwargs': {'net_arch': [512, 512, 512]},
        'target_sharpe': 1.5
    }
}

def print_banner(text):
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def make_env(data_dict):
    def _init():
        return UniversalTradingEnv(
            data=data_dict,
            initial_balance=10000,
            commission=0.001,
            max_steps=1000
        )
    return _init

def calculate_sharpe(model, data_dict, episodes=10):
    returns = []
    data_length = min(len(df) for df in data_dict.values())
    
    if data_length < 150:
        print(f"\n⚠️  Données trop courtes ({data_length}), skip Sharpe")
        return 0.0
    
    adjusted_max_steps = min(500, data_length - 110)
    
    for _ in range(episodes):
        env = UniversalTradingEnv(
            data=data_dict,
            initial_balance=10000,
            commission=0.001,
            max_steps=adjusted_max_steps
        )
        
        obs, _ = env.reset()
        done = False
        episode_return = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward
        
        returns.append(episode_return)
    
    returns = np.array(returns)
    
    if len(returns) == 0 or returns.std() == 0:
        return 0
    
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    return sharpe

def train_stage(stage_num, use_transfer_learning=False, prev_stage=None, auto_optimize=False):
    """
    Entraîne un stage avec transfer learning optionnel
    
    Args:
        stage_num: Stage à entraîner (1, 2, 3)
        use_transfer_learning: Si True, utilise feature adapter
        prev_stage: Stage source pour transfer (auto si None)
        auto_optimize: Si True, lance optimisation hyperparamètres
    """
    
    stage_key = f'stage{stage_num}'
    config = CALIBRATED_PARAMS[stage_key].copy()
    
    print_banner(f"🎓 STAGE {stage_num} : {config['name']}")
    
    # Récupérer données
    print("📥 Téléchargement des données...")
    fetcher = UniversalDataFetcher()
    data = fetcher.bulk_fetch(config['tickers'], interval='1h')
    
    print(f"✅ {len(data)}/{len(config['tickers'])} tickers récupérés")
    
    # ✅ INITIALISER W&B
    transfer_suffix = "_Transfer" if use_transfer_learning else ""
    run_name = f"Stage{stage_num}_{config['name'].replace(' ', '_')}{transfer_suffix}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    wandb.init(
        project="Ploutos_Curriculum",
        name=run_name,
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True
    )
    
    print(f"\n🔗 W&B Run : {wandb.run.get_url()}")
    print(f"   Projet : Ploutos_Curriculum")
    print(f"   Run    : {run_name}\n")
    
    # Créer environnements
    print("🏭 Création des environnements...")
    env = SubprocVecEnv([make_env(data) for _ in range(config['n_envs'])])
    
    # ✅ Créer environnement d'évaluation (single env)
    eval_env = UniversalTradingEnv(
        data=data,
        initial_balance=10000,
        commission=0.001,
        max_steps=1000
    )
    
    # Transfer Learning avec Feature Adapter
    if use_transfer_learning and stage_num > 1:
        
        # Déterminer stage source
        if prev_stage is None:
            prev_stage = stage_num - 1
        
        prev_model_path = f'models/stage{prev_stage}_final.zip'
        
        if os.path.exists(prev_model_path):
            print(f"\n🔄 TRANSFER LEARNING : Stage {prev_stage} → Stage {stage_num}")
            
            # Charger modèle source
            source_model = PPO.load(prev_model_path)
            
            # Créer adapter
            adapter = FeatureAdapter(source_model, env, device='cuda')
            
            # Récupérer stratégie recommandée
            strategy = adapter.get_transfer_strategy(prev_stage, stage_num)
            
            print(f"\n🎯 Stratégie : {strategy['description']}")
            print(f"   Méthode        : {strategy['method']}")
            print(f"   Freeze layers : {strategy['freeze_layers']}")
            print(f"   LR ajusté     : {config['learning_rate']} × {strategy['learning_rate_factor']}")
            
            # Logger strategy dans W&B
            wandb.config.update({
                'transfer_learning': True,
                'source_stage': prev_stage,
                'adaptation_method': strategy['method'],
                'freeze_layers': strategy['freeze_layers'],
                'lr_factor': strategy['learning_rate_factor']
            })
            
            # Adapter modèle
            model = adapter.adapt(
                method=strategy['method'],
                freeze_layers=strategy['freeze_layers'],
                learning_rate=config['learning_rate'] * strategy['learning_rate_factor']
            )
            
            print(f"✅ Transfer learning appliqué !\n")
            
        else:
            print(f"\n⚠️  Modèle source introuvable : {prev_model_path}")
            print("   Création modèle from scratch...\n")
            use_transfer_learning = False
    
    # Créer modèle from scratch si pas de transfer
    if not use_transfer_learning or stage_num == 1:
        print("🧠 Création modèle from scratch...")
        
        policy_kwargs = config.pop('policy_kwargs')
        target_sharpe = config.pop('target_sharpe')
        timesteps = config.pop('timesteps')
        n_envs = config.pop('n_envs')
        name = config.pop('name')
        tickers = config.pop('tickers')
        
        model = PPO(
            'MlpPolicy',
            env,
            verbose=1,
            tensorboard_log=f'./logs/{stage_key}',
            device='cuda',
            policy_kwargs=policy_kwargs,
            **config
        )
        
        # Restaurer
        config['policy_kwargs'] = policy_kwargs
        config['target_sharpe'] = target_sharpe
        config['timesteps'] = timesteps
        config['n_envs'] = n_envs
        config['name'] = name
        config['tickers'] = tickers
    
    # ✅ CALLBACKS : Checkpoint + WandbCallback + TradingMetricsCallback
    os.makedirs(f'models/{stage_key}', exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 * stage_num,
        save_path=f'./models/{stage_key}',
        name_prefix=f'ploutos_{stage_key}'
    )
    
    wandb_callback = WandbCallback(
        gradient_save_freq=1000,
        model_save_path=f'models/{stage_key}',
        model_save_freq=100000,
        verbose=2
    )
    
    # ✅ NOUVEAU : Trading Metrics Callback
    trading_callback = TradingMetricsCallback(
        eval_env=eval_env,
        eval_freq=10000,          # Évaluer toutes les 10k steps
        n_eval_episodes=5,        # 5 épisodes par évaluation
        log_actions_dist=True,    # Logger distribution actions
        verbose=1
    )
    
    # Combiner callbacks
    callback = CallbackList([
        checkpoint_callback,
        wandb_callback,
        trading_callback  # ✅ Ajouté
    ])
    
    # Entraînement
    print(f"\n🚀 Entraînement : {config['timesteps']:,} timesteps...")
    print(f"⏱️  Durée estimée : ~{config['timesteps'] // 500_000} heures sur RTX 3080")
    print(f"🔗 Suivre en temps réel : {wandb.run.get_url()}")
    print(f"📊 Évaluations : Toutes les 10k steps (Sharpe, Max DD, Win Rate)\n")
    
    model.learn(
        total_timesteps=config['timesteps'],
        callback=callback,
        progress_bar=True
    )
    
    # Sauvegarder
    model_path = f'models/{stage_key}_final'
    model.save(model_path)
    print(f"\n✅ Modèle sauvegardé : {model_path}.zip")
    
    wandb.save(f'{model_path}.zip')
    
    # Évaluation finale
    print("\n📊 Évaluation finale du modèle...")
    
    data_length = min(len(df) for df in data.values())
    test_size = max(200, int(data_length * 0.2))
    
    print(f"  Taille données test : {test_size} lignes")
    
    test_data = {ticker: df.iloc[-test_size:] for ticker, df in data.items()}
    
    sharpe = calculate_sharpe(model, test_data, episodes=10)
    print(f"\n📈 Sharpe Ratio : {sharpe:.2f}")
    print(f"🎯 Objectif      : {config['target_sharpe']:.2f}")
    
    success = sharpe >= config['target_sharpe']
    
    if success:
        print(f"\n✅ STAGE {stage_num} RÉUSSI !")
    else:
        print(f"\n⚠️  Sharpe insuffisant, mais on continue...")
    
    # Logger résultats finaux
    wandb.log({
        'final/sharpe_ratio': sharpe,
        'final/target_sharpe': config['target_sharpe'],
        'final/success': success,
        'final/timesteps': config['timesteps']
    })
    
    wandb.run.summary['sharpe_ratio'] = sharpe
    wandb.run.summary['success'] = success
    wandb.run.summary['stage'] = stage_num
    
    wandb.finish()
    env.close()
    eval_env.close()  # ✅ Fermer eval_env
    
    return model_path, sharpe

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Curriculum Learning pour Ploutos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python3 scripts/train_curriculum.py --stage 1
  python3 scripts/train_curriculum.py --stage 2 --transfer
  python3 scripts/train_curriculum.py --stage 3 --transfer

Métriques W&B loggées :
  Standard (SB3):
    - rollout/ep_rew_mean : Reward moyen
    - train/policy_loss   : Loss policy
    - train/value_loss    : Loss value function
    - train/entropy       : Entropie
  
  Trading (custom):
    - eval/mean_sharpe       : Sharpe Ratio
    - eval/mean_max_dd       : Max Drawdown
    - eval/mean_win_rate     : Win Rate
    - eval/profit_factor     : Profit Factor
    - eval/action_*_pct      : Distribution actions
    - eval/mean_final_portfolio : Portfolio final moyen
        """
    )
    
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3],
                        help='Stage à exécuter (1, 2, ou 3)')
    parser.add_argument('--transfer', action='store_true',
                        help='Active transfer learning avec feature adapter')
    parser.add_argument('--from-stage', type=int, default=None, choices=[1, 2],
                        help='Stage source pour transfer (auto = stage-1 si omis)')
    parser.add_argument('--auto-optimize', action='store_true',
                        help='Active auto-optimisation hyperparamètres (pas encore implémenté)')
    
    args = parser.parse_args()    
    print("\n" + "="*80)
    print("🎓 PLOUTOS CURRICULUM LEARNING")
    print("="*80)
    print(f"\n⏰ Début : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Stage : {args.stage}")
    print(f"🔄 Transfer Learning : {'OUI' if args.transfer else 'NON'}")
    if args.transfer and args.from_stage:
        print(f"🎯 Source : Stage {args.from_stage}")
    print()
    
    # Exécution
    model_path, sharpe = train_stage(
        stage_num=args.stage,
        use_transfer_learning=args.transfer,
        prev_stage=args.from_stage,
        auto_optimize=args.auto_optimize
    )
    
    print(f"\n⏰ Fin : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"✅ Modèle final : {model_path}.zip")
    print(f"📊 Sharpe Ratio : {sharpe:.2f}")
    
    if args.stage < 3 and sharpe > 0:
        print(f"\n💡 PROCHAINE ÉTAPE :")
        print(f"   Sans transfer : python3 scripts/train_curriculum.py --stage {args.stage + 1}")
        print(f"   Avec transfer : python3 scripts/train_curriculum.py --stage {args.stage + 1} --transfer")
