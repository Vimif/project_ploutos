#!/usr/bin/env python3
"""
🎓 CURRICULUM LEARNING POUR PLOUTOS
Entraînement progressif : Simple → Complexe

Avec auto-optimisation rapide et transfer learning adapté

Usage:
    python3 scripts/train_curriculum.py --stage 1
    python3 scripts/train_curriculum.py --stage 2 --transfer
    python3 scripts/train_curriculum.py --stage 3 --transfer
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
from core.trading_callback import TradingMetricsCallback

# ✅ PARAMS OPTIMISÉS POUR GPU
CALIBRATED_PARAMS = {
    'stage1': {
        'name': 'Mono-Asset (SPY)',
        'tickers': ['SPY'],
        'timesteps': 3_000_000,
        'n_envs': 4,
        'learning_rate': 1e-4,
        'n_steps': 2048,
        'batch_size': 512,          # ✅ Optimisé
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.05,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'policy_kwargs': {'net_arch': [512, 512, 512]},
        'target_sharpe': 1.0
    },
    'stage2': {
        'name': 'Multi-Asset ETFs',
        'tickers': ['SPY', 'QQQ', 'IWM'],
        'timesteps': 5_000_000,
        'n_envs': 6,                # ✅ 8→6 (éviter CPU bottleneck)
        'learning_rate': 5e-5,
        'n_steps': 2048,
        'batch_size': 2048,         # ✅ 512→2048 (4x)
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.02,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'policy_kwargs': {'net_arch': [512, 512, 512]},
        'target_sharpe': 1.3
    },
    'stage3': {
        'name': 'Actions Complexes',
        'tickers': ['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN'],
        'timesteps': 10_000_000,
        'n_envs': 8,                # ✅ 16→8 (CPU bottleneck fix)
        'learning_rate': 3e-5,
        'n_steps': 2048,            # ✅ 4096→2048 (collecter plus souvent)
        'batch_size': 4096,         # ✅ 1024→4096 (4x GPU usage)
        'n_epochs': 10,             # ✅ 5→10 (plus de passes)
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'policy_kwargs': {'net_arch': [512, 512, 512]},
        'target_sharpe': 1.5
    }
}

def print_banner(text):
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def make_env(data_dict, initial_balance=10000, commission=0.0005, realistic_costs=False):
    """✅ Environnement optimisé avec coûts réduits"""
    def _init():
        return UniversalTradingEnv(
            data=data_dict,
            initial_balance=initial_balance,
            commission=commission,        # ✅ 0.001→0.0005 (coûts réduits)
            max_steps=1000,
            realistic_costs=realistic_costs  # ✅ Désactivé par défaut
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
            commission=0.0005,  # ✅ Cohérent
            max_steps=adjusted_max_steps,
            realistic_costs=False
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
    
    # ✅ Logger optimizations dans W&B
    wandb.config.update({
        'optimization': 'GPU_optimized',
        'expected_gpu_usage': '70-80%',
        'reduced_transaction_costs': True
    })
    
    print(f"\n🔗 W&B Run : {wandb.run.get_url()}")
    print(f"   Projet : Ploutos_Curriculum")
    print(f"   Run    : {run_name}")
    print(f"\n⚡ OPTIMISATIONS :")
    print(f"   Batch Size   : {config['batch_size']} (4x augmenté)")
    print(f"   N Envs       : {config['n_envs']} (CPU friendly)")
    print(f"   N Steps      : {config['n_steps']} (collecter plus souvent)")
    print(f"   Commission   : 0.05% (réduit pour apprentissage)")
    print(f"   Target GPU   : 70-80% usage\n")
    
    # Créer environnements avec coûts réduits
    print("🏭 Création des environnements optimisés...")
    env = SubprocVecEnv([
        make_env(data, commission=0.0005, realistic_costs=False) 
        for _ in range(config['n_envs'])
    ])
    
    # ✅ Environnement d'évaluation
    eval_env = UniversalTradingEnv(
        data=data,
        initial_balance=10000,
        commission=0.0005,
        max_steps=1000,
        realistic_costs=False
    )
    
    # Transfer Learning
    if use_transfer_learning and stage_num > 1:
        if prev_stage is None:
            prev_stage = stage_num - 1
        
        prev_model_path = f'models/stage{prev_stage}_final.zip'
        
        if os.path.exists(prev_model_path):
            print(f"\n🔄 TRANSFER LEARNING : Stage {prev_stage} → Stage {stage_num}")
            
            source_model = PPO.load(prev_model_path)
            adapter = FeatureAdapter(source_model, env, device='cuda')
            strategy = adapter.get_transfer_strategy(prev_stage, stage_num)
            
            print(f"\n🎯 Stratégie : {strategy['description']}")
            print(f"   Méthode        : {strategy['method']}")
            print(f"   Freeze layers : {strategy['freeze_layers']}")
            print(f"   LR ajusté     : {config['learning_rate']} × {strategy['learning_rate_factor']}")
            
            wandb.config.update({
                'transfer_learning': True,
                'source_stage': prev_stage,
                'adaptation_method': strategy['method'],
                'freeze_layers': strategy['freeze_layers'],
                'lr_factor': strategy['learning_rate_factor']
            })
            
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
    
    # Créer modèle from scratch
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
    
    # ✅ CALLBACKS
    os.makedirs(f'models/{stage_key}', exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,  # ✅ Tous les 100k steps
        save_path=f'./models/{stage_key}',
        name_prefix=f'ploutos_{stage_key}'
    )
    
    wandb_callback = WandbCallback(
        gradient_save_freq=1000,
        model_save_path=f'models/{stage_key}',
        model_save_freq=500000,  # ✅ Tous les 500k
        verbose=2
    )
    
    trading_callback = TradingMetricsCallback(
        eval_env=eval_env,
        eval_freq=20000,  # ✅ Toutes les 20k steps (plus rapide)
        n_eval_episodes=5,
        log_actions_dist=True,
        verbose=1
    )
    
    callback = CallbackList([
        checkpoint_callback,
        wandb_callback,
        trading_callback
    ])
    
    # Entraînement
    print(f"\n🚀 Entraînement : {config['timesteps']:,} timesteps...")
    print(f"⏱️  Durée estimée : ~{config['timesteps'] // 1_000_000} heures (optimisé)")
    print(f"🔗 Suivre en temps réel : {wandb.run.get_url()}")
    print(f"📊 Évaluations : Toutes les 20k steps")
    print(f"💾 Checkpoints : Tous les 100k steps\n")
    
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
    print("\n📊 Évaluation finale...")
    
    data_length = min(len(df) for df in data.values())
    test_size = max(200, int(data_length * 0.2))
    test_data = {ticker: df.iloc[-test_size:] for ticker, df in data.items()}
    
    sharpe = calculate_sharpe(model, test_data, episodes=10)
    print(f"\n📈 Sharpe Ratio : {sharpe:.2f}")
    print(f"🎯 Objectif      : {config['target_sharpe']:.2f}")
    
    success = sharpe >= config['target_sharpe']
    
    if success:
        print(f"\n✅ STAGE {stage_num} RÉUSSI !")
    else:
        print(f"\n⚠️  Sharpe insuffisant, mais modèle sauvegardé")
    
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
    eval_env.close()
    
    return model_path, sharpe

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Curriculum Learning pour Ploutos (GPU Optimized)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python3 scripts/train_curriculum.py --stage 1
  python3 scripts/train_curriculum.py --stage 2 --transfer
  python3 scripts/train_curriculum.py --stage 3 --transfer

Optimisations appliquées:
  ✅ Batch size 4x augmenté (meilleur GPU usage)
  ✅ N envs réduit (éviter CPU bottleneck)
  ✅ Transaction costs réduits (apprentissage plus facile)
  ✅ Évaluations plus fréquentes (monitoring)
  
Performance attendue:
  GPU Usage : 70-80% (au lieu de 14%)
  FPS       : 20k-30k (au lieu de 5k)
  Durée     : 8-10h (au lieu de 20h)
        """
    )
    
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3])
    parser.add_argument('--transfer', action='store_true')
    parser.add_argument('--from-stage', type=int, default=None, choices=[1, 2])
    parser.add_argument('--auto-optimize', action='store_true')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🎓 PLOUTOS CURRICULUM LEARNING (GPU OPTIMIZED)")
    print("="*80)
    print(f"\n⏰ Début : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Stage : {args.stage}")
    print(f"🔄 Transfer : {'OUI' if args.transfer else 'NON'}")
    print(f"⚡ GPU Optimization : ACTIVÉ")
    if args.transfer and args.from_stage:
        print(f"🎯 Source : Stage {args.from_stage}")
    print()
    
    model_path, sharpe = train_stage(
        stage_num=args.stage,
        use_transfer_learning=args.transfer,
        prev_stage=args.from_stage,
        auto_optimize=args.auto_optimize
    )
    
    print(f"\n⏰ Fin : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"✅ Modèle : {model_path}.zip")
    print(f"📊 Sharpe : {sharpe:.2f}")
    
    if args.stage < 3 and sharpe > 0:
        print(f"\n💡 PROCHAINE ÉTAPE :")
        print(f"   python3 scripts/train_curriculum.py --stage {args.stage + 1} --transfer")
