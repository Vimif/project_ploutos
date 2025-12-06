#!/usr/bin/env python3
"""
🎓 CURRICULUM LEARNING POUR PLOUTOS
Entraînement progressif : Simple → Complexe

Avec auto-optimisation rapide et transfer learning adapté

Usage:
    python3 scripts/train_curriculum.py --stage 1
    python3 scripts/train_curriculum.py --stage 2 --transfer
    python3 scripts/train_curriculum.py --stage 3 --transfer
    python3 scripts/train_curriculum.py --auto-continue  # ✅ Lance tout
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
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from wandb.integration.sb3 import WandbCallback

from core.data_fetcher import UniversalDataFetcher
from core.universal_environment import UniversalTradingEnv
from core.feature_adapter import FeatureAdapter
from core.trading_callback import TradingMetricsCallback
from core.performance_monitor import PerformanceMonitor

# ✅ PARAMS OPTIMISÉS V8 - RESEARCH-BASED (Reddit RL, ArXiv, Towards AI, SB3)
CALIBRATED_PARAMS = {
    'stage1': {
        'name': 'Mono-Asset (SPY)',
        'tickers': ['SPY'],
        'timesteps': 10_000_000,
        'n_envs': 16,
        'learning_rate': 3e-5,
        'n_steps': 4096,  # ✅ 8192 → 4096 (balance)
        'batch_size': 2048,  # ✅ 4096 → 2048 (updates fréquents)
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.3,  # ✅ 0.2 → 0.3 (adaptabilité)
        'ent_coef': 0.1,  # ✅ 0.01 → 0.1 (exploration active)
        'vf_coef': 0.3,  # ✅ 0.5 → 0.3 (focus policy)
        'max_grad_norm': 0.5,
        'policy_kwargs': {'net_arch': [512, 512, 512]},
        'target_sharpe': 1.0
    },
    'stage2': {
        'name': 'Multi-Asset ETFs',
        'tickers': ['SPY', 'QQQ', 'IWM'],
        'timesteps': 20_000_000,
        'n_envs': 24,
        'learning_rate': 2e-5,
        'n_steps': 4096,
        'batch_size': 4096,  # ✅ 16384 → 4096 (balance GPU/convergence)
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.3,
        'ent_coef': 0.05,  # ✅ Réduit progressivement
        'vf_coef': 0.3,
        'max_grad_norm': 0.5,
        'policy_kwargs': {'net_arch': [512, 512, 512]},
        'target_sharpe': 1.3
    },
    'stage3': {
        'name': 'Actions Complexes',
        'tickers': ['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN'],
        'timesteps': 30_000_000,
        'n_envs': 32,
        'learning_rate': 1e-5,
        'n_steps': 4096,
        'batch_size': 8192,  # ✅ 32768 → 8192 (safe VRAM + convergence)
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.3,
        'ent_coef': 0.01,
        'vf_coef': 0.3,
        'max_grad_norm': 0.5,
        'policy_kwargs': {'net_arch': [512, 512, 512]},
        'target_sharpe': 1.5
    }
}

def print_banner(text):
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def make_env(data_dict, initial_balance=10000, commission=0.0001, realistic_costs=False):
    """✅ Commission réduite à 0.01% (0.0001)"""
    def _init():
        return UniversalTradingEnv(
            data=data_dict,
            initial_balance=initial_balance,
            commission=commission,
            max_steps=2000,
            realistic_costs=realistic_costs
        )
    return _init

def calculate_sharpe(model, data_dict, episodes=10):
    returns = []
    data_length = min(len(df) for df in data_dict.values())
    
    if data_length < 150:
        print(f"\n⚠️  Données trop courtes ({data_length}), skip Sharpe")
        return 0.0
    
    adjusted_max_steps = min(1000, data_length - 110)
    
    for _ in range(episodes):
        env = UniversalTradingEnv(
            data=data_dict,
            initial_balance=10000,
            commission=0.0001,
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
    
    wandb.config.update({
        'optimization': 'research_based_v8',
        'vectorization': 'DummyVecEnv',
        'sources': 'Reddit_RL+ArXiv+TowardsAI+SB3',
        'numpy_precompute': True,
        'reward_function': 'fixed_normalized',
        'commission_reduced': '0.01%',
        'max_steps_increased': 2000,
        'expected_sharpe': '0.6-1.0',
        'expected_fps': '15k'
    })
    
    print(f"\n🔗 W&B Run : {wandb.run.get_url()}")
    print(f"   Projet : Ploutos_Curriculum")
    print(f"   Run    : {run_name}")
    print(f"\n⚡ OPTIMISATIONS V8 (RESEARCH-BASED) :")
    print(f"   Sources         : Reddit RL, ArXiv, Towards AI, SB3")
    print(f"   Vectorization   : DummyVecEnv")
    print(f"   N Envs          : {config['n_envs']}")
    print(f"   Batch Size      : {config['batch_size']:,} (✅ updates fréquents)")
    print(f"   N Steps         : {config['n_steps']:,} (✅ balance)")
    print(f"   Learning Rate   : {config['learning_rate']:.0e}")
    print(f"   Entropy Coef    : {config['ent_coef']} (✅ exploration active)")
    print(f"   VF Coef         : {config['vf_coef']} (✅ focus policy)")
    print(f"   Clip Range      : {config['clip_range']} (✅ adaptabilité)")
    print(f"   Timesteps       : {config['timesteps']:,}")
    print(f"   Target Sharpe   : {config['target_sharpe']}\n")
    
    # ✅ CRÉER ENVIRONNEMENTS (DummyVecEnv)
    print("🏭 Création environnements (DummyVecEnv)...")
    env = DummyVecEnv([
        make_env(data, commission=0.0001, realistic_costs=False)
        for _ in range(config['n_envs'])
    ])
    
    eval_env = UniversalTradingEnv(
        data=data,
        initial_balance=10000,
        commission=0.0001,
        max_steps=2000,
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
    
    # ✅ CALLBACKS (fréquence réduite)
    os.makedirs(f'models/{stage_key}', exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=f'./models/{stage_key}',
        name_prefix=f'ploutos_{stage_key}'
    )
    
    wandb_callback = WandbCallback(
        gradient_save_freq=5000,
        model_save_path=f'models/{stage_key}',
        model_save_freq=500000,
        verbose=2
    )
    
    trading_callback = TradingMetricsCallback(
        eval_env=eval_env,
        eval_freq=50000,
        n_eval_episodes=5,
        log_actions_dist=True,
        verbose=1
    )
    
    perf_monitor = PerformanceMonitor(
        log_freq=10000,
        verbose=1
    )
    
    callback = CallbackList([
        checkpoint_callback,
        wandb_callback,
        trading_callback,
        perf_monitor
    ])
    
    # Entraînement
    print(f"\n🚀 Entraînement : {config['timesteps']:,} timesteps...")
    print(f"⏱️  Durée estimée : ~{int(config['timesteps'] / 15000 / 60)} minutes (FPS ~15k)")
    print(f"🔗 Suivre : {wandb.run.get_url()}")
    print(f"📊 Monitoring : Toutes les 10k steps")
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
        description='Curriculum Learning pour Ploutos (V8 - Research-Based)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python3 scripts/train_curriculum.py --stage 1
  python3 scripts/train_curriculum.py --stage 2 --transfer
  python3 scripts/train_curriculum.py --stage 3 --transfer
  python3 scripts/train_curriculum.py --auto-continue  # ✅ Lance tout

Optimisations V8 (RESEARCH-BASED):
  Sources: Reddit r/RL, ArXiv, Towards AI, SB3 docs
  ✅ ent_coef: 0.01→0.1 (exploration active)
  ✅ vf_coef: 0.5→0.3 (focus policy)
  ✅ clip_range: 0.2→0.3 (adaptabilité)
  ✅ batch_size: 4096→2048 (updates fréquents)
  ✅ n_steps: 8192→4096 (balance)
  ✅ Expected Sharpe: 0.6-1.0 (viable)
  ✅ Expected Win Rate: 20-30%
  
Durées attendues (FPS ~15k):
  Stage 1: ~12min (10M timesteps)
  Stage 2: ~22min (20M timesteps)
  Stage 3: ~33min (30M timesteps)
  --auto-continue: ~70min (stages 1+2+3)
        """
    )
    
    parser.add_argument('--stage', type=int, default=None, choices=[1, 2, 3],
                        help='Stage à entraîner (1, 2 ou 3)')
    parser.add_argument('--transfer', action='store_true',
                        help='Utiliser transfer learning du stage précédent')
    parser.add_argument('--from-stage', type=int, default=None, choices=[1, 2],
                        help='Stage source pour transfer learning')
    parser.add_argument('--auto-continue', action='store_true',
                        help='✅ Lance tous les stages automatiquement (1→2→3)')
    parser.add_argument('--auto-optimize', action='store_true',
                        help='Mode auto-optimisation (expérimental)')
    
    args = parser.parse_args()
    
    # Validation
    if not args.auto_continue and args.stage is None:
        parser.error("--stage requis (ou utiliser --auto-continue)")
    
    print("\n" + "="*80)
    print("🎓 PLOUTOS CURRICULUM LEARNING (V8 - RESEARCH-BASED)")
    print("="*80)
    print(f"\n⏰ Début : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ✅ MODE AUTO-CONTINUE
    if args.auto_continue:
        print("🚀 MODE AUTO-CONTINUE : Stages 1 → 2 → 3")
        print("⏱️  Durée totale : ~70 minutes")
        print("☕ Temps d'un repas !\n")
        
        results = {}
        
        # Stage 1
        print("\n" + "#"*80)
        print("# STAGE 1/3 : MONO-ASSET (SPY)")
        print("#"*80)
        model_path_1, sharpe_1 = train_stage(
            stage_num=1,
            use_transfer_learning=False,
            prev_stage=None,
            auto_optimize=args.auto_optimize
        )
        results['stage1'] = {'model': model_path_1, 'sharpe': sharpe_1}
        
        # Stage 2 avec transfer
        print("\n" + "#"*80)
        print("# STAGE 2/3 : MULTI-ASSET ETFs")
        print("#"*80)
        model_path_2, sharpe_2 = train_stage(
            stage_num=2,
            use_transfer_learning=True,
            prev_stage=1,
            auto_optimize=args.auto_optimize
        )
        results['stage2'] = {'model': model_path_2, 'sharpe': sharpe_2}
        
        # Stage 3 avec transfer
        print("\n" + "#"*80)
        print("# STAGE 3/3 : ACTIONS COMPLEXES")
        print("#"*80)
        model_path_3, sharpe_3 = train_stage(
            stage_num=3,
            use_transfer_learning=True,
            prev_stage=2,
            auto_optimize=args.auto_optimize
        )
        results['stage3'] = {'model': model_path_3, 'sharpe': sharpe_3}
        
        # Résumé final
        print("\n" + "="*80)
        print("🎆 CURRICULUM COMPLET TERMINÉ !")
        print("="*80)
        print(f"\n⏰ Fin : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n📈 RÉSULTATS FINAUX :\n")
        for stage, data in results.items():
            status = "✅" if data['sharpe'] >= CALIBRATED_PARAMS[stage]['target_sharpe'] else "⚠️"
            print(f"  {status} {stage.upper()} : Sharpe = {data['sharpe']:.2f} (objectif: {CALIBRATED_PARAMS[stage]['target_sharpe']:.2f})")
            print(f"      Modèle : {data['model']}.zip")
        
        print(f"\n🎯 MODÈLE FINAL : {results['stage3']['model']}.zip")
        print(f"🚀 Prêt pour le déploiement !\n")
        
    else:
        # ✅ MODE SINGLE STAGE
        print(f"📊 Stage : {args.stage}")
        print(f"🔄 Transfer : {'OUI' if args.transfer else 'NON'}")
        print(f"⚡ V8 : Research-Based (Reddit RL, ArXiv, Towards AI, SB3)")
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
