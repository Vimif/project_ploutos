#!/usr/bin/env python3
"""
💪 PLOUTOS TRAINING V3 FIXED - SANS BUGS

Script d'entraînement avec environnement V3 CORRIGÉ

Corrections critiques:
- ✅ max_trades_per_day fonctionne (DAILY data)
- ✅ NO lookahead bias
- ✅ Rewards larges (-2.0 à +2.0)
- ✅ 115 features (au lieu de 107)

Config optimale:
- 64 envs parallèles
- 10M timesteps total
- Learning rate 3e-4
- Batch 4096
- 10 époques

Objectif:
- Backtest 90j > 90/100
- Backtest 365j > 80/100
- Return 365j > 20%
- Drawdown < 8%
- Trades/jour < 30

Auteur: Ploutos AI Team
Date: 9 Dec 2025
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
import warnings
warnings.filterwarnings('ignore')

from core.universal_environment_v3_fixed import UniversalTradingEnvV3Fixed

# W&B optional
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb non disponible, tracking désactivé")

print("="*80)
print("💪 PLOUTOS TRAINING V3 FIXED - SANS BUGS")
print("="*80)

def load_data(tickers, start_date, end_date):
    """Charge les données pour plusieurs tickers"""
    data = {}
    
    print(f"\n📡 Chargement données pour {len(tickers)} tickers...")
    print(f"   Période: {start_date} → {end_date}")
    
    for ticker in tickers:
        print(f"   {ticker}...", end=" ")
        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
            
            if df.empty or len(df) < 250:
                print("❌ Erreur (données insuffisantes)")
                continue
            
            data[ticker] = df
            print(f"✅ {len(df)} jours")
        except Exception as e:
            print(f"❌ Erreur: {e}")
    
    return data

def get_default_tickers():
    """Tickers par défaut (même que V2)"""
    return [
        # Tech Growth
        'NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN',
        # Indices
        'SPY', 'QQQ', 'VOO',
        # Sectoriels
        'XLE',  # Energy
        'XLF'   # Finance
    ]

def make_env(data, config):
    """Factory pour créer environnement"""
    def _init():
        env = UniversalTradingEnvV3Fixed(
            data=data,
            initial_balance=100000,
            commission=0.0001,
            max_steps=config['max_steps'],
            buy_pct=0.15,  # 15% par position
            max_trades_per_day=config['max_trades_per_day'],
            stop_loss_pct=0.05,  # -5%
            trailing_stop=True,
            take_profit_pct=0.15,  # +15%
            use_smart_sizing=True
        )
        return env
    return _init

def create_callbacks(config):
    """Crée les callbacks pour l'entraînement"""
    callbacks = []
    
    checkpoint_dir = Path(config['output_dir']) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=config['checkpoint_freq'],
        save_path=str(checkpoint_dir),
        name_prefix='ploutos_v3_fixed',
        save_replay_buffer=False,
        save_vecnormalize=False
    )
    callbacks.append(checkpoint_callback)
    
    if WANDB_AVAILABLE and config.get('use_wandb', False):
        wandb_callback = WandbCallback(
            model_save_path=str(checkpoint_dir),
            verbose=2
        )
        callbacks.append(wandb_callback)
    
    return CallbackList(callbacks)

def main():
    parser = argparse.ArgumentParser(
        description='Entraîner Ploutos V3 FIXED (SANS BUGS)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Entraînement standard 10M steps
  python3 scripts/train_v3_fixed.py
  
  # Avec W&B tracking
  python3 scripts/train_v3_fixed.py --wandb --project Ploutos_V3_FIXED
  
  # Tickers custom
  python3 scripts/train_v3_fixed.py --tickers NVDA MSFT AAPL SPY QQQ
  
  # Test rapide 1M steps
  python3 scripts/train_v3_fixed.py --steps 1000000 --envs 32
        """
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        default=10_000_000,
        help='Nombre total de steps (défaut: 10M)'
    )
    
    parser.add_argument(
        '--envs',
        type=int,
        default=64,
        help='Nombre d\'environnements parallèles (défaut: 64)'
    )
    
    parser.add_argument(
        '--tickers',
        nargs='+',
        default=None,
        help='Liste de tickers (défaut: 10 tickers)'
    )
    
    parser.add_argument(
        '--output',
        default='models/ploutos_v3_fixed.zip',
        help='Chemin sauvegarde modèle final'
    )
    
    parser.add_argument(
        '--output-dir',
        default='models/production_v3_fixed',
        help='Dossier pour checkpoints'
    )
    
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Activer tracking W&B'
    )
    
    parser.add_argument(
        '--project',
        default='Ploutos_V3_FIXED_Final',
        help='Nom projet W&B'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=1095,  # 3 ans
        help='Jours de données historiques (défaut: 1095 = 3 ans)'
    )
    
    parser.add_argument(
        '--max-trades-per-day',
        type=int,
        default=30,
        help='Limite trades/jour (défaut: 30)'
    )
    
    args = parser.parse_args()
    
    config = {
        'total_steps': args.steps,
        'n_envs': args.envs,
        'output_path': args.output,
        'output_dir': args.output_dir,
        'use_wandb': args.wandb and WANDB_AVAILABLE,
        'wandb_project': args.project,
        'max_trades_per_day': args.max_trades_per_day,
        'max_steps': 2000,  # Steps max par épisode
        'checkpoint_freq': 100_000  # Checkpoint tous les 100k steps
    }
    
    # ========================================
    # PHASE 1: CHARGEMENT DONNÉES
    # ========================================
    
    tickers = args.tickers if args.tickers else get_default_tickers()
    
    end = datetime.now()
    start = end - timedelta(days=args.days)
    
    print(f"\n📊 Période inclut:")
    if args.days >= 1000:
        print("   ✅ Crash 2022 (baisse -25%)")
        print("   ✅ Bull market 2023-2024")
        print("   ✅ Volatilité 2024")
    
    data = load_data(tickers, start, end)
    
    if len(data) < 3:
        print("\n❌ Erreur: Pas assez de tickers chargés")
        return
    
    print(f"\n✅ {len(data)} tickers prêts pour entraînement")
    
    # ========================================
    # PHASE 2: INITIALISATION W&B
    # ========================================
    
    if config['use_wandb']:
        wandb.init(
            project=config['wandb_project'],
            name=f"v3_fixed_{datetime.now().strftime('%Y%m%d_%H%M')}",
            config={
                'algorithm': 'PPO',
                'env': 'UniversalTradingEnvV3Fixed',
                'n_envs': config['n_envs'],
                'total_timesteps': config['total_steps'],
                'learning_rate': 3e-4,
                'batch_size': 4096,
                'n_steps': 2048,
                'n_epochs': 10,
                'tickers': tickers,
                'max_trades_per_day': config['max_trades_per_day'],
                'commission': 0.0001,
                'stop_loss': 0.05,
                'take_profit': 0.15,
                'position_size': 0.15,
                'smart_sizing': True,
                'trailing_stop': True,
                'lookahead': False,  # ✅ FIXED
                'reward_range': '[-2.0, 2.0]',  # ✅ FIXED
                'observation_size': 115  # ✅ FIXED
            },
            tags=['v3_fixed', 'no_lookahead', 'optimized']
        )
        print("✅ W&B initialisé")
    
    # ========================================
    # PHASE 3: CRÉATION ENVIRONNEMENTS
    # ========================================
    
    print(f"\n🏭 Création {config['n_envs']} environnements parallèles...")
    
    env = SubprocVecEnv([make_env(data, config) for _ in range(config['n_envs'])])
    
    print("✅ Environnements créés")
    
    # ========================================
    # PHASE 4: CRÉATION MODÈLE PPO
    # ========================================
    
    print("\n🧠 Création modèle PPO...")
    
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=4096,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,  # Exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[512, 512, 512],  # Policy network
                vf=[512, 512, 512]   # Value network
            ),
            activation_fn=torch.nn.ReLU
        ),
        verbose=1,
        device='cuda',
        tensorboard_log=None
    )
    
    print("✅ Modèle PPO créé")
    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space}")
    print(f"   Learning rate: 3e-4")
    print(f"   Batch size: 4096")
    print(f"   Device: cuda")
    
    # ========================================
    # PHASE 5: ENTRAÎNEMENT
    # ========================================
    
    print("\n" + "="*80)
    print("🚀 DÉMARRAGE ENTRAÎNEMENT")
    print("="*80)
    print(f"Total timesteps: {config['total_steps']:,}")
    print(f"Environnements: {config['n_envs']}")
    print(f"Checkpoints: tous les {config['checkpoint_freq']:,} steps")
    print(f"Sortie: {config['output_path']}")
    print("="*80 + "\n")
    
    callbacks = create_callbacks(config)
    
    try:
        model.learn(
            total_timesteps=config['total_steps'],
            callback=callbacks,
            progress_bar=True
        )
        
        print("\n" + "="*80)
        print("✅ ENTRAÎNEMENT TERMINÉ !")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Entraînement interrompu par l'utilisateur")
    
    except Exception as e:
        print(f"\n❌ Erreur entraînement: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Sauvegarder modèle final
        print(f"\n💾 Sauvegarde modèle: {config['output_path']}")
        
        output_path = Path(config['output_path'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        model.save(config['output_path'])
        print("✅ Modèle sauvegardé")
        
        # Sauvegarder config
        import json
        config_path = str(output_path).replace('.zip', '.json')
        with open(config_path, 'w') as f:
            json.dump({
                'tickers': tickers,
                'training_timesteps': config['total_steps'],
                'n_envs': config['n_envs'],
                'max_trades_per_day': config['max_trades_per_day'],
                'observation_size': 115,
                'lookahead': False,
                'version': 'v3_fixed',
                'date': datetime.now().isoformat()
            }, f, indent=2)
        print(f"✅ Config sauvegardée: {config_path}")
        
        # Fermer W&B
        if config['use_wandb']:
            wandb.finish()
            print("✅ W&B fermé")
        
        # Fermer environnements
        env.close()
        print("✅ Environnements fermés")
        
        print("\n" + "="*80)
        print("🎉 TERMINÉ !")
        print("="*80)
        print(f"\nModèle: {config['output_path']}")
        print(f"Checkpoints: {config['output_dir']}/checkpoints/")
        print("\nProchaine étape: Backtest avec scripts/backtest_reliability.py")
        print("="*80)

if __name__ == "__main__":
    import torch
    main()
