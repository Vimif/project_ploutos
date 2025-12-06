#!/usr/bin/env python3
"""
🚀 PLOUTOS TRAINING V2 - PRODUCTION

Script d'entraînement avec UniversalTradingEnvV2 validé (+148%)

Améliorations:
- Actions discrètes (BUY/HOLD/SELL)
- Reward PnL réalisé + latent
- Tracking entry prices par ticker
- Vente forcée à la fin

Auteur: Session de debug 6 déc 2025
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
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import warnings
warnings.filterwarnings('ignore')

from core.universal_environment_v2 import UniversalTradingEnvV2

# W&B optional
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb non disponible, tracking désactivé")

print("="*80)
print("🚀 PLOUTOS TRAINING V2 - PRODUCTION")
print("="*80)

def load_data(tickers, start_date, end_date):
    """Charge les données pour plusieurs tickers"""
    data = {}
    
    print(f"\n📡 Chargement données pour {len(tickers)} tickers...")
    
    for ticker in tickers:
        print(f"   {ticker}...", end=" ")
        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
            
            if df.empty or len(df) < 100:
                print("❌ Erreur")
                continue
            
            data[ticker] = df
            print(f"✅ {len(df)} jours")
        except Exception as e:
            print(f"❌ Erreur: {e}")
    
    return data

def get_default_tickers():
    """Tickers par défaut (mix growth/defensive/sectoriels)"""
    return [
        # Tech Growth
        'NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN',
        # Indices
        'SPY', 'QQQ', 'VOO',
        # Sectoriels
        'XLE',  # Energy
        'XLF'   # Finance
    ]

def create_callbacks(config):
    """Crée les callbacks pour l'entraînement"""
    callbacks = []
    
    # ✅ Checkpoint tous les 100k steps
    checkpoint_dir = Path(config['output_dir']) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=str(checkpoint_dir),
        name_prefix='ploutos_v2',
        save_replay_buffer=False,
        save_vecnormalize=False
    )
    callbacks.append(checkpoint_callback)
    
    # ✅ W&B si disponible
    if WANDB_AVAILABLE and config.get('use_wandb', False):
        wandb_callback = WandbCallback(
            model_save_path=str(checkpoint_dir),
            verbose=2
        )
        callbacks.append(wandb_callback)
    
    return callbacks

def main():
    parser = argparse.ArgumentParser(
        description='Entraîner Ploutos V2 en production',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Entraînement standard 1M steps
  python3 scripts/train_v2_production.py
  
  # Entraînement long 2M steps
  python3 scripts/train_v2_production.py --steps 2000000
  
  # Tickers custom
  python3 scripts/train_v2_production.py --tickers NVDA MSFT AAPL SPY QQQ
  
  # Avec W&B tracking
  python3 scripts/train_v2_production.py --wandb --project Ploutos_V2_Production
        """
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        default=1_000_000,
        help='Nombre total de steps (défaut: 1M)'
    )
    
    parser.add_argument(
        '--tickers',
        nargs='+',
        default=None,
        help='Liste de tickers (défaut: 10 tickers mixés)'
    )
    
    parser.add_argument(
        '--output',
        default='models/ploutos_v2_production.zip',
        help='Chemin sauvegarde modèle final'
    )
    
    parser.add_argument(
        '--output-dir',
        default='models/production_v2',
        help='Dossier pour checkpoints'
    )
    
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Activer tracking W&B'
    )
    
    parser.add_argument(
        '--project',
        default='Ploutos_Trading_V2_Production',
        help='Nom projet W&B'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=730,
        help='Jours de données historiques (défaut: 730 = 2 ans)'
    )
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'total_steps': args.steps,
        'output_path': args.output,
        'output_dir': args.output_dir,
        'use_wandb': args.wandb and WANDB_AVAILABLE,
        'wandb_project': args.project
    }
    
    # ╔════════════════════════════════════════════════════════╗
    # ║ PHASE 1: CHARGEMENT DONNÉES
    # ╚════════════════════════════════════════════════════════╝
    
    tickers = args.tickers if args.tickers else get_default_tickers()
    
    end = datetime.now()
    start = end - timedelta(days=args.days)
    
    data = load_data(tickers, start, end)
    
    if len(data) < 3:
        print("\n❌ Erreur: Pas assez de données chargées (minimum 3 tickers)")
        return 1
    
    print(f"\n✅ {len(data)} tickers chargés avec succès")
    
    # Adapter max_steps aux données
    min_length = min(len(df) for df in data.values())
    max_steps = min(400, int(min_length * 0.6))
    print(f"📊 Taille données: {min_length} jours")
    print(f"⏱️  Max steps par épisode: {max_steps}")
    
    # ╔════════════════════════════════════════════════════════╗
    # ║ PHASE 2: CRÉATION ENVIRONNEMENT V2
    # ╚════════════════════════════════════════════════════════╝
    
    print("\n🏗️  Création UniversalTradingEnvV2...")
    
    env = UniversalTradingEnvV2(
        data=data,
        initial_balance=100000,
        commission=0.0001,
        max_steps=max_steps,
        buy_pct=0.2,
        realistic_costs=False
    )
    
    vec_env = DummyVecEnv([lambda: env])
    print("✅ Environnement créé")
    print(f"   - Tickers: {env.n_assets}")
    print(f"   - Action space: MultiDiscrete({[3] * env.n_assets})")
    print(f"   - Observation space: {env.observation_space.shape}")
    
    # ╔════════════════════════════════════════════════════════╗
    # ║ PHASE 3: INITIALISATION W&B (optionnel)
    # ╚════════════════════════════════════════════════════════╝
    
    if config['use_wandb']:
        print("\n📊 Initialisation W&B...")
        wandb.init(
            project=config['wandb_project'],
            config={
                'algorithm': 'PPO',
                'env': 'UniversalTradingEnvV2',
                'total_steps': config['total_steps'],
                'n_assets': env.n_assets,
                'tickers': list(data.keys()),
                'action_space': 'Discrete(3) per ticker',
                'reward': 'PnL réalisé + latent',
                'network': '[256, 256]',
                'batch_size': 64,
                'learning_rate': 3e-4
            },
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True
        )
        print("✅ W&B initialisé")
    
    # ╔════════════════════════════════════════════════════════╗
    # ║ PHASE 4: INITIALISATION MODÈLE PPO
    # ╚════════════════════════════════════════════════════════╝
    
    print("\n🧠 Initialisation PPO...")
    
    # ✅ Hyperparamètres validés (+148%)
    model = PPO(
        'MlpPolicy',
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,  # Exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={'net_arch': [256, 256]},
        verbose=1,
        tensorboard_log=f"runs/{config['wandb_project']}" if config['use_wandb'] else None
    )
    
    print("✅ Modèle initialisé")
    print(f"   - Learning rate: 3e-4")
    print(f"   - Batch size: 64")
    print(f"   - Network: [256, 256]")
    print(f"   - Total steps: {config['total_steps']:,}")
    
    # ╔════════════════════════════════════════════════════════╗
    # ║ PHASE 5: ENTRAÎNEMENT
    # ╚════════════════════════════════════════════════════════╝
    
    print("\n" + "="*80)
    print(f"🚀 DÉMARRAGE ENTRAÎNEMENT ({config['total_steps']:,} steps)")
    print("="*80)
    print("\n⏳ Durée estimée:")
    print(f"   - CPU : ~{config['total_steps'] // 5000} minutes")
    print(f"   - GPU : ~{config['total_steps'] // 20000} minutes")
    print("\n💾 Checkpoints auto tous les 100k steps")
    print(f"   Dossier: {config['output_dir']}/checkpoints/\n")
    
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
        print("\n" + "="*80)
        print("⚠️  ENTRAÎNEMENT INTERROMPU")
        print("="*80)
        print("\n💾 Sauvegarde du modèle actuel...")
    
    # ╔════════════════════════════════════════════════════════╗
    # ║ PHASE 6: SAUVEGARDE MODÈLE FINAL
    # ╚════════════════════════════════════════════════════════╝
    
    output_path = Path(config['output_path'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.save(config['output_path'])
    print(f"\n✅ Modèle sauvegardé: {config['output_path']}")
    
    # Métadonnées
    metadata = {
        'version': 'v2',
        'date': datetime.now().isoformat(),
        'tickers': list(data.keys()),
        'n_assets': env.n_assets,
        'total_steps': config['total_steps'],
        'action_space': 'MultiDiscrete(3) per ticker',
        'reward_type': 'PnL réalisé + 0.5% PnL latent',
        'fixes': [
            'Actions discrètes (BUY/HOLD/SELL)',
            'Reward PnL réalisé',
            'Tracking entry prices',
            'Vente forcée fin'
        ]
    }
    
    import json
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Métadonnées: {metadata_path}")
    
    if config['use_wandb']:
        wandb.finish()
        print("✅ W&B fermé")
    
    print("\n" + "="*80)
    print("🎉 SUCCÈS TOTAL !")
    print("="*80)
    print("\n🚀 Prochaines étapes:")
    print("   1. Tester le modèle: python3 scripts/test_model_v2.py")
    print("   2. Backtest: python3 scripts/backtest_v2.py")
    print("   3. Déployer: Copier sur VPS et relancer le service")
    print("\n")
    
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
