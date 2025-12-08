#!/usr/bin/env python3
"""
📈 PLOUTOS TRAINING V3 - TREND FOLLOWING

Script d'entraînement avec environnement V3 OPTIMISÉ

Nouvelles capacités:
- Features de tendance (EMA, ADX, ROC, ATR)
- Anticipation lookahead (5 steps)
- Limite overtrading (50 trades/jour)
- Reward intelligente (bonus/malus tendance)
- Données 2022-2024 (inclut crash pour robustesse)

Objectif:
- Backtest 90j > 15%
- Backtest 365j > 10%
- Score fiabilité > 70/100
- Drawdown < 15%

Auteur: Ploutos AI Team
Date: Dec 2025
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
from stable_baselines3.common.callbacks import CheckpointCallback
import warnings
warnings.filterwarnings('ignore')

from core.universal_environment_v3_trend import UniversalTradingEnvV3Trend

# W&B optional
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb non disponible, tracking désactivé")

print("="*80)
print("📈 PLOUTOS TRAINING V3 - TREND FOLLOWING")
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

def create_callbacks(config):
    """Crée les callbacks pour l'entraînement"""
    callbacks = []
    
    checkpoint_dir = Path(config['output_dir']) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=str(checkpoint_dir),
        name_prefix='ploutos_v3_trend',
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
    
    return callbacks

def main():
    parser = argparse.ArgumentParser(
        description='Entraîner Ploutos V3 avec TREND FOLLOWING',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Entraînement standard 2M steps
  python3 scripts/train_v3_trend.py
  
  # Entraînement long 5M steps
  python3 scripts/train_v3_trend.py --steps 5000000
  
  # Avec W&B tracking
  python3 scripts/train_v3_trend.py --wandb --project Ploutos_V3_Trend
  
  # Tickers custom
  python3 scripts/train_v3_trend.py --tickers NVDA MSFT AAPL SPY QQQ
        """
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        default=2_000_000,
        help='Nombre total de steps (défaut: 2M)'
    )
    
    parser.add_argument(
        '--tickers',
        nargs='+',
        default=None,
        help='Liste de tickers (défaut: 10 tickers)'
    )
    
    parser.add_argument(
        '--output',
        default='models/ploutos_v3_trend.zip',
        help='Chemin sauvegarde modèle final'
    )
    
    parser.add_argument(
        '--output-dir',
        default='models/production_v3',
        help='Dossier pour checkpoints'
    )
    
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Activer tracking W&B'
    )
    
    parser.add_argument(
        '--project',
        default='Ploutos_V3_Trend_Following',
        help='Nom projet W&B'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=1095,  # 3 ans (inclut crash 2022)
        help='Jours de données historiques (défaut: 1095 = 3 ans)'
    )
    
    parser.add_argument(
        '--max-trades-per-day',
        type=int,
        default=50,
        help='Limite trades/jour (défaut: 50, anti-overtrading)'
    )
    
    args = parser.parse_args()
    
    config = {
        'total_steps': args.steps,
        'output_path': args.output,
        'output_dir': args.output_dir,
        'use_wandb': args.wandb and WANDB_AVAILABLE,
        'wandb_project': args.project,
        'max_trades_per_day': args.max_trades_per_day
    }
    
    # ╭────────────────────────────────────────────────────────╮
    # │ PHASE 1: CHARGEMENT DONNÉES (INCLUT CRASH 2022)
    # ╰────────────────────────────────────────────────────────╯
    
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
        print("\n❌ Erreur: Pas assez de données (minimum 3 tickers)")
        return 1
    
    print(f"\n✅ {len(data)} tickers chargés")
    
    min_length = min(len(df) for df in data.values())
    max_steps = min(500, int(min_length * 0.6))
    print(f"📊 Taille données: {min_length} jours")
    print(f"⏱️  Max steps par épisode: {max_steps}")
    
    # ╭────────────────────────────────────────────────────────╮
    # │ PHASE 2: CRÉATION ENVIRONNEMENT V3 TREND
    # ╰────────────────────────────────────────────────────────╯
    
    print("\n🏟️  Création UniversalTradingEnvV3Trend...")
    
    env = UniversalTradingEnvV3Trend(
        data=data,
        initial_balance=100000,
        commission=0.0001,
        max_steps=max_steps,
        buy_pct=0.2,
        max_trades_per_day=config['max_trades_per_day'],
        lookahead_steps=5  # Anticipe 5 steps futurs
    )
    
    vec_env = DummyVecEnv([lambda: env])
    print("✅ Environnement V3 créé")
    print(f"   - Tickers: {env.n_assets}")
    print(f"   - Action space: MultiDiscrete({[3] * env.n_assets})")
    print(f"   - Observation space: {env.observation_space.shape} (103 features)")
    print(f"   - Max trades/jour: {config['max_trades_per_day']}")
    print(f"   - Lookahead: 5 steps (anticipation)")
    
    # ╭────────────────────────────────────────────────────────╮
    # │ PHASE 3: INITIALISATION W&B
    # ╰────────────────────────────────────────────────────────╯
    
    if config['use_wandb']:
        print("\n📊 Initialisation W&B...")
        wandb.init(
            project=config['wandb_project'],
            config={
                'algorithm': 'PPO',
                'env': 'UniversalTradingEnvV3Trend',
                'version': 'v3',
                'total_steps': config['total_steps'],
                'n_assets': env.n_assets,
                'tickers': list(data.keys()),
                'action_space': 'Discrete(3) per ticker',
                'observation_features': 103,
                'features_per_ticker': 10,
                'new_features': ['EMA50/200', 'ADX', 'ROC', 'ATR', 'trend_signal'],
                'reward': 'PnL + lookahead anticipation',
                'max_trades_per_day': config['max_trades_per_day'],
                'lookahead_steps': 5,
                'network': '[256, 256]',
                'batch_size': 64,
                'learning_rate': 3e-4,
                'data_period_days': args.days,
                'includes_crash_2022': args.days >= 1000
            },
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True
        )
        print("✅ W&B initialisé")
    
    # ╭────────────────────────────────────────────────────────╮
    # │ PHASE 4: INITIALISATION MODÈLE PPO
    # ╰────────────────────────────────────────────────────────╯
    
    print("\n🧠 Initialisation PPO...")
    
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
        ent_coef=0.05,
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
    
    # ╭────────────────────────────────────────────────────────╮
    # │ PHASE 5: ENTRAÎNEMENT
    # ╰────────────────────────────────────────────────────────╯
    
    print("\n" + "="*80)
    print(f"🚀 DÉMARRAGE ENTRAÎNEMENT V3 ({config['total_steps']:,} steps)")
    print("="*80)
    print("\n🎯 Objectifs V3:")
    print("   - Moins de trades (<100/jour)")
    print("   - Meilleur timing (anticipe tendance)")
    print("   - Score 365j > 70/100")
    print("   - Return 365j > 10%")
    print("   - Drawdown < 15%")
    print("\n⏳ Durée estimée:")
    print(f"   - GPU RTX 3080 : ~{config['total_steps'] // 20000} minutes")
    print(f"   - CPU : ~{config['total_steps'] // 5000} minutes")
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
        print("✅ ENTRAÎNEMENT V3 TERMINÉ !")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n" + "="*80)
        print("⚠️  ENTRAÎNEMENT INTERROMPU")
        print("="*80)
        print("\n💾 Sauvegarde du modèle actuel...")
    
    # ╭────────────────────────────────────────────────────────╮
    # │ PHASE 6: SAUVEGARDE MODÈLE FINAL
    # ╰────────────────────────────────────────────────────────╯
    
    output_path = Path(config['output_path'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.save(config['output_path'])
    print(f"\n✅ Modèle V3 sauvegardé: {config['output_path']}")
    
    # Métadonnées
    metadata = {
        'version': 'v3_trend',
        'date': datetime.now().isoformat(),
        'tickers': list(data.keys()),
        'n_assets': env.n_assets,
        'total_steps': config['total_steps'],
        'observation_features': 103,
        'features_per_ticker': 10,
        'action_space': 'MultiDiscrete(3) per ticker',
        'max_trades_per_day': config['max_trades_per_day'],
        'lookahead_steps': 5,
        'reward_type': 'PnL + lookahead anticipation + trend bonuses',
        'data_period': f"{start.date()} to {end.date()}",
        'includes_crash': args.days >= 1000,
        'new_features': [
            'EMA 50/200 (tendance long terme)',
            'ADX (force tendance)',
            'Momentum ROC (vitesse)',
            'ATR (volatilité)',
            'Trend signal (+1 bull / -1 bear)',
            'EMA distance (force)',
            'Lookahead reward (anticipation)'
        ],
        'improvements_vs_v2': [
            'Anticipe tendance (ne réagit plus)',
            'Limite overtrading (50/jour)',
            'Bonus BUY avant hausse',
            'Bonus SELL avant baisse',
            'Malus BUY en tendance baissiere',
            'Bonus HOLD en tendance haussiere'
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
    print("🎉 SUCCÈS TOTAL V3 !")
    print("="*80)
    print("\n🚀 Prochaines étapes:")
    print("   1. Backtest 90j: python3 scripts/backtest_reliability.py --days 90 --episodes 10")
    print("   2. Backtest 365j: python3 scripts/backtest_reliability.py --days 365 --episodes 10")
    print("   3. Si score > 70/100 → Déployer paper trading")
    print("   4. Sinon → Ajuster hyperparams et re-entraîner")
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
