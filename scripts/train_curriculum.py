#!/usr/bin/env python3
"""
🎓 CURRICULUM LEARNING POUR PLOUTOS
Entraînement progressif : Simple → Complexe
"""

import sys
sys.path.insert(0, '.')

import os
import wandb
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from core.data_fetcher import UniversalDataFetcher
from core.universal_environment import UniversalTradingEnv

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'stage1': {
        'name': 'Mono-Asset (SPY)',
        'tickers': ['SPY'],
        'timesteps': 3_000_000,
        'n_envs': 4,
        'learning_rate': 1e-4,
        'n_steps': 2048,
        'batch_size': 256,
        'n_epochs': 10,
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
        'n_epochs': 10,
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
        'n_epochs': 10,
        'target_sharpe': 1.5
    }
}

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def print_banner(text):
    """Affiche une bannière"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def make_env(ticker, df):
    """Crée un environnement de trading"""
    def _init():
        return UniversalTradingEnv(
            df=df,
            ticker=ticker,
            initial_balance=10000,
            transaction_fee=0.001
        )
    return _init

def calculate_sharpe(model, env, episodes=10):
    """Calcule le Sharpe Ratio du modèle"""
    import numpy as np
    
    returns = []
    
    for _ in range(episodes):
        obs = env.reset()
        done = False
        episode_return = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_return += reward
        
        returns.append(episode_return)
    
    returns = np.array(returns)
    
    if returns.std() == 0:
        return 0
    
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    return sharpe

# ============================================================================
# STAGE 1 : MONO-ASSET (SPY)
# ============================================================================

def train_stage1():
    """Étape 1 : Apprendre sur SPY uniquement"""
    
    print_banner("🎓 STAGE 1 : " + CONFIG['stage1']['name'])
    
    # Initialiser W&B
    wandb.init(
        project="Ploutos_Curriculum",
        name=f"Stage1_SPY_{datetime.now().strftime('%Y%m%d_%H%M')}",
        config=CONFIG['stage1']
    )
    
    # Récupérer données
    print("📥 Téléchargement des données...")
    fetcher = UniversalDataFetcher()
    data = fetcher.bulk_fetch(CONFIG['stage1']['tickers'], interval='1h')
    
    if 'SPY' not in data:
        raise ValueError("❌ Impossible de récupérer SPY")
    
    print(f"✅ Données récupérées : {len(data['SPY'])} bougies")
    
    # Créer environnement
    print("\n🏗️  Création de l'environnement...")
    env = SubprocVecEnv([
        make_env('SPY', data['SPY']) 
        for _ in range(CONFIG['stage1']['n_envs'])
    ])
    
    # Créer modèle
    print("\n🧠 Création du modèle PPO...")
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=CONFIG['stage1']['learning_rate'],
        n_steps=CONFIG['stage1']['n_steps'],
        batch_size=CONFIG['stage1']['batch_size'],
        n_epochs=CONFIG['stage1']['n_epochs'],
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        policy_kwargs=dict(net_arch=[512, 512, 512]),
        verbose=1,
        tensorboard_log='./logs/stage1',
        device='cuda'
    )
    
    # Callbacks
    os.makedirs('models/stage1', exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path='./models/stage1',
        name_prefix='ploutos_stage1'
    )
    
    # Entraînement
    print(f"\n🚀 Entraînement : {CONFIG['stage1']['timesteps']:,} timesteps...")
    print(f"⏱️  Durée estimée : ~2-3 heures sur RTX 3080\n")
    
    model.learn(
        total_timesteps=CONFIG['stage1']['timesteps'],
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Sauvegarder
    model_path = 'models/stage1_spy_final'
    model.save(model_path)
    print(f"\n✅ Modèle sauvegardé : {model_path}.zip")
    
    # Évaluation
    print("\n📊 Évaluation du modèle...")
    test_env = UniversalTradingEnv(
        df=data['SPY'].iloc[-1000:],  # Dernières 1000 bougies
        ticker='SPY',
        initial_balance=10000
    )
    
    sharpe = calculate_sharpe(model, test_env, episodes=10)
    print(f"\n📈 Sharpe Ratio : {sharpe:.2f}")
    print(f"🎯 Objectif      : {CONFIG['stage1']['target_sharpe']:.2f}")
    
    if sharpe >= CONFIG['stage1']['target_sharpe']:
        print("\n✅ STAGE 1 RÉUSSI ! Passage au Stage 2.")
        success = True
    else:
        print("\n⚠️  Sharpe insuffisant, mais on continue...")
        success = False
    
    wandb.log({
        'stage': 1,
        'sharpe_ratio': sharpe,
        'target_sharpe': CONFIG['stage1']['target_sharpe'],
        'success': success
    })
    
    wandb.finish()
    env.close()
    
    return model_path, sharpe

# ============================================================================
# STAGE 2 : MULTI-ASSET (ETFs)
# ============================================================================

def train_stage2(prev_model_path=None):
    """Étape 2 : Généraliser sur plusieurs ETFs"""
    
    print_banner("🎓 STAGE 2 : " + CONFIG['stage2']['name'])
    
    # Initialiser W&B
    wandb.init(
        project="Ploutos_Curriculum",
        name=f"Stage2_ETFs_{datetime.now().strftime('%Y%m%d_%H%M')}",
        config=CONFIG['stage2']
    )
    
    # Récupérer données
    print("📥 Téléchargement des données...")
    fetcher = UniversalDataFetcher()
    data = fetcher.bulk_fetch(CONFIG['stage2']['tickers'], interval='1h')
    
    if len(data) < len(CONFIG['stage2']['tickers']):
        print(f"⚠️  Seulement {len(data)}/{len(CONFIG['stage2']['tickers'])} tickers récupérés")
    
    # Créer environnements
    print("\n🏗️  Création des environnements...")
    envs = []
    for ticker in data.keys():
        for _ in range(CONFIG['stage2']['n_envs'] // len(data)):
            envs.append(make_env(ticker, data[ticker]))
    
    env = SubprocVecEnv(envs)
    
    # Charger modèle précédent ou créer nouveau
    if prev_model_path and os.path.exists(f"{prev_model_path}.zip"):
        print(f"\n🔄 Transfer Learning depuis : {prev_model_path}")
        model = PPO.load(prev_model_path, env=env, device='cuda')
        
        # Ajuster learning rate (plus fin)
        model.learning_rate = CONFIG['stage2']['learning_rate']
    else:
        print("\n🧠 Création d'un nouveau modèle...")
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=CONFIG['stage2']['learning_rate'],
            n_steps=CONFIG['stage2']['n_steps'],
            batch_size=CONFIG['stage2']['batch_size'],
            n_epochs=CONFIG['stage2']['n_epochs'],
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            vf_coef=0.5,
            policy_kwargs=dict(net_arch=[512, 512, 512]),
            verbose=1,
            tensorboard_log='./logs/stage2',
            device='cuda'
        )
    
    # Callbacks
    os.makedirs('models/stage2', exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path='./models/stage2',
        name_prefix='ploutos_stage2'
    )
    
    # Entraînement
    print(f"\n🚀 Entraînement : {CONFIG['stage2']['timesteps']:,} timesteps...")
    print(f"⏱️  Durée estimée : ~4-5 heures sur RTX 3080\n")
    
    model.learn(
        total_timesteps=CONFIG['stage2']['timesteps'],
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Sauvegarder
    model_path = 'models/stage2_etfs_final'
    model.save(model_path)
    print(f"\n✅ Modèle sauvegardé : {model_path}.zip")
    
    # Évaluation sur chaque ticker
    print("\n📊 Évaluation du modèle...")
    sharpes = {}
    
    for ticker in data.keys():
        test_env = UniversalTradingEnv(
            df=data[ticker].iloc[-1000:],
            ticker=ticker,
            initial_balance=10000
        )
        sharpe = calculate_sharpe(model, test_env, episodes=5)
        sharpes[ticker] = sharpe
        print(f"  {ticker} : Sharpe = {sharpe:.2f}")
    
    avg_sharpe = sum(sharpes.values()) / len(sharpes)
    print(f"\n📈 Sharpe Moyen : {avg_sharpe:.2f}")
    print(f"🎯 Objectif     : {CONFIG['stage2']['target_sharpe']:.2f}")
    
    success = avg_sharpe >= CONFIG['stage2']['target_sharpe']
    
    if success:
        print("\n✅ STAGE 2 RÉUSSI ! Passage au Stage 3.")
    else:
        print("\n⚠️  Sharpe insuffisant, mais on continue...")
    
    wandb.log({
        'stage': 2,
        'sharpe_ratio': avg_sharpe,
        'target_sharpe': CONFIG['stage2']['target_sharpe'],
        'success': success,
        **{f'sharpe_{t}': s for t, s in sharpes.items()}
    })
    
    wandb.finish()
    env.close()
    
    return model_path, avg_sharpe

# ============================================================================
# STAGE 3 : ACTIONS COMPLEXES
# ============================================================================

def train_stage3(prev_model_path=None):
    """Étape 3 : Maîtriser des actions individuelles"""
    
    print_banner("🎓 STAGE 3 : " + CONFIG['stage3']['name'])
    
    # Initialiser W&B
    wandb.init(
        project="Ploutos_Curriculum",
        name=f"Stage3_Stocks_{datetime.now().strftime('%Y%m%d_%H%M')}",
        config=CONFIG['stage3']
    )
    
    # Récupérer données
    print("📥 Téléchargement des données...")
    fetcher = UniversalDataFetcher()
    data = fetcher.bulk_fetch(CONFIG['stage3']['tickers'], interval='1h')
    
    print(f"✅ {len(data)}/{len(CONFIG['stage3']['tickers'])} tickers récupérés")
    
    # Créer environnements
    print("\n🏗️  Création des environnements...")
    envs = []
    for ticker in data.keys():
        for _ in range(CONFIG['stage3']['n_envs'] // len(data)):
            envs.append(make_env(ticker, data[ticker]))
    
    env = SubprocVecEnv(envs)
    
    # Transfer Learning
    if prev_model_path and os.path.exists(f"{prev_model_path}.zip"):
        print(f"\n🔄 Transfer Learning depuis : {prev_model_path}")
        model = PPO.load(prev_model_path, env=env, device='cuda')
        model.learning_rate = CONFIG['stage3']['learning_rate']
    else:
        print("\n🧠 Création d'un nouveau modèle...")
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=CONFIG['stage3']['learning_rate'],
            n_steps=CONFIG['stage3']['n_steps'],
            batch_size=CONFIG['stage3']['batch_size'],
            n_epochs=CONFIG['stage3']['n_epochs'],
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.001,
            vf_coef=0.5,
            policy_kwargs=dict(net_arch=[512, 512, 512]),
            verbose=1,
            tensorboard_log='./logs/stage3',
            device='cuda'
        )
    
    # Callbacks
    os.makedirs('models/stage3', exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=200000,
        save_path='./models/stage3',
        name_prefix='ploutos_stage3'
    )
    
    # Entraînement
    print(f"\n🚀 Entraînement : {CONFIG['stage3']['timesteps']:,} timesteps...")
    print(f"⏱️  Durée estimée : ~8-10 heures sur RTX 3080\n")
    
    model.learn(
        total_timesteps=CONFIG['stage3']['timesteps'],
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Sauvegarder
    model_path = 'models/stage3_stocks_final'
    model.save(model_path)
    print(f"\n✅ Modèle sauvegardé : {model_path}.zip")
    
    # Évaluation finale
    print("\n📊 ÉVALUATION FINALE")
    print("="*80)
    
    sharpes = {}
    for ticker in data.keys():
        test_env = UniversalTradingEnv(
            df=data[ticker].iloc[-1000:],
            ticker=ticker,
            initial_balance=10000
        )
        sharpe = calculate_sharpe(model, test_env, episodes=10)
        sharpes[ticker] = sharpe
        print(f"  {ticker:6s} : Sharpe = {sharpe:5.2f}")
    
    avg_sharpe = sum(sharpes.values()) / len(sharpes)
    print(f"\n📈 Sharpe Moyen : {avg_sharpe:.2f}")
    print(f"🎯 Objectif     : {CONFIG['stage3']['target_sharpe']:.2f}")
    
    success = avg_sharpe >= CONFIG['stage3']['target_sharpe']
    
    if success:
        print("\n🎉 CURRICULUM LEARNING TERMINÉ AVEC SUCCÈS !")
        print("✅ Modèle prêt pour déploiement en production")
    else:
        print("\n⚠️  Objectif non atteint, réentraînement recommandé")
    
    wandb.log({
        'stage': 3,
        'sharpe_ratio': avg_sharpe,
        'target_sharpe': CONFIG['stage3']['target_sharpe'],
        'success': success,
        **{f'sharpe_{t}': s for t, s in sharpes.items()}
    })
    
    wandb.finish()
    env.close()
    
    return model_path, avg_sharpe

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Curriculum Learning pour Ploutos')
    parser.add_argument('--stage', type=int, default=0, 
                        help='Stage à exécuter (0=tous, 1-3=stage spécifique)')
    parser.add_argument('--skip-stage1', action='store_true',
                        help='Sauter le stage 1 (utiliser modèle existant)')
    parser.add_argument('--skip-stage2', action='store_true',
                        help='Sauter le stage 2 (utiliser modèle existant)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🎓 PLOUTOS CURRICULUM LEARNING")
    print("="*80)
    print(f"\n⏰ Début : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Exécution
    if args.stage == 0:
        # Exécuter tous les stages
        
        if not args.skip_stage1:
            model1, sharpe1 = train_stage1()
        else:
            model1 = 'models/stage1_spy_final'
            print(f"\n⏩ Stage 1 sauté, utilisation de : {model1}")
        
        if not args.skip_stage2:
            model2, sharpe2 = train_stage2(prev_model_path=model1)
        else:
            model2 = 'models/stage2_etfs_final'
            print(f"\n⏩ Stage 2 sauté, utilisation de : {model2}")
        
        model3, sharpe3 = train_stage3(prev_model_path=model2)
        
        print("\n" + "="*80)
        print("✅ CURRICULUM LEARNING TERMINÉ")
        print("="*80)
        print(f"\n📊 Résultats finaux :")
        if not args.skip_stage1:
            print(f"   Stage 1 (SPY)    : Sharpe = {sharpe1:.2f}")
        if not args.skip_stage2:
            print(f"   Stage 2 (ETFs)   : Sharpe = {sharpe2:.2f}")
        print(f"   Stage 3 (Stocks) : Sharpe = {sharpe3:.2f}")
        print(f"\n🎯 Modèle final : {model3}.zip")
        
    elif args.stage == 1:
        train_stage1()
        
    elif args.stage == 2:
        prev_model = 'models/stage1_spy_final' if not args.skip_stage1 else None
        train_stage2(prev_model_path=prev_model)
        
    elif args.stage == 3:
        prev_model = 'models/stage2_etfs_final' if not args.skip_stage2 else None
        train_stage3(prev_model_path=prev_model)
    
    print(f"\n⏰ Fin : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
