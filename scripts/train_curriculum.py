#!/usr/bin/env python3
"""
🎓 CURRICULUM LEARNING POUR PLOUTOS
Entraînement progressif : Simple → Complexe

Avec auto-optimisation rapide optionnelle (15 trials)

Usage:
    python3 scripts/train_curriculum.py --stage 1                     # Sans optimisation
    python3 scripts/train_curriculum.py --stage 1 --auto-optimize     # Avec optimisation rapide
    python3 scripts/train_curriculum.py --stage 2 --load-model models/stage1_spy_final
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
from stable_baselines3.common.callbacks import CheckpointCallback
from core.data_fetcher import UniversalDataFetcher
from core.universal_environment import UniversalTradingEnv

# ============================================================================
# PARAMS PRÉ-CALIBRÉS (issus d'expériences précédentes)
# ============================================================================

CALIBRATED_PARAMS = {
    'stage1': {
        'name': 'Mono-Asset (SPY)',
        'tickers': ['SPY'],
        'timesteps': 3_000_000,
        'n_envs': 4,
        'learning_rate': 1e-4,
        'n_steps': 2048,
        'batch_size': 256,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
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
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.005,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
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
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.001,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'policy_kwargs': {'net_arch': [512, 512, 512]},
        'target_sharpe': 1.5
    }
}

# ============================================================================
# AUTO-OPTIMISATION RAPIDE (15 trials)
# ============================================================================

def quick_optimize(tickers, base_params, stage_name, n_trials=15):
    """
    Auto-optimisation RAPIDE (15 trials au lieu de 50)
    Optimise seulement les 3 params les plus critiques
    
    Args:
        tickers: Liste de tickers pour optimisation
        base_params: Params de départ
        stage_name: Nom du stage (pour logs)
        n_trials: Nombre d'essais (15 par défaut)
        
    Returns:
        dict: Params optimisés
    """
    
    print("\n" + "="*80)
    print(f"⚡ AUTO-OPTIMISATION RAPIDE - {stage_name}")
    print("="*80)
    print(f"  Trials       : {n_trials}")
    print(f"  Durée estimée : ~30-40 minutes")
    print(f"  Assets test  : {', '.join(tickers[:2])}")
    print(f"  Params testés : learning_rate, n_steps, ent_coef\n")
    
    import optuna
    from optuna.pruners import MedianPruner
    
    # Charger données
    data = {}
    for ticker in tickers[:2]:  # Max 2 tickers pour rapidité
        cache_file = f'data_cache/{ticker}.csv'
        if os.path.exists(cache_file):
            data[ticker] = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    
    if len(data) == 0:
        print("⚠️  Aucune donnée disponible, skip optimisation")
        return base_params
    
    def objective(trial):
        """Objective Optuna - teste seulement 3 params critiques"""
        
        # Partir des params de base
        params = base_params.copy()
        
        # Optimiser SEULEMENT les 3 plus importants
        params['learning_rate'] = trial.suggest_float(
            'learning_rate',
            base_params['learning_rate'] * 0.5,
            base_params['learning_rate'] * 2.0,
            log=True
        )
        
        params['n_steps'] = trial.suggest_categorical(
            'n_steps',
            [base_params['n_steps'] // 2, base_params['n_steps'], base_params['n_steps'] * 2]
        )
        
        params['ent_coef'] = trial.suggest_float(
            'ent_coef',
            base_params['ent_coef'] * 0.1,
            base_params['ent_coef'] * 10.0,
            log=True
        )
        
        # Évaluer sur 2 tickers
        sharpes = []
        
        for ticker, df in data.items():
            try:
                # Créer env simple
                env = UniversalTradingEnv(
                    data={ticker: df},
                    initial_balance=10000,
                    commission=0.001,
                    max_steps=500
                )
                
                # Entraînement court (100k timesteps)
                policy_kwargs = params.pop('policy_kwargs', {'net_arch': [256, 256]})
                
                model = PPO(
                    'MlpPolicy',
                    env,
                    verbose=0,
                    device='cuda',
                    policy_kwargs=policy_kwargs,
                    **{k: v for k, v in params.items() if k != 'target_sharpe'}
                )
                
                model.learn(total_timesteps=100_000)
                
                # Backtest rapide
                obs, _ = env.reset()
                values = []
                done = False
                
                for _ in range(300):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    values.append(info['portfolio_value'])
                    done = terminated or truncated
                    if done:
                        break
                
                # Calculer Sharpe
                if len(values) > 10:
                    df_val = pd.DataFrame({'value': values})
                    df_val['ret'] = df_val['value'].pct_change().fillna(0)
                    
                    mean_ret = df_val['ret'].mean()
                    std_ret = df_val['ret'].std()
                    
                    if std_ret > 0:
                        sharpe = (mean_ret / std_ret) * np.sqrt(252)
                        sharpes.append(sharpe)
                
                # Restaurer policy_kwargs
                params['policy_kwargs'] = policy_kwargs
                
            except Exception as e:
                print(f"    ⚠️ Trial {trial.number} échec sur {ticker}: {str(e)[:50]}")
                continue
        
        if len(sharpes) == 0:
            return float('-inf')
        
        mean_sharpe = float(np.mean(sharpes))
        
        # Pruning
        trial.report(mean_sharpe, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return mean_sharpe
    
    # Créer étude
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=MedianPruner()
    )
    
    # Optimiser
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Merger meilleurs params avec base
    optimized_params = base_params.copy()
    optimized_params.update(study.best_params)
    
    print(f"\n✅ OPTIMISATION TERMINÉE")
    print(f"  Sharpe amélioré      : {study.best_value:.3f}")
    print(f"  Learning rate     : {base_params['learning_rate']:.6f} → {optimized_params['learning_rate']:.6f}")
    print(f"  N steps           : {base_params['n_steps']} → {optimized_params['n_steps']}")
    print(f"  Ent coef          : {base_params['ent_coef']:.6f} → {optimized_params['ent_coef']:.6f}\n")
    
    # Sauvegarder
    os.makedirs(f'models/stage{stage_name[-1]}', exist_ok=True)
    with open(f'models/stage{stage_name[-1]}/optimized_params.json', 'w') as f:
        json.dump(optimized_params, f, indent=2)
    
    return optimized_params

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def print_banner(text):
    """Affiche une bannière"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def make_env(data_dict):
    """Crée un environnement de trading"""
    def _init():
        return UniversalTradingEnv(
            data=data_dict,
            initial_balance=10000,
            commission=0.001,
            max_steps=1000
        )
    return _init

def calculate_sharpe(model, data_dict, episodes=10):
    """Calcule le Sharpe Ratio du modèle"""
    returns = []
    
    for _ in range(episodes):
        env = UniversalTradingEnv(
            data=data_dict,
            initial_balance=10000,
            commission=0.001,
            max_steps=1000
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
    
    if returns.std() == 0:
        return 0
    
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    return sharpe

# ============================================================================
# TRAIN STAGE GÉNÉRIQUE
# ============================================================================

def train_stage(stage_num, prev_model_path=None, auto_optimize=False):
    """
    Entraîne un stage du curriculum
    
    Args:
        stage_num: Numéro du stage (1, 2, ou 3)
        prev_model_path: Chemin vers modèle précédent (transfer learning)
        auto_optimize: Si True, lance optimisation rapide avant entraînement
    """
    
    stage_key = f'stage{stage_num}'
    config = CALIBRATED_PARAMS[stage_key].copy()
    
    print_banner(f"🎓 STAGE {stage_num} : {config['name']}")
    
    # Récupérer données
    print("📥 Téléchargement des données...")
    fetcher = UniversalDataFetcher()
    data = fetcher.bulk_fetch(config['tickers'], interval='1h')
    
    print(f"✅ {len(data)}/{len(config['tickers'])} tickers récupérés")
    
    # Auto-optimisation optionnelle
    if auto_optimize:
        config = quick_optimize(
            tickers=config['tickers'],
            base_params=config,
            stage_name=f"Stage {stage_num}",
            n_trials=15
        )
    else:
        print("\n📋 Utilisation params pré-calibrés (pas d'optimisation)\n")
    
    # Initialiser W&B
    wandb.init(
        project="Ploutos_Curriculum",
        name=f"Stage{stage_num}_{config['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}",
        config=config
    )
    
    # Créer environnements
    print("🏭 Création des environnements...")
    env = SubprocVecEnv([make_env(data) for _ in range(config['n_envs'])])
    
    # Charger modèle précédent ou créer nouveau
    if prev_model_path and os.path.exists(f"{prev_model_path}.zip"):
        print(f"\n🔄 Transfer Learning depuis : {prev_model_path}")
        model = PPO.load(prev_model_path, env=env, device='cuda')
        model.learning_rate = config['learning_rate']
    else:
        print("\n🧠 Création d'un nouveau modèle...")
        
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
        
        # Restaurer pour la suite
        config['policy_kwargs'] = policy_kwargs
        config['target_sharpe'] = target_sharpe
        config['timesteps'] = timesteps
        config['n_envs'] = n_envs
        config['name'] = name
        config['tickers'] = tickers
    
    # Callbacks
    os.makedirs(f'models/{stage_key}', exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 * stage_num,
        save_path=f'./models/{stage_key}',
        name_prefix=f'ploutos_{stage_key}'
    )
    
    # Entraînement
    print(f"\n🚀 Entraînement : {config['timesteps']:,} timesteps...")
    print(f"⏱️  Durée estimée : ~{config['timesteps'] // 500_000} heures sur RTX 3080\n")
    
    model.learn(
        total_timesteps=config['timesteps'],
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Sauvegarder
    model_path = f'models/{stage_key}_final'
    model.save(model_path)
    print(f"\n✅ Modèle sauvegardé : {model_path}.zip")
    
    # Évaluation
    print("\n📊 Évaluation du modèle...")
    test_data = {ticker: df.iloc[-1000:] for ticker, df in data.items()}
    
    sharpe = calculate_sharpe(model, test_data, episodes=10)
    print(f"\n📈 Sharpe Ratio : {sharpe:.2f}")
    print(f"🎯 Objectif      : {config['target_sharpe']:.2f}")
    
    success = sharpe >= config['target_sharpe']
    
    if success:
        print(f"\n✅ STAGE {stage_num} RÉUSSI !")
    else:
        print(f"\n⚠️  Sharpe insuffisant, mais on continue...")
    
    wandb.log({
        'stage': stage_num,
        'sharpe_ratio': sharpe,
        'target_sharpe': config['target_sharpe'],
        'success': success
    })
    
    wandb.finish()
    env.close()
    
    return model_path, sharpe

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Curriculum Learning pour Ploutos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python3 scripts/train_curriculum.py --stage 1
  python3 scripts/train_curriculum.py --stage 1 --auto-optimize
  python3 scripts/train_curriculum.py --stage 2 --load-model models/stage1_final
  python3 scripts/train_curriculum.py --stage 3 --load-model models/stage2_final --auto-optimize
        """
    )
    
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3],
                        help='Stage à exécuter (1, 2, ou 3)')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Chemin vers modèle précédent pour transfer learning')
    parser.add_argument('--auto-optimize', action='store_true',
                        help='Active auto-optimisation rapide (15 trials, +30min)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🎓 PLOUTOS CURRICULUM LEARNING")
    print("="*80)
    print(f"\n⏰ Début : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Stage : {args.stage}")
    print(f"⚡ Auto-optimize : {'OUI (+30min)' if args.auto_optimize else 'NON'}\n")
    
    # Déterminer modèle précédent par défaut
    if args.load_model is None and args.stage > 1:
        args.load_model = f'models/stage{args.stage - 1}_final'
        print(f"🔗 Transfer learning automatique depuis : {args.load_model}\n")
    
    # Exécution
    model_path, sharpe = train_stage(
        stage_num=args.stage,
        prev_model_path=args.load_model,
        auto_optimize=args.auto_optimize
    )
    
    print(f"\n⏰ Fin : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"✅ Modèle final : {model_path}.zip")
    print(f"📊 Sharpe Ratio : {sharpe:.2f}")
    
    # Suggérer prochaine étape
    if args.stage < 3 and sharpe > 0:
        print(f"\n💡 PROCHAINE ÉTAPE : python3 scripts/train_curriculum.py --stage {args.stage + 1}")
        if args.auto_optimize:
            print("               Ou avec optimisation : --auto-optimize")
