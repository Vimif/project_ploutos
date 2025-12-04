import sys
import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.environment import TradingEnv

N_ENVS = 64

def filter_data_by_difficulty(csv_path, difficulty="easy"):
    """Filtre les données selon la difficulté du marché"""
    
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    original_len = len(df)
    
    if difficulty == "easy":
        # FACILE : Périodes haussières stables (tendance claire + faible volatilité)
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Filtrer : rendements positifs + volatilité < 2%
        df = df[(df['returns'] > 0) & (df['volatility'] < 0.02)]
        df = df.drop(['returns', 'volatility'], axis=1)
        
        print(f"   📗 EASY : {len(df)}/{original_len} bougies ({len(df)/original_len*100:.1f}%)")
        
    elif difficulty == "medium":
        # MOYEN : Exclure les variations extrêmes (> 5%)
        df['returns'] = df['Close'].pct_change()
        df = df[abs(df['returns']) < 0.05]
        df = df.drop(['returns'], axis=1)
        
        print(f"   📙 MEDIUM : {len(df)}/{original_len} bougies ({len(df)/original_len*100:.1f}%)")
        
    else:  # "hard"
        # DIFFICILE : Toutes les données brutes
        print(f"   📕 HARD : {len(df)}/{original_len} bougies (100%)")
    
    # Vérifier qu'il reste assez de données
    if len(df) < 500:
        print(f"   ⚠️  WARNING : Seulement {len(df)} bougies, peut être insuffisant")
    
    # Sauvegarder fichier filtré
    filtered_path = csv_path.replace(".csv", f"_{difficulty}.csv")
    df.to_csv(filtered_path)
    
    return filtered_path

def train_curriculum(ticker, base_csv_path):
    """Entraînement progressif en 3 phases"""
    
    print("\n" + "="*70)
    print(f"🎓 CURRICULUM LEARNING : {ticker}")
    print("="*70)
    
    # Initialisation W&B
    run_name = f"{ticker}_CURRICULUM_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"
    wandb.init(
        project="Ploutos_Curriculum",
        name=run_name,
        config={"ticker": ticker, "method": "curriculum"}
    )
    
    # ============================================
    # PHASE 1 : EASY (1M steps)
    # ============================================
    print("\n🎓 PHASE 1/3 : Marché FACILE (tendance claire, faible volatilité)")
    easy_path = filter_data_by_difficulty(base_csv_path, "easy")
    
    def make_env_easy():
        return TradingEnv(csv_path=easy_path)
    
    env = SubprocVecEnv([make_env_easy for _ in range(N_ENVS)])
    
    # Créer le modèle
    policy_kwargs = dict(net_arch=dict(pi=[512, 512, 512], vf=[512, 512, 512]))
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cuda",
        learning_rate=1e-4,
        batch_size=8192,
        n_steps=2048,
        n_epochs=10,
        policy_kwargs=policy_kwargs,
        tensorboard_log=f"tensorboard/{ticker}_curriculum"
    )
    
    print("   🏁 Entraînement Phase 1...")
    model.learn(total_timesteps=1_000_000, progress_bar=True)
    model.save(f"models/{ticker}_phase1_easy.zip")
    
    # ============================================
    # PHASE 2 : MEDIUM (2M steps)
    # ============================================
    print("\n🎓 PHASE 2/3 : Marché MODÉRÉ (volatilité normale)")
    medium_path = filter_data_by_difficulty(base_csv_path, "medium")
    
    def make_env_medium():
        return TradingEnv(csv_path=medium_path)
    
    env.close()
    env = SubprocVecEnv([make_env_medium for _ in range(N_ENVS)])
    model.set_env(env)  # Transférer le modèle pré-entraîné
    
    print("   🏁 Entraînement Phase 2...")
    model.learn(total_timesteps=2_000_000, progress_bar=True, reset_num_timesteps=False)
    model.save(f"models/{ticker}_phase2_medium.zip")
    
    # ============================================
    # PHASE 3 : HARD (5M steps)
    # ============================================
    print("\n🎓 PHASE 3/3 : Marché COMPLET (toutes conditions réelles)")
    
    def make_env_hard():
        return TradingEnv(csv_path=base_csv_path)
    
    env.close()
    env = SubprocVecEnv([make_env_hard for _ in range(N_ENVS)])
    model.set_env(env)
    
    print("   🏁 Entraînement Phase 3 (finale)...")
    model.learn(total_timesteps=5_000_000, progress_bar=True, reset_num_timesteps=False)
    
    # SAUVEGARDE FINALE
    final_path = f"models/{ticker}_curriculum_final.zip"
    model.save(final_path)
    print(f"\n✅ Modèle curriculum complet sauvegardé : {final_path}")
    
    env.close()
    wandb.finish()

if __name__ == "__main__":
    # Entraîner tous les tickers
    tickers = ["NVDA", "TSLA", "AAPL", "AMD", "MSFT", "AMZN", "GOOGL"]
    
    for ticker in tickers:
        csv_path = f"data_cache/{ticker}.csv"
        if os.path.exists(csv_path):
            train_curriculum(ticker, csv_path)
        else:
            print(f"⚠️  SKIP {ticker} : fichier {csv_path} introuvable")
