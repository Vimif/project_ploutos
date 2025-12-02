# ai_trainer.py
# ---------------------------------------------------------
# ENTRAÎNEUR IA "IRON MAN" (W&B + LSTM + VECNORMALIZE)
# ---------------------------------------------------------
import pandas as pd
import numpy as np
import os
import torch
import sys
import shutil

# --- W&B INTEGRATION ---
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    USE_WANDB = True
    print("📊 Monitoring W&B : ACTIVÉ")
except ImportError:
    USE_WANDB = False
    print("⚠️ W&B non installé (pip install wandb). Fallback sur TensorBoard.")

# --- ARCHITECTURES ---
try:
    from sb3_contrib import RecurrentPPO
    USE_LSTM = True
    print("🧠 Architecture : LSTM (Recurrent Neural Network)")
except ImportError:
    from stable_baselines3 import PPO
    USE_LSTM = False
    print("🧠 Architecture : MLP (Standard)")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from trading_env import StockTradingEnv
from trading_bot import TradingBrain

# CONFIGURATION
MODEL_FILE = "ppo_trading_brain"
TRAINING_TICKERS = ["SPY", "NVDA", "JPM", "XOM", "GLD", "TSLA", "AMZN", "GOOGL"]
PROJECT_NAME = "Ploutos_Trading_V30"

# SECTEURS USINE
SECTORS = {
    "TECH": ["NVDA", "AAPL", "MSFT", "AMD", "TSLA", "QQQ", "META"],
    "DEFENSIVE": ["KO", "PG", "JNJ", "MCD", "WMT", "XLV"],
    "ENERGY": ["XOM", "CVX", "SLB", "XLE", "CAT"],
    "CRYPTO": ["COIN", "MSTR", "MARA", "BITO"]
}

# Gestion des Arguments CLI
if len(sys.argv) > 1:
    arg = sys.argv[1].upper()
    if arg in SECTORS:
        print(f"🏭 CIBLE SECTORIELLE : {arg}")
        TRAINING_TICKERS = SECTORS[arg]
        MODEL_FILE = f"brain_{arg.lower()}"

torch.set_num_threads(4)
DATA_CACHE = {}

def precompute_all_data():
    print(f"💾 [{MODEL_FILE}] Chargement RAM...")
    brain = TradingBrain()
    for t in TRAINING_TICKERS:
        df = brain.telecharger_donnees(t)
        if df is None or df.empty: continue
        
        df = df.copy()
        close = df['Close']
        
        # Indicateurs
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 0.001)
        df['RSI'] = 100 - (100 / (1 + rs))
        df['SMA_Ratio'] = close / close.rolling(50).mean()
        e1 = close.ewm(span=12).mean(); e2 = close.ewm(span=26).mean()
        df['MACD'] = e1 - e2
        
        DATA_CACHE[t] = df.dropna().reset_index(drop=True)

def make_env(ticker):
    df = DATA_CACHE.get(ticker, pd.DataFrame())
    if df.empty: return None 
    return StockTradingEnv(df)

def train_model():
    precompute_all_data()
    if not DATA_CACHE: return

    print(f"🚀 Démarrage Entraînement : {MODEL_FILE}")
    
    # Nettoyage fichiers locaux
    if os.path.exists(MODEL_FILE + ".zip"): os.remove(MODEL_FILE + ".zip")
    if os.path.exists(MODEL_FILE + "_vecnorm.pkl"): os.remove(MODEL_FILE + "_vecnorm.pkl")

    EPOCHS = 5
    TOTAL_TIMESTEPS = 150000
    
    model = None
    
    # Config Architecture
    if USE_LSTM:
        policy_kwargs = dict(lstm_hidden_size=256, enable_critic_lstm=True)
        AlgoClass = RecurrentPPO
        policy_type = "MlpLstmPolicy"
        batch_size = 256
    else:
        policy_kwargs = dict(net_arch=dict(pi=[512, 512, 512], vf=[512, 512, 512]))
        AlgoClass = PPO
        policy_type = "MlpPolicy"
        batch_size = 1024

    # --- NORMALISATION ---
    # On crée l'env global normalisé
    dummy_env = DummyVecEnv([lambda: make_env(TRAINING_TICKERS[0])])
    norm_env = VecNormalize(dummy_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    for epoch in range(EPOCHS):
        print(f"\n🔄 {MODEL_FILE} - CYCLE {epoch+1}/{EPOCHS}")
        
        for ticker in TRAINING_TICKERS:
            if ticker not in DATA_CACHE: continue
            
            # Échange dynamique de l'env interne
            norm_env.venv = DummyVecEnv([lambda: make_env(ticker)])
            
            # --- INITIALISATION W&B ---
            if USE_WANDB:
                run = wandb.init(
                    project=PROJECT_NAME,
                    group=MODEL_FILE, # Groupe par "Cerveau" (Tech, Energy...)
                    name=f"{ticker}_Ep{epoch}",
                    config={
                        "policy_type": policy_type,
                        "total_timesteps": TOTAL_TIMESTEPS,
                        "ticker": ticker,
                        "epoch": epoch
                    },
                    reinit=True,
                    sync_tensorboard=True, # Capture aussi les logs SB3 classiques
                    monitor_gym=True
                )
            
            # Création / Update Modèle
            if model is None:
                model = AlgoClass(
                    policy_type, 
                    norm_env, 
                    verbose=1, 
                    device="cpu", 
                    learning_rate=0.00025, 
                    n_steps=4096, 
                    batch_size=batch_size, 
                    gamma=0.95, 
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=f"runs/{run.id}" if USE_WANDB else "./logs_tensorboard/"
                )
            else:
                model.set_env(norm_env)
            
            # Callback W&B
            callbacks = []
            if USE_WANDB:
                callbacks.append(WandbCallback(
                    gradient_save_freq=100,
                    model_save_path=f"models/{run.id}",
                    verbose=2
                ))

            # Apprentissage
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS, 
                callback=callbacks,
                reset_num_timesteps=False
            )
            
            # Fin du run W&B
            if USE_WANDB: run.finish()
            
            # Sauvegarde Checkpoint
            model.save(MODEL_FILE)
            norm_env.save(MODEL_FILE + "_vecnorm.pkl")

    print(f"✅ Entraînement terminé. Modèle sauvegardé.")

if __name__ == "__main__":
    train_model()
