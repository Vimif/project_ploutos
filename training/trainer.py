import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from core.environment import TradingEnv
from config.settings import TRAINING_CONFIG, WANDB_CONFIG, USE_GPU
import wandb
from wandb.integration.sb3 import WandbCallback

def download_and_cache_data(ticker, data_dir="data_cache"):
    """Télécharge et met en cache les données une seule fois"""
    os.makedirs(data_dir, exist_ok=True)
    file_path = f"{data_dir}/{ticker}.csv"
    
    # Vérifier si le fichier existe et n'est pas trop vieux (< 7 jours)
    if os.path.exists(file_path):
        file_age = (pd.Timestamp.now() - pd.Timestamp(os.path.getmtime(file_path), unit='s')).days
        if file_age < 7:
            print(f"📂 Utilisation du cache local pour {ticker} (age: {file_age} jours)")
            return file_path
    
    # Télécharger les données
    print(f"📥 Téléchargement des données pour {ticker} (5 ans, hourly)...")
    try:
        df = yf.download(ticker, period="5y", interval="1h", auto_adjust=True, progress=False)
        
        if df.empty:
            print(f"❌ Pas de données pour {ticker}")
            return None
            
        # Nettoyer si MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker, axis=1, level=1)
        
        df.to_csv(file_path)
        print(f"✅ Données sauvegardées : {file_path} ({len(df)} lignes)")
        return file_path
        
    except Exception as e:
        print(f"❌ Erreur téléchargement {ticker}: {e}")
        return None

def train_model(ticker, timesteps=5_000_000):
    """Entraîne un modèle PPO pour un ticker spécifique"""
    
    print("\n" + "="*70)
    print(f"🎯 TRAINING: {ticker}")
    print("="*70)
    
    # 1. PRÉ-CHARGEMENT DES DONNÉES (UNE SEULE FOIS)
    csv_path = download_and_cache_data(ticker)
    if csv_path is None:
        print(f"⚠️ Impossible de charger les données pour {ticker}, skip.")
        return
    
    # 2. Initialiser W&B
    run = wandb.init(
        project=WANDB_CONFIG["project"],
        name=f"{ticker}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}",
        config={
            "ticker": ticker,
            "timesteps": timesteps,
            "algorithm": "PPO",
            "policy": "MlpPolicy",
            **TRAINING_CONFIG
        },
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        reinit=True
    )
    
    # 3. CONFIGURATION GPU/CPU
    device = "cuda" if USE_GPU else "cpu"
    print(f"🖥️  Device: {device}")
    
    # 4. ENVIRONNEMENTS PARALLÈLES (Exploitation CPU maximale)
    n_envs = 64  # <-- AUGMENTÉ À 64 pour utiliser tous tes cœurs
    print(f"🔧 Création de {n_envs} environnements...")
    
    # Lambda pour créer des environnements qui lisent le MÊME fichier CSV
    def make_env():
        return TradingEnv(csv_path=csv_path)
    
    env = SubprocVecEnv([make_env for _ in range(n_envs)])
    
    # 5. CONFIGURATION RÉSEAU NEURONAL (Plus gros pour GPU)
    policy_kwargs = dict(
        net_arch=dict(
            pi=[512, 512, 512],  # Policy network (actions)
            vf=[512, 512, 512]   # Value network (critique)
        )
    )
    
    print("🧠 Création du modèle PPO...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        learning_rate=1e-4,
        
        # PARAMÈTRES OPTIMISÉS POUR GPU
        batch_size=4096,      # Batch énorme pour GPU (vs 64 par défaut)
        n_steps=2048,         # Steps par env avant update (64 * 2048 = 131k buffer)
        n_epochs=10,          # Passe 10 fois sur les données
        
        policy_kwargs=policy_kwargs,
        tensorboard_log=f"{TRAINING_CONFIG.get('tensorboard_dir', 'tensorboard')}/{ticker}"
    )
    
    # 6. CALLBACKS
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=f"models/checkpoints/{ticker}",
        name_prefix=f"{ticker}_model"
    )
    
    wandb_callback = WandbCallback(
        model_save_path=f"models/{ticker}",
        verbose=2
    )
    
    # 7. ENTRAÎNEMENT
    print(f"🚀 Début entraînement ({timesteps:,} timesteps)...")
    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_callback, wandb_callback],
        progress_bar=True
    )
    
    # 8. SAUVEGARDE FINALE
    model_path = f"models/{ticker}_final.zip"
    model.save(model_path)
    print(f"💾 Modèle sauvegardé: {model_path}")
    
    # 9. NETTOYAGE
    env.close()
    wandb.finish()
    
    print(f"✅ Training terminé pour {ticker}")

if __name__ == "__main__":
    # Liste des tickers à entraîner
    tickers = ["NVDA", "MSFT", "AAPL", "GOOGL", "AMZN", "TSLA", "AMD"]
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] {ticker}")
        train_model(ticker, timesteps=5_000_000)
