import sys
import os
import time
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback
import wandb
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="gym")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Import du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.environment import TradingEnv
from config.settings import TRAINING_CONFIG, WANDB_CONFIG, USE_GPU

# --- CONFIGURATION ---
N_ENVS = 32            # Nombre d'environnements parallèles (CPU)
BATCH_SIZE = 4096      # Taille du batch pour GPU
N_STEPS = 2048         # Pas par environnement
TIMESTEPS = 5_000_000  # Durée totale

def print_header(ticker, current, total):
    """Affiche un joli header stylé"""
    print("\n" + "="*70)
    print(f"🚀 ENTRAÎNEMENT [{current}/{total}] : {ticker}")
    print("="*70)
    print(f"📅 Date : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  Machine : {os.uname().nodename}")
    print(f"🔧 Coeurs CPU : {N_ENVS} envs parallèles")
    print(f"🎮 GPU Mode : {'ACTIVÉ (CUDA)' if USE_GPU else 'DÉSACTIVÉ (CPU)'}")
    print("="*70 + "\n")

def download_and_cache_data(ticker, data_dir="data_cache"):
    """Télécharge et met en cache les données (Max 730j pour 1h)"""
    os.makedirs(data_dir, exist_ok=True)
    file_path = f"{data_dir}/{ticker}.csv"
    
    # Vérifier cache récent (< 24h pour être sûr d'avoir les dernières données)
    if os.path.exists(file_path):
        file_age_hours = (pd.Timestamp.now() - pd.Timestamp(os.path.getmtime(file_path), unit='s')).total_seconds() / 3600
        if file_age_hours < 24:
            print(f"📂 [CACHE] Chargement depuis le disque : {ticker} ({file_age_hours:.1f}h old)")
            return file_path
    
    print(f"📥 [YAHOO] Téléchargement des données pour {ticker} (2 ans, 1h)...")
    try:
        # ATTENTION: period="730d" maximum pour interval="1h"
        df = yf.download(ticker, period="730d", interval="1h", auto_adjust=True, progress=False)
        
        if df.empty:
            print(f"❌ [ERREUR] Pas de données reçues pour {ticker}")
            return None
            
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker, axis=1, level=1)
        
        # Nettoyage et sauvegarde
        df = df.dropna()
        df.to_csv(file_path)
        print(f"✅ [SUCCÈS] {len(df)} bougies sauvegardées dans {file_path}")
        return file_path
        
    except Exception as e:
        print(f"❌ [CRASH] Erreur téléchargement {ticker}: {e}")
        return None

def train_model(ticker, timesteps=TIMESTEPS, current=1, total=1):
    
    print_header(ticker, current, total)
    
    # 1. CACHE DES DONNÉES
    csv_path = download_and_cache_data(ticker)
    if csv_path is None:
        print(f"⚠️  SKIP : Impossible de charger les données pour {ticker}")
        return

    # 2. WANDB INIT
    run_name = f"{ticker}_ULTRA_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"
    run = wandb.init(
        project=WANDB_CONFIG["project"],
        name=run_name,
        config={
            "ticker": ticker,
            "timesteps": timesteps,
            "n_envs": N_ENVS,
            "batch_size": BATCH_SIZE,
            "device": "cuda" if USE_GPU else "cpu",
            **TRAINING_CONFIG
        },
        sync_tensorboard=True,
        reinit=True
    )
    
    # 3. CRÉATION ENVIRONNEMENT MULTI-PROCESS
    print(f"🔨 Création du cluster d'environnements ({N_ENVS} workers)...")
    def make_env():
        return TradingEnv(csv_path=csv_path) # Chaque worker lit son CSV localement
    
    env = SubprocVecEnv([make_env for _ in range(N_ENVS)])
    
    # 4. MODÈLE PPO (CONFIGURATION LOURDE)
    policy_kwargs = dict(
        net_arch=dict(pi=[512, 512, 512], vf=[512, 512, 512]) # Gros cerveau
    )
    
    print("🧠 Initialisation du modèle PPO (Large Network)...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cuda" if USE_GPU else "cpu",
        learning_rate=1e-4,     # Un peu plus lent pour la stabilité
        batch_size=BATCH_SIZE,  # Optimisé pour RTX 3080
        n_steps=N_STEPS,        # Buffer conséquent
        n_epochs=10,            # Bien digérer les données
        ent_coef=0.01,          # Encourager l'exploration
        policy_kwargs=policy_kwargs,
        tensorboard_log=f"{TRAINING_CONFIG.get('tensorboard_dir', 'tensorboard')}/{ticker}"
    )
    
    # 5. CALLBACKS
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // N_ENVS, 1), # Ajusté selon n_envs
        save_path=f"models/checkpoints/{ticker}",
        name_prefix=f"{ticker}_model"
    )
    
    wandb_callback = WandbCallback(
        model_save_path=f"models/{ticker}",
        verbose=2,
        gradient_save_freq=1000
    )
    
    # 6. ENTRAÎNEMENT
    start_time = time.time()
    print(f"\n🏁 DÉBUT DE LA COURSE ({timesteps:,} steps)... Go!")
    
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=[checkpoint_callback, wandb_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Interruption manuelle ! Sauvegarde d'urgence...")
    
    duration = (time.time() - start_time) / 60
    print(f"\n✅ TRAINING TERMINÉ en {duration:.1f} minutes.")
    
    # 7. SAUVEGARDE
    final_path = f"models/{ticker}_final.zip"
    model.save(final_path)
    print(f"💾 Modèle sauvegardé sous : {final_path}")
    
    # Nettoyage
    env.close()
    wandb.finish()

if __name__ == "__main__":
    # Liste des tickers (Top Liquidité)
    tickers = ["NVDA", "TSLA", "AAPL", "AMD", "MSFT", "AMZN", "GOOGL"]
    
    total_tickers = len(tickers)
    print(f"\n🔥 DÉMARRAGE DE LA SESSION D'ENTRAÎNEMENT ({total_tickers} actifs)")
    
    for i, ticker in enumerate(tickers, 1):
        train_model(ticker, current=i, total=total_tickers)
