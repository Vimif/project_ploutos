# ai_trainer.py
# ---------------------------------------------------------
# ENTRAÎNEUR IA "IRON MAN TURBO" - RTX 3060 OPTIMIZED
# ---------------------------------------------------------
import pandas as pd
import numpy as np
import os
import torch
import sys
import shutil
from datetime import datetime

# --- W&B INTEGRATION ---
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    USE_WANDB = True
    print("📊 Monitoring W&B : ACTIVÉ")
except ImportError:
    USE_WANDB = False
    print("⚠️ W&B non installé. Fallback sur TensorBoard.")

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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from trading_env import StockTradingEnv
from trading_bot import TradingBrain

# ============================================
# 🎮 CONFIGURATION GPU & HARDWARE
# ============================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n{'='*60}")
print(f"🎮 Device: {device.upper()}")
if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   Compute Capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
    torch.backends.cudnn.benchmark = True  # Optimisation CUDA
print(f"{'='*60}\n")

# Exploitation multi-cœurs Ryzen 7 9800X3D
torch.set_num_threads(8)
os.environ['OMP_NUM_THREADS'] = '8'

# ============================================
# 📊 CONFIGURATION ENTRAÎNEMENT
# ============================================
MODEL_FILE = "ppo_trading_brain"
PROJECT_NAME = "Ploutos_Trading_V40_GPU"

# SECTEURS USINE
SECTORS = {
    "TECH": ["NVDA", "AAPL", "MSFT", "AMD", "TSLA", "QQQ", "META", "GOOGL"],
    "DEFENSIVE": ["KO", "PG", "JNJ", "MCD", "WMT", "XLV", "CVS", "PFE"],
    "ENERGY": ["XOM", "CVX", "SLB", "XLE", "CAT", "OXY", "COP"],
    "CRYPTO": ["COIN", "MSTR", "MARA", "BITO", "RIOT", "CLSK"]
}

# Gestion Arguments CLI
TRAINING_TICKERS = ["SPY", "NVDA", "JPM", "XOM", "GLD", "TSLA", "AMZN", "GOOGL"]
if len(sys.argv) > 1:
    arg = sys.argv[1].upper()
    if arg in SECTORS:
        print(f"🏭 CIBLE SECTORIELLE : {arg}")
        TRAINING_TICKERS = SECTORS[arg]
        MODEL_FILE = f"brain_{arg.lower()}"

# Hyperparamètres AGRESSIFS (optimisés pour RTX 3060)
EPOCHS = 10                    # Cycles d'entraînement complets
TOTAL_TIMESTEPS = 500000       # Steps par ticker (3.3x plus qu'avant)
N_ENVS = 8                     # Environnements parallèles (1 par cœur CPU)
EVAL_FREQ = 10000              # Fréquence d'évaluation
SAVE_FREQ = 50000              # Fréquence de sauvegarde checkpoints

# Cache global des données
DATA_CACHE = {}

# ============================================
# 📦 PRÉCHARGEMENT DONNÉES EN RAM
# ============================================
def precompute_all_data():
    """Charge toutes les données en RAM pour éviter I/O pendant l'entraînement"""
    print(f"\n💾 [{MODEL_FILE}] Préchargement des données en RAM...")
    brain = TradingBrain()
    
    for i, ticker in enumerate(TRAINING_TICKERS, 1):
        print(f"   [{i}/{len(TRAINING_TICKERS)}] Téléchargement {ticker}...", end=" ")
        df = brain.telecharger_donnees(ticker)
        
        if df is None or df.empty:
            print("❌ ERREUR")
            continue
        
        df = df.copy()
        close = df['Close']
        
        # Calcul Indicateurs Techniques
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 0.001)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # SMA Ratio
        df['SMA_Ratio'] = close / close.rolling(50).mean()
        
        # MACD
        e1 = close.ewm(span=12).mean()
        e2 = close.ewm(span=26).mean()
        df['MACD'] = e1 - e2
        
        # Volatilité
        df['Volatility'] = close.rolling(20).std() / close.rolling(20).mean()
        
        # Volume normalisé
        df['Volume_Norm'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        DATA_CACHE[ticker] = df.dropna().reset_index(drop=True)
        print(f"✅ {len(df)} lignes")
    
    print(f"\n✅ {len(DATA_CACHE)} tickers chargés ({sum(len(v) for v in DATA_CACHE.values())} lignes totales)\n")

# ============================================
# 🏗️ FACTORY D'ENVIRONNEMENTS
# ============================================
def make_env(ticker, rank=0):
    """Crée un environnement de trading pour un ticker donné"""
    def _init():
        df = DATA_CACHE.get(ticker, pd.DataFrame())
        if df.empty:
            raise ValueError(f"Pas de données pour {ticker}")
        
        env = StockTradingEnv(df)
        env.seed(42 + rank)
        return env
    
    return _init

# ============================================
# 🎯 FONCTION PRINCIPALE D'ENTRAÎNEMENT
# ============================================
def train_model():
    """Entraîne le modèle avec environnements parallèles et GPU"""
    
    # Préchargement
    precompute_all_data()
    if not DATA_CACHE:
        print("❌ Aucune donnée chargée. Arrêt.")
        return
    
    print(f"🚀 Démarrage Entraînement : {MODEL_FILE}")
    print(f"   • Device: {device}")
    print(f"   • Epochs: {EPOCHS}")
    print(f"   • Steps/ticker: {TOTAL_TIMESTEPS:,}")
    print(f"   • Environnements parallèles: {N_ENVS}")
    
    # Nettoyage fichiers locaux
    for ext in [".zip", "_vecnorm.pkl"]:
        filepath = MODEL_FILE + ext
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"🗑️  Suppression ancien {filepath}")
    
    # Configuration Architecture selon LSTM ou MLP
    if USE_LSTM:
        policy_kwargs = dict(
            lstm_hidden_size=512,        # Taille LSTM augmentée
            n_lstm_layers=2,             # 2 couches LSTM empilées
            enable_critic_lstm=True,
            shared_lstm=False,
            net_arch=[512, 512]          # Couches denses après LSTM
        )
        AlgoClass = RecurrentPPO
        policy_type = "MlpLstmPolicy"
        batch_size = 2048                # Optimisé pour 12GB VRAM
        n_steps = 8192
        n_epochs_ppo = 10
    else:
        policy_kwargs = dict(
            net_arch=dict(
                pi=[1024, 1024, 512, 256],  # Policy network profond
                vf=[1024, 1024, 512, 256]   # Value network profond
            ),
            activation_fn=torch.nn.ReLU
        )
        AlgoClass = PPO
        policy_type = "MlpPolicy"
        batch_size = 4096                # Gros batch pour GPU
        n_steps = 8192
        n_epochs_ppo = 10
    
    print(f"\n📐 Architecture:")
    print(f"   • Type: {policy_type}")
    print(f"   • Batch Size: {batch_size}")
    print(f"   • N Steps: {n_steps}")
    print(f"   • PPO Epochs: {n_epochs_ppo}\n")
    
    model = None
    start_time = datetime.now()
    
    # ============================================
    # 🔄 BOUCLE D'ENTRAÎNEMENT MULTI-EPOCHS
    # ============================================
    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"🔄 {MODEL_FILE} - EPOCH {epoch+1}/{EPOCHS}")
        print(f"{'='*60}\n")
        
        for ticker_idx, ticker in enumerate(TRAINING_TICKERS, 1):
            if ticker not in DATA_CACHE:
                print(f"⚠️  Skipping {ticker} (no data)")
                continue
            
            print(f"\n📈 [{ticker_idx}/{len(TRAINING_TICKERS)}] Training on {ticker}...")
            
            # --- CRÉATION ENVIRONNEMENTS PARALLÈLES ---
            # Utilise SubprocVecEnv pour vrai parallélisme (multiprocessing)
            env_fns = [make_env(ticker, rank=i) for i in range(N_ENVS)]
            
            if N_ENVS > 1:
                base_env = SubprocVecEnv(env_fns)
            else:
                base_env = DummyVecEnv(env_fns)
            
            # Normalisation (critique pour la stabilité)
            norm_env = VecNormalize(
                base_env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                clip_reward=10.0,
                gamma=0.99
            )
            
            # --- INITIALISATION W&B ---
            if USE_WANDB:
                run = wandb.init(
                    project=PROJECT_NAME,
                    group=MODEL_FILE,
                    name=f"{ticker}_Ep{epoch+1}",
                    config={
                        "policy_type": policy_type,
                        "device": device,
                        "total_timesteps": TOTAL_TIMESTEPS,
                        "ticker": ticker,
                        "epoch": epoch + 1,
                        "n_envs": N_ENVS,
                        "batch_size": batch_size,
                        "n_steps": n_steps,
                        "learning_rate": 3e-4
                    },
                    reinit=True,
                    sync_tensorboard=True,
                    monitor_gym=True
                )
                tensorboard_log = f"./runs/{run.id}"
            else:
                tensorboard_log = f"./logs/{MODEL_FILE}"
            
            # --- CRÉATION / UPDATE MODÈLE ---
            if model is None:
                print(f"🏗️  Création du modèle initial...")
                model = AlgoClass(
                    policy_type,
                    norm_env,
                    verbose=1,
                    device=device,
                    learning_rate=3e-4,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=n_epochs_ppo,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    clip_range_vf=None,
                    ent_coef=0.01,              # Encourage exploration
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=tensorboard_log
                )
            else:
                print(f"🔄 Mise à jour environnement du modèle...")
                model.set_env(norm_env)
            
            # --- CALLBACKS ---
            callbacks = []
            
            # Checkpoint réguliers
            checkpoint_callback = CheckpointCallback(
                save_freq=SAVE_FREQ,
                save_path=f"./checkpoints/{MODEL_FILE}/",
                name_prefix=f"{ticker}_ep{epoch}"
            )
            callbacks.append(checkpoint_callback)
            
            # W&B Callback
            if USE_WANDB:
                wandb_callback = WandbCallback(
                    gradient_save_freq=100,
                    model_save_path=f"models/{run.id}",
                    verbose=2
                )
                callbacks.append(wandb_callback)
            
            # --- ENTRAÎNEMENT ---
            print(f"🎓 Apprentissage {TOTAL_TIMESTEPS:,} steps...")
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                callback=callbacks,
                reset_num_timesteps=False,  # Continue l'apprentissage
                progress_bar=True
            )
            
            # --- SAUVEGARDE ---
            print(f"💾 Sauvegarde du modèle...")
            model.save(MODEL_FILE)
            norm_env.save(MODEL_FILE + "_vecnorm.pkl")
            
            # Fermeture W&B
            if USE_WANDB:
                run.finish()
            
            # Nettoyage environnement
            norm_env.close()
            
            elapsed = datetime.now() - start_time
            print(f"⏱️  Temps écoulé: {elapsed}")
    
    # ============================================
    # ✅ FIN DE L'ENTRAÎNEMENT
    # ============================================
    total_time = datetime.now() - start_time
    print(f"\n{'='*60}")
    print(f"✅ ENTRAÎNEMENT TERMINÉ")
    print(f"{'='*60}")
    print(f"📊 Statistiques:")
    print(f"   • Durée totale: {total_time}")
    print(f"   • Modèle: {MODEL_FILE}.zip")
    print(f"   • Normalisation: {MODEL_FILE}_vecnorm.pkl")
    print(f"   • Epochs: {EPOCHS}")
    print(f"   • Tickers: {len(TRAINING_TICKERS)}")
    print(f"   • Steps totaux: {EPOCHS * len(TRAINING_TICKERS) * TOTAL_TIMESTEPS:,}")
    print(f"{'='*60}\n")

# ============================================
# 🚀 POINT D'ENTRÉE
# ============================================
if __name__ == "__main__":
    try:
        train_model()
    except KeyboardInterrupt:
        print("\n\n⚠️  Entraînement interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
