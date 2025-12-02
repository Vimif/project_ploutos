# ai_trainer.py
# ---------------------------------------------------------
# ENTRAÎNEUR IA "IRON MAN ULTIMATE" - RTX 3080 MAX PERFORMANCE
# Version ultra-optimisée: FP16, SubprocVecEnv, Batch Size Maximum
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
# 🎮 CONFIGURATION GPU & HARDWARE MAXIMALE
# ============================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n{'='*60}")
print(f"🎮 Device: {device.upper()}")
if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    compute_cap = torch.cuda.get_device_properties(0).major
    
    print(f"   GPU: {gpu_name}")
    print(f"   VRAM: {vram_total:.1f} GB")
    print(f"   Compute Capability: {compute_cap}.{torch.cuda.get_device_properties(0).minor}")
    
    # OPTIMISATIONS CUDA AVANCÉES
    torch.backends.cudnn.benchmark = True        # Auto-tune kernels
    torch.backends.cuda.matmul.allow_tf32 = True # TensorFloat-32 (RTX 30xx)
    torch.set_float32_matmul_precision('high')   # Mixed precision
    
    # Vider le cache GPU
    torch.cuda.empty_cache()
    
    print(f"   ⚡ Optimisations CUDA activées (TF32, cuDNN benchmark)")
    
print(f"{'='*60}\n")

# Exploitation MAXIMALE multi-cœurs Ryzen 7 9800X3D
torch.set_num_threads(8)
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'

# ============================================
# 📊 CONFIGURATION ENTRAÎNEMENT AGRESSIF
# ============================================
MODEL_FILE = "ppo_trading_brain"
PROJECT_NAME = "Ploutos_Trading_V50_ULTIMATE"

# SECTEURS USINE
SECTORS = {
    "TECH": ["NVDA", "AAPL", "MSFT", "AMD", "TSLA", "QQQ", "META", "GOOGL", "AMZN", "NFLX"],
    "DEFENSIVE": ["KO", "PG", "JNJ", "MCD", "WMT", "XLV", "CVS", "PFE", "UNH", "ABT"],
    "ENERGY": ["XOM", "CVX", "SLB", "XLE", "CAT", "OXY", "COP", "EOG", "PSX"],
    "CRYPTO": ["COIN", "MSTR", "MARA", "BITO", "RIOT", "CLSK", "HUT", "CIFR"]
}

# Gestion Arguments CLI
TRAINING_TICKERS = ["SPY", "NVDA", "JPM", "XOM", "GLD", "TSLA", "AMZN", "GOOGL"]
if len(sys.argv) > 1:
    arg = sys.argv[1].upper()
    if arg in SECTORS:
        print(f"🏭 CIBLE SECTORIELLE : {arg}")
        TRAINING_TICKERS = SECTORS[arg]
        MODEL_FILE = f"brain_{arg.lower()}"

# Hyperparamètres ULTRA-AGRESSIFS (RTX 3080 10GB + 8 cores)
EPOCHS = 12                    # Plus d'epochs pour meilleur apprentissage
TOTAL_TIMESTEPS = 750000       # 750k steps par ticker (1.5x plus qu'avant)
N_ENVS = 8                     # 8 environnements parallèles (1 par cœur)
EVAL_FREQ = 15000              # Évaluation plus fréquente
SAVE_FREQ = 75000              # Checkpoints tous les 75k steps

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
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        # Calcul Indicateurs Techniques AVANCÉS
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 0.001)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # SMA Ratio (court et long terme)
        df['SMA_20'] = close.rolling(20).mean()
        df['SMA_50'] = close.rolling(50).mean()
        df['SMA_Ratio'] = close / df['SMA_50']
        
        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = close.rolling(20).mean()
        df['BB_Std'] = close.rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
        df['BB_Position'] = (close - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volatilité (ATR simplifié)
        df['Volatility'] = close.rolling(20).std() / close.rolling(20).mean()
        
        # Volume normalisé
        df['Volume_Norm'] = volume / volume.rolling(20).mean()
        
        # Momentum
        df['Momentum'] = close.pct_change(periods=10)
        
        DATA_CACHE[ticker] = df.dropna().reset_index(drop=True)
        print(f"✅ {len(df)} lignes")
    
    total_rows = sum(len(v) for v in DATA_CACHE.values())
    print(f"\n✅ {len(DATA_CACHE)} tickers | {total_rows:,} lignes totales | RAM: ~{total_rows * 0.001:.1f} MB\n")

# ============================================
# 🏗️ FACTORY D'ENVIRONNEMENTS (MULTIPROCESSING)
# ============================================
def make_env(ticker, rank=0):
    """Crée un environnement de trading pour un ticker donné"""
    def _init():
        df = DATA_CACHE.get(ticker, pd.DataFrame())
        if df.empty:
            raise ValueError(f"Pas de données pour {ticker}")
        
        env = StockTradingEnv(df)
        # Gymnasium gère le seed automatiquement
        return env
    
    return _init

# ============================================
# 🎯 FONCTION PRINCIPALE D'ENTRAÎNEMENT
# ============================================
def train_model():
    """Entraîne le modèle avec optimisations maximales GPU + CPU"""
    
    # Préchargement
    precompute_all_data()
    if not DATA_CACHE:
        print("❌ Aucune donnée chargée. Arrêt.")
        return
    
    print(f"🚀 Démarrage Entraînement ULTIMATE : {MODEL_FILE}")
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
            lstm_hidden_size=512,        # LSTM large
            n_lstm_layers=2,             # 2 couches LSTM
            enable_critic_lstm=True,
            shared_lstm=False,
            net_arch=[1024, 512, 256]    # Réseau profond après LSTM
        )
        AlgoClass = RecurrentPPO
        policy_type = "MlpLstmPolicy"
        batch_size = 4096                # Grand batch pour RTX 3080
        n_steps = 10240                  # Buffer d'expérience augmenté
        n_epochs_ppo = 12
    else:
        policy_kwargs = dict(
            net_arch=dict(
                pi=[2048, 1024, 512, 256],  # Policy network très profond
                vf=[2048, 1024, 512, 256]   # Value network très profond
            ),
            activation_fn=torch.nn.ReLU,
            ortho_init=True                # Meilleure initialisation
        )
        AlgoClass = PPO
        policy_type = "MlpPolicy"
        batch_size = 8192                # BATCH SIZE MAXIMUM pour 10GB VRAM!
        n_steps = 10240
        n_epochs_ppo = 12
    
    print(f"\n📐 Architecture ULTRA:")
    print(f"   • Type: {policy_type}")
    print(f"   • Batch Size: {batch_size:,} (MAXIMUM GPU)")
    print(f"   • N Steps: {n_steps:,}")
    print(f"   • PPO Epochs: {n_epochs_ppo}")
    print(f"   • Learning Rate: 2.5e-4 (optimisée)")
    print(f"   • Total params estimés: ~5-10M\n")
    
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
            
            # --- CRÉATION ENVIRONNEMENTS PARALLÈLES (MULTIPROCESSING) ---
            env_fns = [make_env(ticker, rank=i) for i in range(N_ENVS)]
            
            # Utiliser SubprocVecEnv pour vrai parallélisme (8 cores)
            if N_ENVS > 1:
                print(f"🔧 Création de {N_ENVS} environnements parallèles (SubprocVecEnv)...")
                base_env = SubprocVecEnv(env_fns, start_method='fork')
            else:
                base_env = DummyVecEnv(env_fns)
            
            # Normalisation (critique pour la stabilité)
            norm_env = VecNormalize(
                base_env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                clip_reward=10.0,
                gamma=0.99,
                epsilon=1e-8
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
                        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
                        "total_timesteps": TOTAL_TIMESTEPS,
                        "ticker": ticker,
                        "epoch": epoch + 1,
                        "n_envs": N_ENVS,
                        "batch_size": batch_size,
                        "n_steps": n_steps,
                        "learning_rate": 2.5e-4,
                        "architecture": "ULTIMATE"
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
                print(f"🏗️  Création du modèle ULTIMATE...")
                model = AlgoClass(
                    policy_type,
                    norm_env,
                    verbose=1,
                    device=device,
                    learning_rate=2.5e-4,        # LR optimisé
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=n_epochs_ppo,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    clip_range_vf=None,
                    ent_coef=0.01,               # Exploration
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    target_kl=0.015,             # Early stopping KL divergence
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=tensorboard_log
                )
                
                # Afficher info modèle
                total_params = sum(p.numel() for p in model.policy.parameters())
                print(f"   📊 Paramètres totaux: {total_params:,}")
                print(f"   💾 Taille estimée: {total_params * 4 / 1e6:.1f} MB")
                
            else:
                print(f"🔄 Mise à jour environnement du modèle...")
                model.set_env(norm_env)
            
            # --- CALLBACKS ---
            callbacks = []
            
            # Checkpoint réguliers
            checkpoint_callback = CheckpointCallback(
                save_freq=SAVE_FREQ,
                save_path=f"./checkpoints/{MODEL_FILE}/",
                name_prefix=f"{ticker}_ep{epoch}",
                verbose=1
            )
            callbacks.append(checkpoint_callback)
            
            # W&B Callback
            if USE_WANDB:
                wandb_callback = WandbCallback(
                    gradient_save_freq=500,
                    model_save_path=f"models/{run.id}",
                    verbose=2
                )
                callbacks.append(wandb_callback)
            
            # --- ENTRAÎNEMENT ---
            print(f"🎓 Apprentissage {TOTAL_TIMESTEPS:,} steps...")
            print(f"   Estimation: ~{TOTAL_TIMESTEPS / 1500 / 60:.1f} min avec RTX 3080")
            
            try:
                model.learn(
                    total_timesteps=TOTAL_TIMESTEPS,
                    callback=callbacks,
                    reset_num_timesteps=False,  # Continue l'apprentissage
                    progress_bar=True
                )
            except Exception as e:
                print(f"⚠️ Erreur pendant l'entraînement: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # --- SAUVEGARDE ---
            print(f"💾 Sauvegarde du modèle...")
            model.save(MODEL_FILE)
            norm_env.save(MODEL_FILE + "_vecnorm.pkl")
            
            # Stats GPU
            if device == "cuda":
                vram_used = torch.cuda.memory_allocated() / 1e9
                vram_cached = torch.cuda.memory_reserved() / 1e9
                print(f"   🎮 VRAM utilisée: {vram_used:.2f} GB / Réservée: {vram_cached:.2f} GB")
            
            # Fermeture W&B
            if USE_WANDB:
                run.finish()
            
            # Nettoyage environnement
            norm_env.close()
            
            elapsed = datetime.now() - start_time
            print(f"⏱️  Temps écoulé total: {elapsed}")
            
            # Libération mémoire GPU
            if device == "cuda":
                torch.cuda.empty_cache()
    
    # ============================================
    # ✅ FIN DE L'ENTRAÎNEMENT
    # ============================================
    total_time = datetime.now() - start_time
    total_steps = EPOCHS * len(TRAINING_TICKERS) * TOTAL_TIMESTEPS
    
    print(f"\n{'='*60}")
    print(f"✅ ENTRAÎNEMENT ULTIMATE TERMINÉ")
    print(f"{'='*60}")
    print(f"📊 Statistiques finales:")
    print(f"   • Durée totale: {total_time}")
    print(f"   • Modèle: {MODEL_FILE}.zip")
    print(f"   • Normalisation: {MODEL_FILE}_vecnorm.pkl")
    print(f"   • Epochs: {EPOCHS}")
    print(f"   • Tickers: {len(TRAINING_TICKERS)}")
    print(f"   • Steps totaux: {total_steps:,}")
    print(f"   • Steps/seconde moyen: {total_steps / total_time.total_seconds():.0f}")
    
    if device == "cuda":
        print(f"   • Device: {torch.cuda.get_device_name(0)}")
        print(f"   • VRAM max utilisée: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    print(f"{'='*60}\n")
    print(f"🎯 Modèle prêt pour le trading ! Transférez vers Machine 3:")
    print(f"   scp {MODEL_FILE}.zip root@192.168.x.50:/root/ploutos/")
    print(f"   scp {MODEL_FILE}_vecnorm.pkl root@192.168.x.50:/root/ploutos/")

# ============================================
# 🚀 POINT D'ENTRÉE
# ============================================
if __name__ == "__main__":
    try:
        train_model()
    except KeyboardInterrupt:
        print("\n\n⚠️  Entraînement interrompu par l'utilisateur")
        if device == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"\n❌ ERREUR FATALE: {e}")
        import traceback
        traceback.print_exc()
        if device == "cuda":
            torch.cuda.empty_cache()
