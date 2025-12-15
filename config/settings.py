# config/settings.py
"""Configuration globale du projet Ploutos"""

# === FIX PATH ===
import sys
from pathlib import Path
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))
# ================

import os
import socket

# Détection automatique de la machine
HOSTNAME = socket.gethostname()
IS_WSL = 'WSL' in os.uname().release.upper() if hasattr(os, 'uname') else False
IS_PROXMOX = HOSTNAME.startswith('pve') or HOSTNAME.startswith('lxc') or HOSTNAME.lower().startswith('vps')

# Chemins du projet
BASE_DIR = Path(__file__).parent.parent

# Chemins des DONNÉES
if os.path.exists('/mnt/shared'):
    DATA_DIR = Path('/mnt/shared/ploutos_data')
else:
    DATA_DIR = BASE_DIR / 'data'

# Créer structure
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = DATA_DIR / 'models'
LOGS_DIR = DATA_DIR / 'logs'
TRADES_DIR = DATA_DIR / 'trade_history'

for d in [MODELS_DIR, LOGS_DIR, TRADES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Features
N_FEATURES = 30

# Training Config
if IS_WSL or 'BBC' in HOSTNAME.lower():
    ROLE = "TRAINING"
    USE_GPU = True
    TRAINING_CONFIG = {
        'n_envs': 64,
        'total_timesteps': 5_000_000,
        'eval_freq': 50_000,
        'n_eval_episodes': 5,
        'learning_rate': 3e-4,
        'n_steps': 8192,
        'batch_size': 8192,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'max_grad_norm': 0.5,
    }
elif IS_PROXMOX:
    ROLE = "PRODUCTION"
    USE_GPU = False
    TRAINING_CONFIG = {
        'n_envs': 2,
        'total_timesteps': 1_000_000,
        'eval_freq': 50_000,
        'n_eval_episodes': 5,
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'max_grad_norm': 0.5,
    }
else:
    ROLE = "DEV"
    USE_GPU = False
    TRAINING_CONFIG = {
        'n_envs': 4,
        'total_timesteps': 2_000_000,
        'eval_freq': 50_000,
        'n_eval_episodes': 5,
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'max_grad_norm': 0.5,
    }

# Trading Config
TRADING_CONFIG = {
    'initial_capital': 100_000,
    'paper_trading': True,
    'check_interval_minutes': 60 if ROLE == "PRODUCTION" else 10,
}

# WandB Config
WANDB_CONFIG = {
    'project': 'Ploutos_Trading_V50_ULTIMATE',
    'entity': 'vimif-perso',
}

# ========== ALPACA API CONFIGURATION ==========
# Clés API pour Alpaca (Paper Trading)
# Priorité : Variables d'environnement > Fichier .env > Valeurs par défaut

# Charger .env si disponible
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv pas installé, utiliser variables d'environnement

ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

# Validation
if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    import warnings
    warnings.warn(
        "⚠️  ALPACA_API_KEY ou ALPACA_SECRET_KEY non configurés.\n"
        "Définissez-les dans les variables d'environnement ou dans un fichier .env\n"
        "Exemple:\n"
        "  export ALPACA_API_KEY='PKxxxxxxxxxx'\n"
        "  export ALPACA_SECRET_KEY='xxxxxxxxxx'",
        UserWarning
    )

print(f"🔖  Machine: {HOSTNAME}")
print(f"🎭 Rôle: {ROLE}")
print(f"📂 DATA_DIR: {DATA_DIR}")
if ALPACA_API_KEY:
    print(f"🔑 Alpaca API: {'*' * 10}{ALPACA_API_KEY[-4:]} (Paper Trading)")
