# config/settings.py
"""Configuration globale du projet Ploutos"""

from pathlib import Path
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

# Broker Config
# Choix du broker: 'etoro' (défaut) ou 'alpaca'
# Peut être overridé par la variable d'environnement BROKER
BROKER = os.getenv('BROKER', 'etoro').lower()

# Trading Config
TRADING_CONFIG = {
    'initial_capital': 100_000,
    'paper_trading': True,
    'check_interval_minutes': 60 if ROLE == "PRODUCTION" else 10,
    'broker': BROKER,
}

# WandB Config
WANDB_CONFIG = {
    'project': 'Ploutos_Trading_V50_ULTIMATE',
    'entity': 'vimif-perso',
}

print(f"🖥️  Machine: {HOSTNAME}")
print(f"🎭 Rôle: {ROLE}")
print(f"📂 DATA_DIR: {DATA_DIR}")
print(f"🏦 Broker: {BROKER}")
