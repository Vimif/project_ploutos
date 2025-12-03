# config/settings.py
from pathlib import Path
import os
import socket

# Détection automatique de la machine
HOSTNAME = socket.gethostname()
IS_WSL = 'WSL' in os.uname().release.upper()
IS_PROXMOX = HOSTNAME.startswith('pve') or HOSTNAME.startswith('lxc')

# Chemins du projet (CODE - dans Git)
BASE_DIR = Path(__file__).parent.parent

# Chemins des DONNÉES (HORS Git - Partagé via NFS)
if os.path.exists('/mnt/shared'):
    # Production (NFS monté)
    DATA_DIR = Path('/mnt/shared/ploutos_data')
else:
    # Développement local (fallback)
    DATA_DIR = BASE_DIR / 'data'
    print("⚠️  NFS non monté, utilisation locale")

# Créer structure
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = DATA_DIR / 'models'
LOGS_DIR = DATA_DIR / 'logs'
TRADES_DIR = DATA_DIR / 'trade_history'

for d in [MODELS_DIR, LOGS_DIR, TRADES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"🖥️  Machine: {HOSTNAME}")
print(f"📂 DATA_DIR: {DATA_DIR}")
print(f"🤖 MODELS_DIR: {MODELS_DIR}")

# Config selon la machine
if IS_WSL or 'tower' in HOSTNAME.lower():
    # PC-TOUR : Mode Training
    ROLE = "TRAINING"
    USE_GPU = True
    TRAINING_CONFIG = {
        'n_envs': 8,  # Plus de parallélisme sur GPU
        'total_timesteps': 5_000_000,
    }
    
elif IS_PROXMOX or 'pve' in HOSTNAME.lower():
    # PROXMOX : Mode Production
    ROLE = "PRODUCTION"
    USE_GPU = False
    TRAINING_CONFIG = {
        'n_envs': 2,  # Moins de charge CPU
        'total_timesteps': 1_000_000,  # Training léger si besoin
    }
    
else:
    # Défaut
    ROLE = "DEV"
    USE_GPU = False
    TRAINING_CONFIG = {
        'n_envs': 4,
        'total_timesteps': 2_000_000,
    }

print(f"🎭 Rôle: {ROLE}")

# Trading Config
TRADING_CONFIG = {
    'initial_capital': 100_000,
    'paper_trading': True,
    'check_interval_minutes': 60 if ROLE == "PRODUCTION" else 10,
}
