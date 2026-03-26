# config/settings.py
"""Configuration globale du projet Ploutos."""

from pathlib import Path
import os
import socket

# Détection de la machine (debug)
HOSTNAME = socket.gethostname()

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

# Broker Config
# Choix du broker: 'etoro' (défaut) ou 'alpaca'
# Peut être overridé par la variable d'environnement BROKER
BROKER = os.getenv('BROKER', 'etoro').lower()

# WandB Config
WANDB_CONFIG = {
    'project': 'Ploutos_Trading_V50_ULTIMATE',
    'entity': 'vimif-perso',
}

print(f"🖥️  Machine: {HOSTNAME}")
print(f"📂 DATA_DIR: {DATA_DIR}")
print(f"🏦 Broker: {BROKER}")
