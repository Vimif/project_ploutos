#!/usr/bin/env python3
"""
Script d'entraînement simplifié - Assets fixes
"""
import os
import sys
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Ajouter le parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.universal_environment import UniversalTradingEnv

print("🚀 ENTRAÎNEMENT SIMPLIFIÉ\n")

# ════════════════════════════════════════════════════════════
# 1. CHARGER DONNÉES
# ════════════════════════════════════════════════════════════

print("📥 Chargement données...\n")

TICKERS = ['NVDA', 'MSFT', 'AAPL']
data = {}

for ticker in TICKERS:
    file_path = f'data_cache/{ticker}.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        data[ticker] = df
        print(f"  ✅ {ticker}: {len(df)} lignes")
    else:
        print(f"  ❌ {ticker}: fichier manquant")

if len(data) == 0:
    print("\n❌ Aucune donnée chargée")
    sys.exit(1)

print(f"\n✅ {len(data)} datasets chargés\n")

# ════════════════════════════════════════════════════════════
# 2. CRÉER ENVIRONNEMENT
# ════════════════════════════════════════════════════════════

print("🏗️  Création environnement...\n")

def make_env():
    return UniversalTradingEnv(
        data=data,
        initial_balance=100000,
        commission=0.001,
        max_steps=200
    )

env = DummyVecEnv([make_env])
print("  ✅ Environnement créé\n")

# ════════════════════════════════════════════════════════════
# 3. CRÉER MODÈLE PPO
# ════════════════════════════════════════════════════════════

print("🤖 Création modèle PPO...\n")

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0001,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    verbose=1,
    device='cuda',
    tensorboard_log='logs/tensorboard',
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
)

print("  ✅ Modèle créé\n")

# ════════════════════════════════════════════════════════════
# 4. ENTRAÎNEMENT
# ════════════════════════════════════════════════════════════

print("🏋️  DÉBUT ENTRAÎNEMENT\n")
print("  Timesteps: 50,000")
print("  Device   : cuda")
print("  Assets   : NVDA, MSFT, AAPL")
print()

try:
    model.learn(
        total_timesteps=50000,
        progress_bar=True
    )
    print("\n✅ ENTRAÎNEMENT TERMINÉ\n")
    
except KeyboardInterrupt:
    print("\n⚠️  Interrompu par utilisateur\n")

# ════════════════════════════════════════════════════════════
# 5. SAUVEGARDER
# ════════════════════════════════════════════════════════════

os.makedirs('models/test', exist_ok=True)
model.save('models/test/simple_model.zip')
print("💾 Modèle sauvegardé: models/test/simple_model.zip\n")

env.close()
print("✅ TERMINÉ")
