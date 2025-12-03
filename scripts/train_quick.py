#!/usr/bin/env python3
# scripts/train_quick.py
"""Entraînement rapide des 4 cerveaux"""

# === FIX PATH ===
import sys
from pathlib import Path
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))
# ================

from stable_baselines3 import PPO
from core.environment import TradingEnv
from config.tickers import SECTORS
from config.settings import MODELS_DIR
import torch

print("🚀 ENTRAÎNEMENT RAPIDE DES 4 CERVEAUX")
print("="*70)

# Modèles à entraîner
models_to_train = {
    'brain_crypto': 'BTC-USD',
    'brain_defensive': 'SPY',
    'brain_energy': 'XOM',
    'brain_tech': 'NVDA'
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️  Device: {device}")

for model_name, ticker in models_to_train.items():
    print(f"\n📈 Entraînement de {model_name} sur {ticker}...")
    
    try:
        # Créer environnement
        env = TradingEnv(ticker)
        
        # Créer modèle
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            verbose=0,
            device=device
        )
        
        # Entraîner (version rapide : 100k timesteps)
        print(f"   ⏳ Training 100,000 timesteps...")
        model.learn(total_timesteps=100_000, progress_bar=True)
        
        # Sauvegarder
        model_path = MODELS_DIR / f"{model_name}.zip"
        model.save(model_path)
        print(f"   ✅ Sauvegardé: {model_path}")
        
        env.close()
        
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*70)
print("✅ ENTRAÎNEMENT TERMINÉ")
print(f"📁 Modèles dans: {MODELS_DIR}")
