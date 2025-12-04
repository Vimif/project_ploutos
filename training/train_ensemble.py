import sys
import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.environment import TradingEnv

N_ENVS = 64
TIMESTEPS = 5_000_000

def train_ensemble_member(ticker, csv_path, seed, member_id):
    """Entraîne un membre de l'ensemble avec une seed différente"""
    
    print(f"\n🤖 Entraînement Ensemble #{member_id} (seed={seed})")
    
    def make_env():
        return TradingEnv(csv_path=csv_path)
    
    env = SubprocVecEnv([make_env for _ in range(N_ENVS)])
    
    policy_kwargs = dict(net_arch=dict(pi=[512, 512, 512], vf=[512, 512, 512]))
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cuda",
        learning_rate=1e-4,
        batch_size=8192,
        n_steps=2048,
        n_epochs=10,
        seed=seed,  # SEED DIFFÉRENTE = modèle différent
        policy_kwargs=policy_kwargs
    )
    
    model.learn(total_timesteps=TIMESTEPS, progress_bar=True)
    
    model_path = f"models/ensemble/{ticker}_member{member_id}_seed{seed}.zip"
    os.makedirs("models/ensemble", exist_ok=True)
    model.save(model_path)
    
    print(f"✅ Membre #{member_id} sauvegardé : {model_path}")
    
    env.close()
    return model_path

def train_full_ensemble(ticker, csv_path, n_members=5):
    """Entraîne N modèles avec des seeds différentes"""
    
    print("="*70)
    print(f"🎯 ENSEMBLE LEARNING : {ticker} ({n_members} modèles)")
    print("="*70)
    
    seeds = [42, 123, 456, 789, 999][:n_members]
    model_paths = []
    
    for i, seed in enumerate(seeds, 1):
        path = train_ensemble_member(ticker, csv_path, seed, member_id=i)
        model_paths.append(path)
    
    # Sauvegarder la liste des chemins
    manifest_path = f"models/ensemble/{ticker}_manifest.txt"
    with open(manifest_path, 'w') as f:
        f.write("\n".join(model_paths))
    
    print(f"\n✅ Ensemble complet sauvegardé ! Manifest : {manifest_path}")
    
    return model_paths

if __name__ == "__main__":
    # Entraîner ensemble pour NVDA
    train_full_ensemble("NVDA", "data_cache/NVDA.csv", n_members=5)
