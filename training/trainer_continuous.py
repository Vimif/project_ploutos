import sys
import os
import pandas as pd
from stable_baselines3 import SAC  # SAC pour actions continues
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.environment_continuous import TradingEnvContinuous

N_ENVS = 32  # SAC consomme plus de RAM, réduit un peu
TIMESTEPS = 3_000_000

def train_continuous(ticker, csv_path):
    """Entraîne avec actions continues (SAC)"""
    
    run_name = f"{ticker}_SAC_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"
    wandb.init(project="Ploutos_SAC", name=run_name, reinit=True)
    
    def make_env():
        return TradingEnvContinuous(csv_path=csv_path)
    
    env = SubprocVecEnv([make_env for _ in range(N_ENVS)])
    
    # SAC : Meilleur pour actions continues que PPO
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        device="cuda",
        learning_rate=3e-4,
        buffer_size=100_000,  # Replay buffer
        batch_size=256,
        tau=0.005,  # Soft update
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[512, 512, 512])
    )
    
    checkpoint = CheckpointCallback(
        save_freq=50_000 // N_ENVS,
        save_path=f"models/checkpoints/{ticker}_sac",
        name_prefix=f"{ticker}_sac"
    )
    
    wandb_callback = WandbCallback(
        model_save_path=f"models/{ticker}_sac",
        verbose=2
    )
    
    print(f"🚀 Entraînement SAC pour {ticker}...")
    model.learn(
        total_timesteps=TIMESTEPS,
        callback=[checkpoint, wandb_callback],
        progress_bar=True
    )
    
    model.save(f"models/{ticker}_sac_final.zip")
    env.close()
    wandb.finish()

if __name__ == "__main__":
    train_continuous("NVDA", "data_cache/NVDA.csv")
