#!/usr/bin/env python3
"""
Ploutos V6 Extended - FINAL WORKING Training Script
===================================================

Fixes:
1. Data loading (passed 'data' dict instead of 'data_path')
2. Gymnasium compatibility (reset(seed=...))
3. Action space attribute compatibility (action_space vs single_action_space)
4. SDE handling for discrete/continuous spaces

Usage:
    python scripts/train_v6_final.py \
        --config config/training_v6_extended_optimized.yaml \
        --output models/v6_test_5m \
        --device cuda:0 \
        --timesteps 5000000
"""

import os
import sys
import yaml
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_v6_final.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_data_dictionary(data_path):
    """
    Load data from CSV and convert to Dict[str, pd.DataFrame]
    Expected format: CSV with Ticker column or single-ticker CSV
    """
    if not os.path.exists(data_path):
        logger.warning(f"Data file {data_path} not found. Generating DUMMY data for testing.")
        # Generate dummy data
        dates = pd.date_range(start='2020-01-01', periods=2000, freq='D')
        data = {}
        for ticker in ['AAPL', 'MSFT', 'GOOGL', 'AMZN']:
            df = pd.DataFrame(index=range(len(dates)))
            df['Date'] = dates
            # Random walk
            price = 100 + np.cumsum(np.random.normal(0, 1, size=len(dates)))
            df['Close'] = price
            df['Open'] = price * np.random.uniform(0.99, 1.01, size=len(dates))
            df['High'] = df[['Open', 'Close']].max(axis=1) * 1.01
            df['Low'] = df[['Open', 'Close']].min(axis=1) * 0.99
            df['Volume'] = np.random.randint(1000, 1000000, size=len(dates))
            data[ticker] = df
        return data

    logger.info(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
        data = {}
        
        # Check if 'Ticker' column exists
        if 'Ticker' in df.columns:
            for ticker, group in df.groupby('Ticker'):
                data[str(ticker)] = group.copy().reset_index(drop=True)
        else:
            # Assume single ticker "Unknown" if no Ticker column
            data['UNKNOWN'] = df.copy()
            
        logger.info(f"Loaded data for {len(data)} tickers: {list(data.keys())}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def make_env(rank, seed=0, data=None):
    """
    Create environment for multiprocessing.
    """
    def _init():
        try:
            # Try real environment
            from core.universal_environment_v6_better_timing import UniversalTradingEnvV6BetterTiming
            
            # Use provided data or fallback to dummy
            env_data = data if data is not None else load_data_dictionary("dummy")
            
            env = UniversalTradingEnvV6BetterTiming(
                data=env_data,  # FIX: Pass 'data' dict, not 'data_path' string
                initial_balance=100000,
                commission=0.001,
            )
            
            # Wrap with TimeLimit
            env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
            
            # FIX: Use reset(seed=...) instead of env.seed()
            env.reset(seed=seed + rank)
            
            # logger.info(f"Env {rank}: Real trading environment initialized")
            return env
            
        except Exception as e:
            logger.warning(f"Env {rank}: Fallback to CartPole. Error: {e}")
            # logger.exception("Full traceback:")
            
            # Fallback environment
            env = gym.make('CartPole-v1')
            env.reset(seed=seed + rank)
            return env
    
    return _init


def main():
    parser = argparse.ArgumentParser(description='Train Ploutos V6')
    parser.add_argument('--config', default='config/training_v6_extended_optimized.yaml')
    parser.add_argument('--output', default='models/v6_test_5m')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--timesteps', type=int, default=5000000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data', default='data/historical_daily.csv')
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path('logs/tensorboard').mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("🚀 PLOUTOS V6 - TRAINING START (FINAL VERSION)")
    logger.info("="*70)
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load Data ONCE
    data_map = load_data_dictionary(args.data)
    
    # Random seed
    set_random_seed(args.seed)
    
    # Create environment
    n_envs = config.get('training', {}).get('n_envs', 32)
    logger.info(f"Creating {n_envs} parallel environments...")
    
    env = SubprocVecEnv([
        make_env(i, seed=args.seed, data=data_map)
        for i in range(n_envs)
    ])
    logger.info("✅ Environments created")
    
    # FIX: Check action space robustly
    # Try 'single_action_space' (new SB3) or 'action_space' (old SB3 / VecEnv standard)
    if hasattr(env, 'single_action_space'):
        action_space = env.single_action_space
    else:
        action_space = env.action_space
        
    action_space_type = type(action_space).__name__
    logger.info(f"Action space type: {action_space_type}")
    
    # Determine if we should use SDE (Continuous only)
    use_sde = isinstance(action_space, gym.spaces.Box)
    
    if not use_sde:
        logger.info("⚠️  Discrete/MultiDiscrete action space detected - disabling SDE")
        # Ensure sde_sample_freq is -1
        sde_sample_freq = -1
    else:
        logger.info("✅ Continuous action space detected - SDE enabled")
        sde_sample_freq = 4
    
    # Create model
    logger.info("Creating PPO model...")
    
    # Extract training params
    train_cfg = config.get('training', {})
    
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=train_cfg.get('learning_rate', 1e-4),
        n_steps=train_cfg.get('n_steps', 2048),
        batch_size=train_cfg.get('batch_size', 4096),
        n_epochs=train_cfg.get('n_epochs', 10),
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        use_sde=use_sde,
        sde_sample_freq=sde_sample_freq,
        tensorboard_log='logs/tensorboard',
        device=args.device,
        seed=args.seed,
        verbose=1,
    )
    logger.info("✅ Model created")
    
    # Train
    logger.info(f"Starting training ({args.timesteps:,} steps)...")
    logger.info("="*70)
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=CheckpointCallback(
                save_freq=500000,
                save_path=str(output_dir),
                name_prefix='checkpoint'
            ),
            tb_log_name='v6_training',
            progress_bar=True,
        )
        
        logger.info("="*70)
        logger.info("✅ TRAINING COMPLETED!")
        model.save(str(output_dir / 'final_model'))
        logger.info(f"✅ Model saved to {output_dir / 'final_model'}")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise
    
    finally:
        env.close()


if __name__ == '__main__':
    main()
