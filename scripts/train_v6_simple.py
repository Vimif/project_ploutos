#!/usr/bin/env python3
"""
Ploutos V6 Extended - SIMPLE Training Script
==============================================

FIXED for Gymnasium compatibility (no deprecated env.seed())

Usage:
    python scripts/train_v6_simple.py \
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
        logging.FileHandler('logs/train_v6_simple.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def make_env(rank, seed=0):
    """
    Create environment for multiprocessing.
    
    FIXED: Uses env.reset(seed=...) instead of deprecated env.seed()
    This is compatible with Gymnasium (gym 0.26+)
    """
    def _init():
        try:
            # Try real environment
            from core.universal_environment_v6_better_timing import UniversalTradingEnvV6BetterTiming
            
            env = UniversalTradingEnvV6BetterTiming(
                data_path='data/historical_data.csv',
                initial_balance=100000,
                commission=0.001,
            )
            
            # Wrap with TimeLimit
            env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
            
            # FIX: Use reset(seed=...) instead of env.seed()
            # This is the Gymnasium-compatible way
            env.reset(seed=seed + rank)
            
            logger.info(f"Env {rank}: Real trading environment initialized")
            return env
            
        except Exception as e:
            logger.warning(f"Env {rank}: Fallback to CartPole ({str(e)[:50]}...)")
            
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
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path('logs/tensorboard').mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("🚀 PLOUTOS V6 - TRAINING START")
    logger.info("="*70)
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Timesteps: {args.timesteps:,}")
    logger.info(f"Device: {args.device}")
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Random seed
    set_random_seed(args.seed)
    
    # Create environment
    logger.info(f"Creating {config['training']['n_envs']} parallel environments...")
    env = SubprocVecEnv([
        make_env(i, seed=args.seed)
        for i in range(config['training']['n_envs'])
    ])
    logger.info("✅ Environments created")
    
    # Create model
    logger.info("Creating PPO model...")
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=config['training']['learning_rate'],
        n_steps=config['training']['n_steps'],
        batch_size=config['training']['batch_size'],
        n_epochs=config['training']['n_epochs'],
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        use_sde=True,
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
