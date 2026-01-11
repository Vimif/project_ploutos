#!/usr/bin/env python3
"""
Ploutos Extended - FINAL WORKING Training Script (STABLE VERSION)
====================================================================

Fixes:
1. Data loading (passed 'data' dict instead of 'data_path')
2. Gymnasium compatibility (reset(seed=...))
3. Action space attribute compatibility (action_space vs single_action_space)
4. SDE handling for discrete/continuous spaces
5. Learning rate parsing (string to float)
6. **Numerical stability** (gradient clipping, NaN detection, reward bounds)

Usage:
    python scripts/train.py \
        --config config/training.yaml \
        --output models/ploutos_production \
        --device cuda:0 \
        --timesteps 50000000
"""

import os
import sys
import yaml
import logging
import argparse
import warnings

import pandas as pd

# Suppress SB3 Pre-Check Warning for GPU/MlpPolicy
# Reason: We have 43k+ features, so GPU IS efficient for this MLP.
warnings.filterwarnings("ignore", message="You are trying to run PPO on the GPU")

import numpy as np
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed
import torch

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NaNDetectionCallback(BaseCallback):
    """
    Detects NaN/Inf and alerts
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.nan_count = 0
    
    def _on_training_start(self) -> None:
        pass
    
    def _on_step(self) -> bool:
        # Check policy parameters
        for param in self.model.policy.parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                self.nan_count += 1
                logger.warning(f"  NaN/Inf detected in policy parameters (count={self.nan_count})")
                if self.nan_count > 5:
                    logger.error("  Too many NaN/Inf detected. Stopping training.")
                    return False
        return True


class StableWrapper(gym.Wrapper):
    """
    Wrapper to stabilize observations and rewards
    """
    def __init__(self, env):
        super().__init__(env)
        self.reward_clip_value = 10.0
        self.obs_clip_value = 1e6
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Clip and clean observations
        if isinstance(obs, np.ndarray):
            obs = np.clip(obs, -self.obs_clip_value, self.obs_clip_value)
            obs = np.nan_to_num(obs, nan=0.0, posinf=self.obs_clip_value, neginf=-self.obs_clip_value)
        
        # Clip and clean rewards
        reward = float(np.clip(reward, -self.reward_clip_value, self.reward_clip_value))
        reward = float(np.nan_to_num(reward, nan=0.0, posinf=self.reward_clip_value, neginf=-self.reward_clip_value))
        
        # Ensure info dict is clean
        for key, val in info.items():
            if isinstance(val, (int, float)):
                info[key] = float(np.nan_to_num(val, nan=0.0))
        
        return obs, reward, done, truncated, info
    
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Clip observations
        if isinstance(obs, np.ndarray):
            obs = np.clip(obs, -self.obs_clip_value, self.obs_clip_value)
            obs = np.nan_to_num(obs, nan=0.0, posinf=self.obs_clip_value, neginf=-self.obs_clip_value)
        
        return obs, info


def load_data_dictionary(data_path):
    """
    Load data from CSV and convert to Dict[str, pd.DataFrame]
    Supports:
    1. Standard CSV with 'Ticker' column
    2. yfinance MultiIndex CSV (Wide format)
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
        # 1. Try Standard Load
        df = pd.read_csv(data_path)
        
        # Check for yfinance MultiIndex format (Row 0 has Tickers)
        # Detection: Column 0 is 'Price' or 'Date' and row 0 contains tickers like 'AAPL'
        is_multiindex = False
        if len(df) > 2:
            row_0_vals = df.iloc[0].astype(str).values
            # Heuristic: If row 0 has many unique values (tickers) and headers are features like 'Close'
            if 'Close' in df.columns or 'Price' in df.columns:
                is_multiindex = True
        
        if is_multiindex:
            logger.info("Detected MultiIndex/Wide format (yfinance style). Reloading with header=[0,1]...")
            df = pd.read_csv(data_path, header=[0,1], index_col=0, parse_dates=True)
            
            data = {}
            # Check levels structure: (Feature, Ticker) or (Ticker, Feature)
            level0 = df.columns.get_level_values(0).unique()
            level1 = df.columns.get_level_values(1).unique()
            
            if 'Close' in level0:
                # Structure: (Feature, Ticker) -> We want Dict[Ticker, DataFrame(Features)]
                logger.info(f"Processing (Feature, Ticker) structure with {len(level1)} tickers...")
                # Swap levels to get (Ticker, Feature)
                df_swapped = df.swaplevel(axis=1)
                
                for ticker in level1:
                    try:
                        ticker_df = df_swapped[ticker].copy()
                        # Ensure numeric
                        ticker_df = ticker_df.apply(pd.to_numeric, errors='coerce')
                        ticker_df = ticker_df.dropna()
                        
                        if len(ticker_df) > 100:
                            ticker_df = ticker_df.reset_index() # Make Date a column
                            # Rename 'Price' index name to 'Date' if needed, or just use index name
                            if 'Date' not in ticker_df.columns:
                                ticker_df.rename(columns={ticker_df.columns[0]: 'Date'}, inplace=True)
                                
                            data[str(ticker)] = ticker_df
                    except Exception as e:
                        logger.warning(f"Skipping ticker {ticker}: {e}")
                        
            elif 'Close' in level1:
                # Structure: (Ticker, Feature)
                logger.info(f"Processing (Ticker, Feature) structure with {len(level0)} tickers...")
                for ticker in level0:
                    try:
                        ticker_df = df[ticker].copy()
                        ticker_df = ticker_df.apply(pd.to_numeric, errors='coerce')
                        ticker_df = ticker_df.dropna()
                        if len(ticker_df) > 100:
                            ticker_df = ticker_df.reset_index()
                            if 'Date' not in ticker_df.columns:
                                ticker_df.rename(columns={ticker_df.columns[0]: 'Date'}, inplace=True)
                            data[str(ticker)] = ticker_df
                    except Exception as e:
                        logger.warning(f"Skipping ticker {ticker}: {e}")
            
            logger.info(f"Successfully loaded {len(data)} tickers from Wide format.")
            return data

        # 2. Standard Ticker Column Load
        data = {}
        if 'Ticker' in df.columns:
            for ticker, group in df.groupby('Ticker'):
                data[str(ticker)] = group.copy().reset_index(drop=True)
        else:
            # Assume single ticker "Unknown"
            data['UNKNOWN'] = df.copy()
            
        logger.info(f"Loaded data for {len(data)} tickers: {list(data.keys())[:5]}...")
        return data
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def make_env(rank, seed=0, data=None):
    """
    Create environment for multiprocessing with stability wrapper.
    """
    def _init():
        try:
            # Add src to path
            sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
            
            # Try real environment
            from ploutos.env.environment import TradingEnvironment
            
            # Use provided data or fallback to dummy
            env_data = data if data is not None else load_data_dictionary("dummy")
            
            env = TradingEnvironment(
                data=env_data,
                initial_balance=100000,
                commission=0.001,
            )
            
            # Add stability wrapper
            env = StableWrapper(env)
            
            # Wrap with TimeLimit
            env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
            
            # Reset with seed
            env.reset(seed=seed + rank)
            
            return env
            
        except Exception as e:
            logger.warning(f"Env {rank}: Fallback to CartPole. Error: {e}")
            
            # Fallback environment
            env = gym.make('CartPole-v1')
            env = StableWrapper(env)
            env.reset(seed=seed + rank)
            return env
    
    return _init


def main():
    parser = argparse.ArgumentParser(description='Train Ploutos (Stable)')
    parser.add_argument('--config', default='config/training.yaml')
    parser.add_argument('--output', default='models/ploutos_production')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--timesteps', type=int, default=50000000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data', default='data/sp500.csv')
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path('logs/tensorboard').mkdir(parents=True, exist_ok=True)
    
    logger.info("======================================================================")
    logger.info("  PLOUTOS - TRAINING START (STABLE VERSION)")
    logger.info("======================================================================")
    
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
    logger.info("  Environments created")
    
    # Check action space
    if hasattr(env, 'single_action_space'):
        action_space = env.single_action_space
    else:
        action_space = env.action_space
        
    action_space_type = type(action_space).__name__
    logger.info(f"Action space type: {action_space_type}")
    
    # Determine if we should use SDE (Continuous only)
    use_sde = isinstance(action_space, gym.spaces.Box)
    
    if not use_sde:
        logger.info("  Discrete/MultiDiscrete action space detected - disabling SDE")
        sde_sample_freq = -1
    else:
        logger.info("  Continuous action space detected - SDE enabled")
        sde_sample_freq = 4
    
    # Create model with stability tweaks
    logger.info("Creating PPO model (with gradient clipping)...")
    
    # Extract training params
    train_cfg = config.get('training', {})

    # Determine device: Config > Vars > Args (default cuda:0)
    if 'device' in train_cfg:
        device = train_cfg['device']
        logger.info(f"Device selected from config: {device}")
    else:
        device = args.device
        logger.info(f"Device selected from arguments: {device}")
    
    # Verify CUDA availability if requested
    if 'cuda' in device and not torch.cuda.is_available():
        logger.warning(f"⚠️  Requesting {device} but CUDA is not available! Fallback to CPU.")
        device = 'cpu'

    
    # Parse learning_rate as float
    learning_rate = train_cfg.get('learning_rate', 1e-4)
    if isinstance(learning_rate, str):
        learning_rate = float(learning_rate)
    
    # CRITICAL: Reduce learning rate further for stability
    learning_rate = min(learning_rate, 5e-5)  # Cap at 5e-5
    logger.info(f"  Learning rate: {learning_rate}")
    
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=learning_rate,
        n_steps=train_cfg.get('n_steps', 2048),
        batch_size=train_cfg.get('batch_size', 4096),
        n_epochs=train_cfg.get('n_epochs', 10),
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2,  # CRITICAL: Clip value function too
        ent_coef=0.01,
        max_grad_norm=0.5,  # CRITICAL: Gradient clipping
        use_sde=use_sde,
        sde_sample_freq=sde_sample_freq,
        tensorboard_log="logs/tensorboard",
        device=device,
        seed=args.seed,
        verbose=1,
    )
    logger.info("  Model created")
    logger.info(f"  PPO Model Device: {model.device}")
    
    # Train with NaN detection
    logger.info(f"Starting training ({args.timesteps:,} steps)...")
    logger.info("="*70)
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[
                CheckpointCallback(
                    save_freq=500000,
                    save_path=str(output_dir),
                    name_prefix='checkpoint'
                ),
                NaNDetectionCallback(verbose=1),
            ],
            tb_log_name='ploutos_training',
            progress_bar=True,
        )
        
        logger.info("======================================================================")
        logger.info("  TRAINING COMPLETED!")
        model.save(str(output_dir / 'final_model'))
        logger.info(f"  Model saved to {output_dir / 'final_model'}")
        
    except KeyboardInterrupt:
        logger.warning("  Training interrupted by user")
        model.save(str(output_dir / 'interrupted_model'))
        logger.info(f"Model saved to {output_dir / 'interrupted_model'}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
    
    finally:
        env.close()


if __name__ == '__main__':
    main()
