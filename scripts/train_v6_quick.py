#!/usr/bin/env python3
"""
Quick V6 Training — Entraine un modele V6 BetterTiming rapidement.

Usage:
  python scripts/train_v6_quick.py                      # Default: 500K steps
  python scripts/train_v6_quick.py --timesteps 2000000  # 2M steps
  python scripts/train_v6_quick.py --timesteps 15000000 # Full 15M (production)
"""

import sys
import os
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from core.data_fetcher import UniversalDataFetcher
from core.universal_environment_v6_better_timing import UniversalTradingEnvV6BetterTiming

# ============================================================================
# CONFIG
# ============================================================================

TICKERS = [
    'NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'META', 'TSLA',
    'SPY', 'QQQ', 'VOO', 'VTI', 'XLE', 'XLF', 'XLK', 'XLV'
]

ENV_PARAMS = dict(
    initial_balance=100_000,
    commission=0.0,
    sec_fee=0.0000221,
    finra_taf=0.000145,
    max_steps=2500,
    buy_pct=0.20,
    slippage_model='realistic',
    spread_bps=2.0,
    max_position_pct=0.25,
    max_trades_per_day=10,
    min_holding_period=2,
    reward_scaling=1.5,
    use_sharpe_penalty=True,
    use_drawdown_penalty=True,
    reward_trade_success=0.5,
    penalty_overtrading=0.005,
    drawdown_penalty_factor=3.0,
)


# ============================================================================
# CALLBACKS
# ============================================================================

class ProgressCallback(BaseCallback):
    """Affiche la progression de l'entrainement."""
    def __init__(self, total_timesteps, print_freq=10000, verbose=0):
        super().__init__(verbose)
        self.total = total_timesteps
        self.print_freq = print_freq
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self):
        if self.n_calls % self.print_freq == 0:
            elapsed = time.time() - self.start_time if self.start_time else 0
            pct = self.num_timesteps / self.total * 100
            speed = self.num_timesteps / elapsed if elapsed > 0 else 0
            eta = (self.total - self.num_timesteps) / speed if speed > 0 else 0
            print(f"  [{pct:5.1f}%] {self.num_timesteps:>10,d}/{self.total:,d} steps  |  "
                  f"{speed:,.0f} steps/s  |  ETA: {eta/60:.0f}min")
        return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Quick V6 Training')
    parser.add_argument('--timesteps', type=int, default=500_000, help='Total timesteps (default: 500K)')
    parser.add_argument('--n-envs', type=int, default=4, help='Parallel environments (default: 4)')
    parser.add_argument('--output', type=str, default='data/models/brain_v6_quick.zip', help='Output model path')
    parser.add_argument('--days', type=int, default=365, help='Days of training data (default: 365)')
    args = parser.parse_args()

    print("=" * 60)
    print("  PLOUTOS — ENTRAINEMENT V6 BETTER TIMING")
    print("=" * 60)
    print(f"  Timesteps:  {args.timesteps:,d}")
    print(f"  Envs:       {args.n_envs}")
    print(f"  Output:     {args.output}")
    print(f"  Data:       {args.days} jours, {len(TICKERS)} tickers")
    print("=" * 60)

    # ── 1. FETCH DATA ──
    print("\n[1/3] Telechargement donnees...")
    fetcher = UniversalDataFetcher()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days + 100)

    data = {}
    for ticker in TICKERS:
        try:
            df = fetcher.fetch(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), interval='1h')
            if df is not None and len(df) > 100:
                data[ticker] = df
                print(f"  {ticker}: {len(df)} barres")
        except Exception as e:
            print(f"  {ticker}: ERREUR ({e})")

    if len(data) < 5:
        print("ERREUR: Pas assez de donnees")
        sys.exit(1)

    print(f"  -> {len(data)}/{len(TICKERS)} tickers charges")

    # ── 2. CREATE ENVS ──
    print(f"\n[2/3] Creation {args.n_envs} environnements...")

    def make_env(rank):
        def _init():
            env = UniversalTradingEnvV6BetterTiming(data=data, **ENV_PARAMS)
            env.reset(seed=rank)
            return env
        return _init

    if args.n_envs > 1:
        try:
            vec_env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
            print(f"  SubprocVecEnv({args.n_envs}) cree")
        except Exception:
            print("  SubprocVecEnv echoue, fallback DummyVecEnv...")
            vec_env = DummyVecEnv([make_env(i) for i in range(args.n_envs)])
    else:
        vec_env = DummyVecEnv([make_env(0)])
        print(f"  DummyVecEnv(1) cree")

    # Check obs shape
    obs = vec_env.reset()
    print(f"  Observation shape: {obs.shape}")

    # ── 3. TRAIN ──
    print(f"\n[3/3] Entrainement PPO ({args.timesteps:,d} steps)...")

    # Adjusted hyperparams for quick training
    if args.timesteps <= 1_000_000:
        lr = 3e-4
        n_steps = 1024
        batch_size = 256
        n_epochs = 10
        ent_coef = 0.15  # More exploration for quick runs
    else:
        lr = 1e-4
        n_steps = 4096
        batch_size = 2048
        n_epochs = 20
        ent_coef = 0.10

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.02,
        policy_kwargs=dict(
            net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256]),
        ),
        verbose=0,
        device='auto',
    )

    print(f"  Policy: MLP pi=[512,512,256] vf=[512,512,256]")
    print(f"  LR={lr}, batch={batch_size}, epochs={n_epochs}, ent={ent_coef}")
    print(f"  Device: {model.device}")
    print()

    # Callbacks
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_path.parent / f"{output_path.stem}_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ProgressCallback(args.timesteps, print_freq=max(10000, args.timesteps // 20)),
        CheckpointCallback(save_freq=max(50000, args.timesteps // 5), save_path=str(checkpoint_dir), name_prefix="v6"),
    ]

    start = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)
    elapsed = time.time() - start

    # ── SAVE ──
    model.save(str(output_path).replace('.zip', ''))
    print(f"\n{'='*60}")
    print(f"  ENTRAINEMENT TERMINE")
    print(f"{'='*60}")
    print(f"  Duree:      {elapsed/60:.1f} min")
    print(f"  Steps/sec:  {args.timesteps/elapsed:,.0f}")
    print(f"  Modele:     {output_path}")
    print(f"  Obs shape:  {model.observation_space.shape}")
    print(f"{'='*60}")
    print(f"\n  Valider avec:")
    print(f"  python scripts/backtest_ultimate.py --model {output_path} --quick\n")

    vec_env.close()


if __name__ == '__main__':
    main()
