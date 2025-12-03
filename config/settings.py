# config/settings.py
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = DATA_DIR / "logs"
TRADES_DIR = DATA_DIR / "trade_history"

for d in [DATA_DIR, MODELS_DIR, LOGS_DIR, TRADES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

N_FEATURES = 30

TRAINING_CONFIG = {
    'n_envs': 4,
    'total_timesteps': 5_000_000,
    'eval_freq': 50_000,
    'n_eval_episodes': 5,
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'max_grad_norm': 0.5,
}

TRADING_CONFIG = {
    'initial_capital': 100_000,
    'paper_trading': True,
    'check_interval_minutes': 60,
}

WANDB_CONFIG = {
    'project': 'Ploutos_Trading_V50_ULTIMATE',
    'entity': 'vimif-perso',
}

USE_GPU = False
ROLE = "PRODUCTION"