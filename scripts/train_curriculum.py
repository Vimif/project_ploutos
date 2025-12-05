#!/usr/bin/env python3
"""
Entraînement en Curriculum Learning
"""
import sys
sys.path.append('.')

from core.data_fetcher import UniversalDataFetcher
from core.universal_environment import UniversalTradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import wandb

# Config W&B
wandb.init(project="Ploutos_Curriculum", name="V2_Full_Pipeline")

# ========================================
# STAGE 1 : MONO-ASSET (SPY)
# ========================================
print("\n" + "="*80)
print("🎓 STAGE 1 : Apprendre sur SPY uniquement")
print("="*80)

fetcher = UniversalDataFetcher()
df_spy = fetcher.fetch('SPY', interval='1h')

env_spy = UniversalTradingEnv(
    df=df_spy,
    ticker='SPY',
    initial_balance=10000
)

model_stage1 = PPO(
    'MlpPolicy',
    env_spy,
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    verbose=1,
    tensorboard_log='./logs/stage1'
)

model_stage1.learn(
    total_timesteps=3_000_000,
    callback=wandb.keras.WandbCallback()
)

model_stage1.save('models/stage1_spy')
print("✅ Stage 1 terminé : models/stage1_spy.zip")

# ========================================
# STAGE 2 : MULTI-ASSET SIMPLE (ETFs)
# ========================================
print("\n" + "="*80)
print("🎓 STAGE 2 : Généraliser sur SPY + QQQ + IWM")
print("="*80)

data_stage2 = fetcher.bulk_fetch(['SPY', 'QQQ', 'IWM'], interval='1h')

def make_env(ticker):
    def _init():
        return UniversalTradingEnv(
            df=data_stage2[ticker],
            ticker=ticker,
            initial_balance=10000
        )
    return _init

env_stage2 = SubprocVecEnv([make_env(t) for t in ['SPY', 'QQQ', 'IWM']])

# ✅ TRANSFER LEARNING : Charger stage1 comme base
model_stage2 = PPO.load('models/stage1_spy', env=env_stage2)

model_stage2.learn(
    total_timesteps=5_000_000,
    callback=wandb.keras.WandbCallback()
)

model_stage2.save('models/stage2_etfs')
print("✅ Stage 2 terminé : models/stage2_etfs.zip")

# ========================================
# STAGE 3 : ACTIONS COMPLEXES
# ========================================
print("\n" + "="*80)
print("🎓 STAGE 3 : Maîtriser NVDA, MSFT, AAPL, GOOGL, AMZN")
print("="*80)

tickers_stage3 = ['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN']
data_stage3 = fetcher.bulk_fetch(tickers_stage3, interval='1h')

env_stage3 = SubprocVecEnv([make_env(t) for t in tickers_stage3])

model_stage3 = PPO.load('models/stage2_etfs', env=env_stage3)

model_stage3.learn(
    total_timesteps=10_000_000,
    callback=wandb.keras.WandbCallback()
)

model_stage3.save('models/stage3_stocks')
print("✅ Stage 3 terminé : models/stage3_stocks.zip")

# ========================================
# VALIDATION
# ========================================
print("\n" + "="*80)
print("🧪 VALIDATION FINALE")
print("="*80)

from scripts.validate import validate_model

results = validate_model('models/stage3_stocks', data_stage3)

print(f"""
📊 RÉSULTATS FINAUX :
   Sharpe Ratio : {results['sharpe']:.2f}
   Return Total : {results['total_return']:.1f}%
   Max Drawdown : {results['max_dd']:.1f}%
   Win Rate     : {results['win_rate']:.1f}%
""")

if results['sharpe'] > 1.5:
    print("✅ OBJECTIF ATTEINT ! Prêt pour déploiement.")
else:
    print("⚠️ Sharpe insuffisant, réentraînement nécessaire.")

wandb.finish()
