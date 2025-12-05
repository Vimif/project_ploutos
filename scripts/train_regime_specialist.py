#!/usr/bin/env python3
"""
Entraîne un modèle spécialisé par régime
"""

def train_bull_specialist():
    """Modèle optimisé pour marchés haussiers"""
    # Filtrer données bull (2020-2021)
    df_bull = df[(df.index >= '2020-01-01') & (df.index <= '2021-12-31')]
    
    # Assets de croissance
    tickers = ['NVDA', 'TSLA', 'AMD', 'AAPL']
    
    model = PPO('MlpPolicy', env_bull, learning_rate=2e-4)
    model.learn(5_000_000)
    model.save('models/bull_specialist')
    
def train_bear_specialist():
    """Modèle optimisé pour marchés baissiers"""
    # Filtrer données bear (2022)
    df_bear = df[(df.index >= '2022-01-01') & (df.index <= '2022-12-31')]
    
    # Assets défensifs
    tickers = ['SPY', 'QQQ', 'VTI', 'VOO']
    
    model = PPO('MlpPolicy', env_bear, learning_rate=5e-5)
    model.learn(5_000_000)
    model.save('models/bear_specialist')

def train_sideways_specialist():
    """Modèle optimisé pour marchés range-bound"""
    # Filtrer données sideways (2023-2024)
    df_sideways = df[(df.index >= '2023-01-01')]
    
    # Mix équilibré
    tickers = ['SPY', 'NVDA', 'MSFT', 'AAPL', 'GOOGL']
    
    model = PPO('MlpPolicy', env_sideways, learning_rate=1e-4)
    model.learn(7_000_000)
    model.save('models/sideways_specialist')

# Entraîner les 3 modèles
train_bull_specialist()
train_bear_specialist()
train_sideways_specialist()
