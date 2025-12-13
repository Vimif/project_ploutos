#!/usr/bin/env python3
"""
🤌 PLOUTOS V7.1 - Hyperparameter Optimizer

Utilise Optuna pour trouver les meilleurs hyperpar. pour chaque expert.

FIX: Utilise VRAIS indicateurs techniques, pas de données random.

Commande :
    python scripts/v7_hyperparameter_optimizer.py --expert momentum --trials 50
"""

import argparse
import json
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from datetime import datetime
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

# Dynamic import for pandas_ta if available
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    print("⚠️ pandas_ta not found. Some features will be unavailable. pip install pandas_ta")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# [MODELS & LOSS CLASSES - UNCHANGED]...

# ========== FEATURE ENGINEERING ==========

def calculate_features(df):
    """Calcule les indicateurs techniques pour un DataFrame.
       Retourne un DataFrame avec les features normalisées.
    """
    # Basic features
    df['returns'] = df['Close'].pct_change().fillna(0)
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
    
    # Volatility
    df['volatility'] = df['log_returns'].rolling(window=21).std() * np.sqrt(252)
    
    # Momentum
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['rsi'] = ta.rsi(df['Close'], length=14) if PANDAS_TA_AVAILABLE else df['sma_20'] # Fallback
    
    # Volume
    df['volume_sma_20'] = df['Volume'].rolling(20).mean()
    
    # Advanced features with pandas_ta
    if PANDAS_TA_AVAILABLE:
        df.ta.macd(append=True)
        df.ta.bbands(append=True)
        df.ta.adx(append=True)
        df.ta.stoch(append=True)
        df.ta.obv(append=True)
    
    df = df.dropna()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    return df

# ========== OPTIMIZATION OBJECTIVE ==========

class OptunaObjective:
    # ... [INIT UNCHANGED] ...
    def __init__(self, X_train, y_train, X_val, y_val, expert_type='momentum'):
        y_train = np.asarray(y_train).flatten()
        y_val = np.asarray(y_val).flatten()
        assert X_train.shape[0] == y_train.shape[0]
        assert X_val.shape[0] == y_val.shape[0]
        
        self.X_train = torch.FloatTensor(X_train)
        self.y_train = torch.LongTensor(y_train)
        self.X_val = torch.FloatTensor(X_val)
        self.y_val = torch.LongTensor(y_val)
        self.expert_type = expert_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ... [CALL METHOD UNCHANGED] ...

# ========== MAIN OPTIMIZER ==========

def optimize_expert(expert_type, tickers, trials=50, timeout=3600):
    logger.info(f"🔍 Téléchargement et feature engineering pour {len(tickers)} tickers...")
    
    all_features = []
    all_targets = []

    feature_map = {
        'momentum': ['returns', 'log_returns', 'sma_20', 'sma_50', 'rsi', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'ADX_14', 'STOCHk_14_3_3'],
        'reversion': ['BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0', 'rsi', 'STOCHd_14_3_3'],
        'volatility': ['volatility', 'BBP_5_2.0', 'ADX_14']
    }

    target_map = {
        'momentum': lambda df: (df['Close'].shift(-5) > df['Close']).astype(int),
        'reversion': lambda df: (df['Close'].shift(-1) < df['sma_20']).astype(int),
        'volatility': lambda df: (df['volatility'].shift(-5) > df['volatility']).astype(int)
    }
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="2y", progress=False)
            if df.empty:
                continue

            # Calculate ALL features first
            df_features = calculate_features(df.copy())
            if df_features.empty:
                continue
            
            # Extract relevant features and target for the expert
            current_features = [f for f in feature_map[expert_type] if f in df_features.columns]
            if len(current_features) < 3:
                logger.warning(f"⚠️ Not enough features for {ticker} ({expert_type})")
                continue
            
            X = df_features[current_features]
            y = target_map[expert_type](df_features)

            # Align X and y after target shift
            if len(X) > len(y):
                X = X.iloc[:len(y)]

            all_features.append(X)
            all_targets.append(y)

        except Exception as e:
            logger.warning(f"⚠️ Error processing {ticker}: {e}")

    if not all_features:
        logger.error("❌ No data processed!")
        return None

    X = pd.concat(all_features).dropna()
    y = pd.concat(all_targets).dropna()

    # Align final X and y
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index].values
    y = y.loc[common_index].values

    logger.info(f"📊 Raw data shapes: X={X.shape}, y={y.shape}")
    
    # ... [Rest of the script is the same: Normalize, Split, Optimize] ...

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    logger.info(f"✅ Data loaded: X_train={X_train.shape}, y_train={y_train.shape}")
    
    # ... [Optuna study setup & run - UNCHANGED] ...

# ... [MAIN EXECUTION BLOCK - UNCHANGED] ...

