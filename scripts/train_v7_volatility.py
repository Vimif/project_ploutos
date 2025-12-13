#!/usr/bin/env python3
"""
Ploutos V7 - Expert 3: Volatility & Trend Predictor
Specialized in detecting strong trend regimes vs choppy markets.

Philosophy: "Don't trade the chop"
Focus: ATR, ADX, Volume anomalies, Historical Volatility

Usage:
    python scripts/train_v7_volatility.py --data data/multi_ticker_history.csv --output models/v7_volatility
"""

import os
import logging
import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VolatilityFeatureExtractor:
    """Extract features focused on market regime (Trend vs Chop)"""
    
    def extract_features(self, df):
        features_list = []
        targets_list = []
        
        for ticker in df['Ticker'].unique():
            tdf = df[df['Ticker'] == ticker].copy().reset_index(drop=True)
            if len(tdf) < 50: continue
            
            # === VOLATILITY FEATURES ===
            
            # 1. ATR (Average True Range)
            high_low = tdf['High'] - tdf['Low']
            high_close = np.abs(tdf['High'] - tdf['Close'].shift())
            low_close = np.abs(tdf['Low'] - tdf['Close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            tdf['atr_pct'] = atr / (tdf['Close'] + 1e-6)
            
            # 2. Historical Volatility
            returns = tdf['Close'].pct_change()
            tdf['vol_20'] = returns.rolling(20).std() * np.sqrt(252) # Annualized
            tdf['vol_5'] = returns.rolling(5).std() * np.sqrt(252)
            
            # 3. Choppiness Index (0=Trend, 100=Chop)
            high_14 = tdf['High'].rolling(14).max()
            low_14 = tdf['Low'].rolling(14).min()
            tdf['chop_idx'] = 100 * np.log10(tr.rolling(14).sum() / (high_14 - low_14 + 1e-6)) / np.log10(14)
            
            # 4. Volume Volatility
            tdf['vol_volatility'] = tdf['Volume'].pct_change().rolling(20).std()
            
            # 5. Trend Efficiency
            # Net change / Total path traveled
            net_change = np.abs(tdf['Close'] - tdf['Close'].shift(10))
            total_path = np.abs(tdf['Close'].diff()).rolling(10).sum()
            tdf['efficiency'] = net_change / (total_path + 1e-6)
            
            # Target: 
            # We want to predict if tomorrow's move is SIGNIFICANT (High Volatility)
            # 1 = Big Move (either up or down), 0 = Small Move
            # Threshold: > 1% move
            tomorrow_ret = np.abs(tdf['Close'].shift(-1).pct_change())
            tdf['target'] = (tomorrow_ret > 0.01).astype(int)
            
            tdf = tdf.dropna()
            
            cols = ['atr_pct', 'vol_20', 'vol_5', 'chop_idx', 'vol_volatility', 'efficiency']
            
            X = tdf[cols].values
            y = tdf['target'].values
            
            features_list.append(X)
            targets_list.append(y)
            
        return np.vstack(features_list), np.hstack(targets_list), cols

class VolatilityModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x): return self.net(x)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(args.data)
    X, y, cols = VolatilityFeatureExtractor().extract_features(df)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    with open(out / 'scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
    
    split = int(len(X)*0.8)
    X_tr, X_te = torch.FloatTensor(X_scaled[:split]), torch.FloatTensor(X_scaled[split:])
    y_tr, y_te = torch.LongTensor(y[:split]), torch.LongTensor(y[split:])
    
    # Weight balancing because big moves are rare
    counts = torch.bincount(y_tr)
    weights = 1. / counts.float()
    sampler = WeightedRandomSampler(weights[y_tr], len(y_tr))
    
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, sampler=sampler)
    model = VolatilityModel(len(cols)).to(device)
    opt = optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    
    best_acc = 0
    for ep in range(100):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
            
        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(X_te.to(device)), 1).cpu().numpy()
            acc = accuracy_score(y_te, preds)
            
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), out / 'best_model.pth')
            
    logger.info(f"✅ Volatility Model Trained. Best Accuracy: {best_acc:.3f}")
    
    meta = {'features': cols, 'accuracy': best_acc, 'type': 'Volatility'}
    with open(out / 'metadata.json', 'w') as f: json.dump(meta, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    train(parser.parse_args())
