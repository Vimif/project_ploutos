#!/usr/bin/env python3
"""
Ploutos V7 - Expert 2: Mean Reversion Predictor
Specialized in detecting overbought/oversold conditions and reversals.

Philosophy: "What goes up must come down" (and vice versa)
Focus: Bollinger Bands, Stochastic, Distance from SMA, RSI extremes

Usage:
    python scripts/train_v7_mean_reversion.py --data data/multi_ticker_history.csv --output models/v7_mean_reversion
"""

import os
import logging
import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MeanReversionFeatureExtractor:
    """Extract features focused on mean reversion signals"""
    
    def extract_features(self, df):
        features_list = []
        targets_list = []
        
        for ticker in df['Ticker'].unique():
            tdf = df[df['Ticker'] == ticker].copy().reset_index(drop=True)
            if len(tdf) < 50: continue
            
            # === MEAN REVERSION FEATURES ===
            
            # 1. Bollinger Bands Position (The King of Mean Reversion)
            sma_20 = tdf['Close'].rolling(20).mean()
            std_20 = tdf['Close'].rolling(20).std()
            upper = sma_20 + 2 * std_20
            lower = sma_20 - 2 * std_20
            
            # 0 = at lower band, 0.5 = at mid, 1 = at upper band
            # Mean reversion signal: < 0 (oversold) or > 1 (overbought)
            tdf['bb_pct'] = (tdf['Close'] - lower) / (upper - lower + 1e-6)
            tdf['bb_width'] = (upper - lower) / (sma_20 + 1e-6)
            
            # 2. Distance from Moving Averages
            tdf['dist_sma_20'] = (tdf['Close'] - sma_20) / sma_20
            tdf['dist_sma_50'] = (tdf['Close'] - tdf['Close'].rolling(50).mean()) / tdf['Close'].rolling(50).mean()
            
            # 3. Oscillators (RSI & Stochastic)
            delta = tdf['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-6)
            tdf['rsi'] = 100 - (100 / (1 + rs))
            
            # Stochastic
            low_14 = tdf['Low'].rolling(14).min()
            high_14 = tdf['High'].rolling(14).max()
            tdf['stoch_k'] = 100 * (tdf['Close'] - low_14) / (high_14 - low_14 + 1e-6)
            tdf['stoch_d'] = tdf['stoch_k'].rolling(3).mean()
            
            # 4. Williams %R (similar to Stoch but different scale)
            tdf['williams_r'] = -100 * (high_14 - tdf['Close']) / (high_14 - low_14 + 1e-6)
            
            # 5. Z-Score (Statistical deviation)
            tdf['z_score'] = (tdf['Close'] - sma_20) / (std_20 + 1e-6)
            
            # Target: REVERSAL prediction
            # If price is high (Z > 1), we predict DOWN (0)
            # If price is low (Z < -1), we predict UP (1)
            # But the ground truth is simply: Does price go UP tomorrow?
            tdf['target'] = (tdf['Close'].shift(-1) > tdf['Close']).astype(int)
            
            tdf = tdf.dropna()
            
            cols = ['bb_pct', 'bb_width', 'dist_sma_20', 'dist_sma_50', 
                   'rsi', 'stoch_k', 'stoch_d', 'williams_r', 'z_score']
            
            X = tdf[cols].values
            y = tdf['target'].values
            
            features_list.append(X)
            targets_list.append(y)
            
        return np.vstack(features_list), np.hstack(targets_list), cols

class ReversionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),  # Tanh often works better for mean reversion (bounded outputs)
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 2)
        )
    def forward(self, x): return self.net(x)

# Training logic is identical to other scripts, simplified here for brevity
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(args.data)
    X, y, cols = MeanReversionFeatureExtractor().extract_features(df)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    with open(out / 'scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
    
    # Split
    split = int(len(X)*0.8)
    X_tr, X_te = torch.FloatTensor(X_scaled[:split]), torch.FloatTensor(X_scaled[split:])
    y_tr, y_te = torch.LongTensor(y[:split]), torch.LongTensor(y[split:])
    
    # Balance
    class_counts = torch.bincount(y_tr)
    weights = 1. / class_counts.float()
    samples_weights = weights[y_tr]
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))
    
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, sampler=sampler)
    model = ReversionModel(len(cols)).to(device)
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
            
        # Eval
        model.eval()
        with torch.no_grad():
            logits = model(X_te.to(device))
            preds = torch.argmax(logits, 1).cpu().numpy()
            acc = accuracy_score(y_te, preds)
            
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), out / 'best_model.pth')
            
    logger.info(f"✅ Mean Reversion Model Trained. Best Accuracy: {best_acc:.3f}")
    
    # Metadata
    meta = {'features': cols, 'accuracy': best_acc, 'type': 'MeanReversion'}
    with open(out / 'metadata.json', 'w') as f: json.dump(meta, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    train(parser.parse_args())
