#!/usr/bin/env python3
"""
Ploutos V7 - Ensemble Prediction System
Combines 3 expert models to make robust trading decisions.

The Council of Experts:
1. Momentum Expert (Trend following)
2. Mean Reversion Expert (Contrarian)
3. Volatility Expert (Regime filter)

Usage:
    python scripts/v7_ensemble_predict.py --ticker AAPL
"""

import os
import sys
import logging
import argparse
import json
import pickle
import torch
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Import model architectures (must match training scripts)
# We redefine them here to keep prediction script standalone
import torch.nn as nn

class RobustMomentumClassifier(nn.Module):
    def __init__(self, input_dim=28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 2),
        )
    def forward(self, x): return self.net(x)

class ReversionModel(nn.Module):
    def __init__(self, input_dim=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.Tanh(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 2)
        )
    def forward(self, x): return self.net(x)

class VolatilityModel(nn.Module):
    def __init__(self, input_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2)
        )
    def forward(self, x): return self.net(x)

# Feature Extractors
from scripts.train_v7_multiticker import MultiTickerFeatureExtractor
from scripts.train_v7_mean_reversion import MeanReversionFeatureExtractor
from scripts.train_v7_volatility import VolatilityFeatureExtractor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class EnsemblePredictor:
    def __init__(self, models_dir='models'):
        self.models_dir = Path(models_dir)
        self.device = torch.device('cpu') # Inference on CPU is fine
        
        # Load Momentum Model
        self.mom_meta = json.load(open(self.models_dir / 'v7_multiticker/metadata.json'))
        self.mom_scaler = pickle.load(open(self.models_dir / 'v7_multiticker/scaler.pkl', 'rb'))
        self.mom_model = RobustMomentumClassifier(self.mom_meta['input_dim'])
        self.mom_model.load_state_dict(torch.load(self.models_dir / 'v7_multiticker/best_model.pth', map_location='cpu'))
        self.mom_model.eval()
        
        # Load Mean Reversion Model
        self.rev_meta = json.load(open(self.models_dir / 'v7_mean_reversion/metadata.json'))
        self.rev_scaler = pickle.load(open(self.models_dir / 'v7_mean_reversion/scaler.pkl', 'rb'))
        self.rev_model = ReversionModel(len(self.rev_meta['features']))
        self.rev_model.load_state_dict(torch.load(self.models_dir / 'v7_mean_reversion/best_model.pth', map_location='cpu'))
        self.rev_model.eval()
        
        # Load Volatility Model
        self.vol_meta = json.load(open(self.models_dir / 'v7_volatility/metadata.json'))
        self.vol_scaler = pickle.load(open(self.models_dir / 'v7_volatility/scaler.pkl', 'rb'))
        self.vol_model = VolatilityModel(len(self.vol_meta['features']))
        self.vol_model.load_state_dict(torch.load(self.models_dir / 'v7_volatility/best_model.pth', map_location='cpu'))
        self.vol_model.eval()
        
    def predict(self, ticker):
        logger.info(f"🔍 Analyzing {ticker}...")
        
        # Fetch live data (last 100 days)
        df = yf.download(ticker, period="1y", progress=False, auto_adjust=False)
        if df.empty:
            logger.error(f"❌ No data for {ticker}")
            return None
            
        df = df.reset_index()
        # Clean columns
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'][:len(df.columns)]
        df['Ticker'] = ticker
        
        # 1. Momentum Prediction
        mom_ext = MultiTickerFeatureExtractor()
        X_mom, _, _ = mom_ext.extract_features(df)
        X_mom_scaled = self.mom_scaler.transform(X_mom[-1:]) # Last row only
        with torch.no_grad():
            logits = self.mom_model(torch.FloatTensor(X_mom_scaled))
            probs = torch.softmax(logits, 1).numpy()[0]
            mom_pred = "UP" if probs[1] > 0.5 else "DOWN"
            mom_conf = probs[1] if mom_pred == "UP" else probs[0]
            
        # 2. Reversion Prediction
        rev_ext = MeanReversionFeatureExtractor()
        X_rev, _, _ = rev_ext.extract_features(df)
        X_rev_scaled = self.rev_scaler.transform(X_rev[-1:])
        with torch.no_grad():
            logits = self.rev_model(torch.FloatTensor(X_rev_scaled))
            probs = torch.softmax(logits, 1).numpy()[0]
            # Target was (shift(-1) > close), so 1=UP, 0=DOWN
            rev_pred = "UP" if probs[1] > 0.5 else "DOWN"
            rev_conf = probs[1] if rev_pred == "UP" else probs[0]

        # 3. Volatility Prediction
        vol_ext = VolatilityFeatureExtractor()
        X_vol, _, _ = vol_ext.extract_features(df)
        X_vol_scaled = self.vol_scaler.transform(X_vol[-1:])
        with torch.no_grad():
            logits = self.vol_model(torch.FloatTensor(X_vol_scaled))
            probs = torch.softmax(logits, 1).numpy()[0]
            vol_pred = "HIGH_VOL" if probs[1] > 0.5 else "LOW_VOL"
            vol_conf = probs[1] if vol_pred == "HIGH_VOL" else probs[0]
            
        # === ENSEMBLE LOGIC ===
        score = 0
        if mom_pred == "UP": score += 1
        else: score -= 1
        
        if rev_pred == "UP": score += 1
        else: score -= 1
        
        final_decision = "NEUTRAL"
        if score == 2: final_decision = "BUY"
        elif score == -2: final_decision = "SELL"
        elif score == 0: final_decision = "HOLD"
        
        # Filter by volatility
        if vol_pred == "LOW_VOL" and final_decision != "HOLD":
            final_decision += " (Weak)"
        elif vol_pred == "HIGH_VOL" and final_decision != "HOLD":
            final_decision = "STRONG " + final_decision
            
        print("\n" + "="*50)
        print(f"🤖 PLOUTOS V7 ENSEMBLE REPORT: {ticker}")
        print("="*50)
        print(f"1. Momentum Expert:      {mom_pred} ({mom_conf*100:.1f}%)")
        print(f"2. Reversion Expert:     {rev_pred} ({rev_conf*100:.1f}%)")
        print(f"3. Volatility Expert:    {vol_pred} ({vol_conf*100:.1f}%)")
        print("-" * 30)
        print(f"📢 FINAL DECISION:       {final_decision}")
        print("="*50 + "\n")
        
        return {
            'ticker': ticker,
            'decision': final_decision,
            'details': {
                'momentum': {'pred': mom_pred, 'conf': float(mom_conf)},
                'reversion': {'pred': rev_pred, 'conf': float(rev_conf)},
                'volatility': {'pred': vol_pred, 'conf': float(vol_conf)}
            }
        }

if __name__ == '__main__':
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, required=True)
    args = parser.parse_args()
    
    predictor = EnsemblePredictor()
    predictor.predict(args.ticker)
