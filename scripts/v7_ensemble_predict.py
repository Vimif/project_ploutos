#!/usr/bin/env python3
"""
Ploutos V7 - Ensemble Prediction System
Combines 3 expert models to make robust trading decisions.

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
from pathlib import Path
from datetime import datetime

import torch.nn as nn

# Model classes
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

# Feature Extractors (simplified versions)
class SimpleFeatureExtractor:
    """Simplified momentum features"""
    def extract_features(self, df):
        df = df.copy()
        df['returns'] = df['Close'].pct_change()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()
        df['price_position'] = (df['Close'] - df['sma_20']) / (df['sma_20'] + 1e-6)
        df['high_low_ratio'] = (df['High'] - df['Low']) / df['Close']
        df['close_open_ratio'] = (df['Close'] - df['Open']) / (df['Open'] + 1e-6)
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-6)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['rsi_7'] = self._rsi(df['Close'], 7)
        
        macd, signal, hist = self._macd(df['Close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_histogram'] = hist
        df['momentum_10'] = df['Close'] - df['Close'].shift(10)
        df['momentum_5'] = df['Close'] - df['Close'].shift(5)
        df['rate_of_change'] = (df['Close'] - df['Close'].shift(1)) / (df['Close'].shift(1) + 1e-6)
        
        low_14 = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        df['stoch_k'] = 100 * (df['Close'] - low_14) / (high_14 - low_14 + 1e-6)
        
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['volatility_5'] = df['returns'].rolling(5).std()
        df['atr'] = (df['High'] - df['Low']).rolling(14).mean()
        df['atr_ratio'] = df['atr'] / (df['Close'] + 1e-6)
        
        upper = df['Close'].rolling(20).mean() + df['Close'].rolling(20).std() * 2
        lower = df['Close'].rolling(20).mean() - df['Close'].rolling(20).std() * 2
        df['bb_position'] = (df['Close'] - lower) / (upper - lower + 1e-6)
        
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / (df['volume_sma'] + 1e-6)
        df['price_volume_trend'] = df['Close'].pct_change() * df['Volume']
        df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        df['obv_momentum'] = df['obv'] - df['obv'].shift(5)
        
        df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['ema_ratio'] = df['ema_12'] / (df['ema_26'] + 1e-6)
        df['trend_strength'] = (df['Close'] - df['Close'].rolling(20).min()) / (df['Close'].rolling(20).max() - df['Close'].rolling(20).min() + 1e-6)
        
        cols = ['returns', 'sma_20', 'sma_50', 'price_position', 'high_low_ratio', 'close_open_ratio',
                'rsi_14', 'rsi_7', 'macd', 'macd_signal', 'macd_histogram', 'momentum_10', 'momentum_5',
                'rate_of_change', 'stoch_k', 'volatility_20', 'volatility_5', 'atr', 'atr_ratio', 'bb_position',
                'volume_sma', 'volume_ratio', 'price_volume_trend', 'obv', 'obv_momentum',
                'ema_12', 'ema_26', 'ema_ratio', 'trend_strength']
        
        df = df.dropna()
        return df[cols].values, cols
    
    def _rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-6)
        return 100 - (100 / (1 + rs))
    
    def _macd(self, prices):
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal, macd - signal

class ReversionFeatureExtractor:
    """Mean reversion features"""
    def extract_features(self, df):
        df = df.copy()
        sma_20 = df['Close'].rolling(20).mean()
        std_20 = df['Close'].rolling(20).std()
        upper = sma_20 + 2 * std_20
        lower = sma_20 - 2 * std_20
        
        df['bb_pct'] = (df['Close'] - lower) / (upper - lower + 1e-6)
        df['bb_width'] = (upper - lower) / (sma_20 + 1e-6)
        df['dist_sma_20'] = (df['Close'] - sma_20) / sma_20
        df['dist_sma_50'] = (df['Close'] - df['Close'].rolling(50).mean()) / df['Close'].rolling(50).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-6)))
        
        low_14 = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        df['stoch_k'] = 100 * (df['Close'] - low_14) / (high_14 - low_14 + 1e-6)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        df['williams_r'] = -100 * (high_14 - df['Close']) / (high_14 - low_14 + 1e-6)
        df['z_score'] = (df['Close'] - sma_20) / (std_20 + 1e-6)
        
        cols = ['bb_pct', 'bb_width', 'dist_sma_20', 'dist_sma_50', 'rsi', 'stoch_k', 'stoch_d', 'williams_r', 'z_score']
        df = df.dropna()
        return df[cols].values, cols

class VolatilityFeatureExtractor:
    """Volatility features"""
    def extract_features(self, df):
        df = df.copy()
        
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        df['atr_pct'] = atr / (df['Close'] + 1e-6)
        
        returns = df['Close'].pct_change()
        df['vol_20'] = returns.rolling(20).std() * np.sqrt(252)
        df['vol_5'] = returns.rolling(5).std() * np.sqrt(252)
        
        high_14 = df['High'].rolling(14).max()
        low_14 = df['Low'].rolling(14).min()
        df['chop_idx'] = 100 * np.log10(tr.rolling(14).sum() / (high_14 - low_14 + 1e-6)) / np.log10(14)
        df['vol_volatility'] = df['Volume'].pct_change().rolling(20).std()
        
        net_change = np.abs(df['Close'] - df['Close'].shift(10))
        total_path = np.abs(df['Close'].diff()).rolling(10).sum()
        df['efficiency'] = net_change / (total_path + 1e-6)
        
        cols = ['atr_pct', 'vol_20', 'vol_5', 'chop_idx', 'vol_volatility', 'efficiency']
        df = df.dropna()
        return df[cols].values, cols

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class EnsemblePredictor:
    def __init__(self, models_dir='models'):
        self.models_dir = Path(models_dir)
        self.device = torch.device('cpu')
        
        logger.info("🔍 Loading 3 Expert Models...")
        
        # Load Momentum
        self.mom_meta = json.load(open(self.models_dir / 'v7_multiticker/metadata.json'))
        self.mom_scaler = pickle.load(open(self.models_dir / 'v7_multiticker/scaler.pkl', 'rb'))
        self.mom_model = RobustMomentumClassifier(self.mom_meta['input_dim'])
        self.mom_model.load_state_dict(torch.load(self.models_dir / 'v7_multiticker/best_model.pth', map_location='cpu'))
        self.mom_model.eval()
        logger.info("✅ Momentum Expert loaded")
        
        # Load Reversion
        self.rev_meta = json.load(open(self.models_dir / 'v7_mean_reversion/metadata.json'))
        self.rev_scaler = pickle.load(open(self.models_dir / 'v7_mean_reversion/scaler.pkl', 'rb'))
        self.rev_model = ReversionModel(len(self.rev_meta['features']))
        self.rev_model.load_state_dict(torch.load(self.models_dir / 'v7_mean_reversion/best_model.pth', map_location='cpu'))
        self.rev_model.eval()
        logger.info("✅ Reversion Expert loaded")
        
        # Load Volatility
        self.vol_meta = json.load(open(self.models_dir / 'v7_volatility/metadata.json'))
        self.vol_scaler = pickle.load(open(self.models_dir / 'v7_volatility/scaler.pkl', 'rb'))
        self.vol_model = VolatilityModel(len(self.vol_meta['features']))
        self.vol_model.load_state_dict(torch.load(self.models_dir / 'v7_volatility/best_model.pth', map_location='cpu'))
        self.vol_model.eval()
        logger.info("✅ Volatility Expert loaded\n")
        
    def predict(self, ticker):
        logger.info(f"🔍 Analyzing {ticker}...")
        
        # Fetch data
        df = yf.download(ticker, period="2y", progress=False)
        if df.empty:
            logger.error(f"❌ No data for {ticker}")
            return None
        df = df.reset_index()
        
        # 1. MOMENTUM
        X_mom, _ = SimpleFeatureExtractor().extract_features(df)
        X_mom_scaled = self.mom_scaler.transform(X_mom[-1:])
        with torch.no_grad():
            logits = self.mom_model(torch.FloatTensor(X_mom_scaled))
            probs = torch.softmax(logits, 1).numpy()[0]
            mom_pred = 1 if probs[1] > 0.5 else 0
            mom_conf = float(probs[1])
        
        # 2. REVERSION
        X_rev, _ = ReversionFeatureExtractor().extract_features(df)
        X_rev_scaled = self.rev_scaler.transform(X_rev[-1:])
        with torch.no_grad():
            logits = self.rev_model(torch.FloatTensor(X_rev_scaled))
            probs = torch.softmax(logits, 1).numpy()[0]
            rev_pred = 1 if probs[1] > 0.5 else 0
            rev_conf = float(probs[1])
        
        # 3. VOLATILITY
        X_vol, _ = VolatilityFeatureExtractor().extract_features(df)
        X_vol_scaled = self.vol_scaler.transform(X_vol[-1:])
        with torch.no_grad():
            logits = self.vol_model(torch.FloatTensor(X_vol_scaled))
            probs = torch.softmax(logits, 1).numpy()[0]
            vol_pred = 1 if probs[1] > 0.5 else 0
            vol_conf = float(probs[1])
        
        # VOTING LOGIC
        score = mom_pred + rev_pred - 1  # -1, 0, or 1
        
        if score >= 1: decision = "BUY"
        elif score <= -1: decision = "SELL"
        else: decision = "HOLD"
        
        # Volatility filter
        if vol_pred == 0:  # LOW volatility
            decision += " (WEAK)"
        else:  # HIGH volatility
            decision = "STRONG " + decision
        
        print("\n" + "="*60)
        print(f"🤖 PLOUTOS V7 ENSEMBLE - {ticker}")
        print("="*60)
        print(f"1️⃣  Momentum Expert:      {'UP' if mom_pred else 'DOWN':<4} ({mom_conf*100:5.1f}%)")
        print(f"2️⃣  Reversion Expert:     {'UP' if rev_pred else 'DOWN':<4} ({rev_conf*100:5.1f}%)")
        print(f"3️⃣  Volatility Expert:    {'HIGH' if vol_pred else 'LOW':<4} ({vol_conf*100:5.1f}%)")
        print("-" * 60)
        print(f"📢 FINAL SIGNAL:          {decision}")
        print("="*60 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', required=True)
    args = parser.parse_args()
    
    predictor = EnsemblePredictor()
    predictor.predict(args.ticker.upper())
