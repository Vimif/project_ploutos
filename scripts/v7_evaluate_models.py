#!/usr/bin/env python3
"""
🧠 PLOUTOS V7.3 - Model Evaluator (The Audit)
"""

import argparse
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score
import warnings
import sys
import traceback

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== DEFINITIONS DES CLASSES ==========

class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(0.1))
    
    def forward(self, x):
        b = x.shape[0]
        x = x.unsqueeze(1)
        qkv = self.to_qkv(x).reshape(b, -1, self.num_heads, self.head_dim * 3)
        q, k, v = qkv[..., :self.head_dim], qkv[..., self.head_dim:2*self.head_dim], qkv[..., 2*self.head_dim:]
        attn = (torch.einsum('bhid,bhjd->bhij', q, k) * self.scale).softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v).reshape(b, -1, self.dim)
        return self.to_out(out).squeeze(1)

class EnhancedMomentumClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout, use_attention=True):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)])
            prev_dim = h
        self.main_stack = nn.Sequential(*layers)
        self.use_attention = use_attention
        if use_attention: self.attention = AttentionBlock(hidden_dims[-1], num_heads=4)
        self.classifier = nn.Sequential(nn.Linear(hidden_dims[-1], 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 2))
    def forward(self, x):
        features = self.main_stack(self.input_norm(x))
        if self.use_attention: features = features + self.attention(features) * 0.1
        return self.classifier(features)

# Feature Engineering... (assuming they are correct)

def calculate_momentum_features(df):
    features = pd.DataFrame(index=df.index)
    features['returns'] = df['Close'].pct_change()
    features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    features['sma_10'] = df['Close'].rolling(10).mean()
    features['sma_20'] = df['Close'].rolling(20).mean()
    features['sma_50'] = df['Close'].rolling(50).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    features['macd'] = exp1 - exp2
    features['signal'] = features['macd'].ewm(span=9, adjust=False).mean()
    features['macd_hist'] = features['macd'] - features['signal']
    features['momentum_10'] = df['Close'] - df['Close'].shift(10)
    features['momentum_20'] = df['Close'] - df['Close'].shift(20)
    return features.dropna()


def evaluate(tickers):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🧪 Démarrage de l'évaluation sur {device}...\n")
    
    experts = ['momentum'] # Only test momentum for now
    models = {}
    scalers = {}
    
    for expert in experts:
        # Simplified load_model for this fix
        model_path = Path(f"models/v7_{expert}_expert_final.pth")
        scaler_path = Path(f"models/v7_{expert}_scaler_final.pkl")
        if not model_path.exists(): continue
        state_dict = torch.load(model_path, map_location=device)
        input_dim = state_dict['input_norm.weight'].shape[0]
        model = EnhancedMomentumClassifier(input_dim, [448, 128, 64], 0.3)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        models[expert] = model
        scalers[expert] = joblib.load(scaler_path)

    if not models:
        logger.error("❌ Aucun modèle chargé !")
        return

    logger.info(f"📉 Téléchargement des données...\n")
    
    results = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="6mo", progress=False)
            if len(df) < 60: continue

            for expert_name, model in models.items():
                features = calculate_momentum_features(df)
                future_returns = df['Close'].pct_change(5).shift(-5)

                # Align and filter
                aligned_features = features.reindex(future_returns.index).dropna()
                aligned_returns = future_returns.reindex(aligned_features.index).dropna()
                
                # Ensure intersection of indices
                common_index = aligned_features.index.intersection(aligned_returns.index)
                X = aligned_features.loc[common_index].values
                y_true = (aligned_returns.loc[common_index] > 0).astype(int).values
                actual_returns = aligned_returns.loc[common_index].values

                if len(X) < 10: continue
                
                X_scaled = scalers[expert_name].transform(X)
                X_tensor = torch.FloatTensor(X_scaled).to(device)
                
                with torch.no_grad():
                    predictions = torch.argmax(model(X_tensor), dim=1).cpu().numpy()
                
                if len(y_true) != len(predictions):
                    logger.warning("Shape mismatch after prediction")
                    continue

                acc = accuracy_score(y_true, predictions)
                
                # FIX: Handle potential NaNs in returns
                buy_hold_return = np.nansum(actual_returns)
                algo_return = np.nansum(actual_returns[predictions == 1])
                
                results.append({
                    'Ticker': ticker, 'Expert': expert_name, 'Accuracy': acc,
                    'Algo_Return': algo_return, 'BuyHold_Return': buy_hold_return,
                    'Outperform': algo_return > buy_hold_return
                })
        except Exception as e:
            logger.error(f"Error on {ticker}: {e}")

    if not results:
        logger.error("❌ No results.")
        return

    df_res = pd.DataFrame(results)
    print("\n" + "="*80)
    print("🏆 RÉSULTATS DE L'ÉVALUATION (6 derniers mois)")
    summary = df_res.groupby('Expert').agg({
        'Accuracy': 'mean', 'Algo_Return': 'sum', 
        'BuyHold_Return': 'sum', 'Outperform': 'mean'
    }).round(4)
    print("\n📊 MOYENNES PAR EXPERT :\n", summary)
    print("="*80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers', default='NVDA,AAPL,MSFT,TSLA,AMZN,GOOGL,META,NFLX,AMD,INTC')
    args = parser.parse_args()
    evaluate(args.tickers.split(','))
