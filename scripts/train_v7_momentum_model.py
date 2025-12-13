#!/usr/bin/env python3
"""
Ploutos V7 - Momentum Predictor Model
=====================================

Ce script entraîne un modèle de CLASSIFICATION qui prédit si le prix va monter ou baisser
dans les 24h prochaines.

Contrairement au V6 (Reinforcement Learning complexe), c'est simplement :
  INPUT:  Features techniques (RSI, MACD, Volume, etc.) d'AUJOURD'HUI
  OUTPUT: Prédiction binaire (Haut (1) ou Bas (0)) pour DEMAIN
  LOSS:   Cross-Entropy (classification standard)

C'est plus simple, plus robuste, et plus facile à valider.

Usage:
    python scripts/train_v7_momentum_model.py \
        --data data/historical_daily.csv \
        --output models/v7_momentum \
        --epochs 100
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import json

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_v7_momentum.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MomentumFeatureExtractor:
    """
    Extrait les features techniques pour la prédiction de momentum court-terme.
    
    INPUT: DataFrame avec OHLCV (Open, High, Low, Close, Volume)
    OUTPUT: 30 features pour le modèle
    """
    
    def __init__(self, lookback=20):
        self.lookback = lookback
    
    def calculate_rsi(self, prices, period=14):
        """Relative Strength Index: 0-100 (>70 overbought, <30 oversold)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices):
        """MACD: Momentum indicator (prix momentum)"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram
    
    def extract_features(self, df):
        """
        Extrait 30 features par ligne (timestamp).
        
        Args:
            df: DataFrame avec colonnes ['Open', 'High', 'Low', 'Close', 'Volume']
        
        Returns:
            feature_matrix: Shape (n_rows, 30)
            labels: Shape (n_rows,) - 1 if prix_demain > prix_aujourd, else 0
        """
        df = df.copy()
        
        # 1. PRICE-BASED FEATURES (6)
        df['returns'] = df['Close'].pct_change()
        df['price_sma_20'] = df['Close'].rolling(20).mean()
        df['price_sma_50'] = df['Close'].rolling(50).mean()
        df['price_position'] = (df['Close'] - df['price_sma_20']) / (df['price_sma_20'] + 1e-6)
        df['high_low_ratio'] = (df['High'] - df['Low']) / df['Close']
        df['close_open_ratio'] = (df['Close'] - df['Open']) / (df['Open'] + 1e-6)
        
        # 2. MOMENTUM INDICATORS (9)
        df['rsi_14'] = self.calculate_rsi(df['Close'], 14)
        df['rsi_7'] = self.calculate_rsi(df['Close'], 7)
        macd, signal, hist = self.calculate_macd(df['Close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_histogram'] = hist
        df['momentum_10'] = df['Close'] - df['Close'].shift(10)
        df['rate_of_change'] = (df['Close'] - df['Close'].shift(1)) / (df['Close'].shift(1) + 1e-6)
        df['stoch_k'] = ((df['Close'] - df['Low'].rolling(14).min()) / 
                        (df['High'].rolling(14).max() - df['Low'].rolling(14).min() + 1e-6)) * 100
        
        # 3. VOLATILITY INDICATORS (6)
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['atr'] = (df['High'] - df['Low']).rolling(14).mean()
        df['atr_ratio'] = df['atr'] / (df['Close'] + 1e-6)
        upper = df['Close'].rolling(20).mean() + df['Close'].rolling(20).std() * 2
        lower = df['Close'].rolling(20).mean() - df['Close'].rolling(20).std() * 2
        df['bb_position'] = (df['Close'] - lower) / (upper - lower + 1e-6)
        df['bb_width'] = (upper - lower) / (df['Close'] + 1e-6)
        
        # 4. VOLUME-BASED FEATURES (5)
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / (df['volume_sma'] + 1e-6)
        df['price_volume_trend'] = df['Close'].pct_change() * df['Volume']
        df['on_balance_volume'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        df['obv_sma'] = df['on_balance_volume'].rolling(20).mean()
        
        # 5. TREND INDICATORS (4)
        df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['ema_ratio'] = df['ema_12'] / (df['ema_26'] + 1e-6)
        df['trend_strength'] = (df['Close'] - df['Close'].rolling(20).min()) / \
                              (df['Close'].rolling(20).max() - df['Close'].rolling(20).min() + 1e-6)
        
        feature_cols = [
            'returns', 'price_sma_20', 'price_sma_50', 'price_position', 'high_low_ratio',
            'close_open_ratio', 'rsi_14', 'rsi_7', 'macd', 'macd_signal', 'macd_histogram',
            'momentum_10', 'rate_of_change', 'stoch_k', 'volatility_20', 'atr', 'atr_ratio',
            'bb_position', 'bb_width', 'volume_sma', 'volume_ratio', 'price_volume_trend',
            'on_balance_volume', 'obv_sma', 'ema_12', 'ema_26', 'ema_ratio', 'trend_strength'
        ]
        
        # Create target: 1 if prix_demain > prix_aujourd, else 0
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Remove NaN rows
        df = df.dropna()
        
        feature_matrix = df[feature_cols].values
        labels = df['target'].values
        
        logger.info(f"✅ Features extraites: {feature_matrix.shape}")
        logger.info(f"🎯 UP: {np.sum(labels)} | DOWN: {len(labels) - np.sum(labels)}")
        
        return feature_matrix, labels, feature_cols


class MomentumClassifier(nn.Module):
    """
    Petit réseau de neurones pour classification binaire.
    """
    
    def __init__(self, input_dim=30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
        )
    
    def forward(self, x):
        return self.net(x)


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y_batch.cpu().numpy())
    
    return total_loss / len(train_loader), accuracy_score(all_targets, all_preds)


def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_batch.cpu().numpy())
    
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except:
        auc = 0.5
    
    return {
        'loss': total_loss / len(test_loader),
        'accuracy': accuracy_score(all_targets, all_preds),
        'f1': f1_score(all_targets, all_preds, zero_division=0),
        'auc': auc,
        'predictions': all_preds,
        'targets': all_targets,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/historical_daily.csv')
    parser.add_argument('--output', default='models/v7_momentum')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("🚀 PLOUTOS V7 - MOMENTUM PREDICTOR")
    logger.info("="*70)
    
    # Load & extract features
    df = pd.read_csv(args.data)
    if 'Ticker' in df.columns:
        ticker = df['Ticker'].iloc[0]
        df = df[df['Ticker'] == ticker].copy()
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index(drop=True)
    
    extractor = MomentumFeatureExtractor()
    X, y, feature_cols = extractor.extract_features(df)
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    split = int(len(X) * 0.8)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y[:split], y[split:]
    
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
        batch_size=args.batch_size
    )
    
    # Model
    model = MomentumClassifier(X_scaled.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    logger.info(f"Training {sum(p.numel() for p in model.parameters())} parameters...\n")
    
    best_f1 = 0
    patience = 0
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_metrics = evaluate(model, test_loader, criterion, device)
        
        if test_metrics['f1'] > best_f1:
            best_f1 = test_metrics['f1']
            patience = 0
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
        else:
            patience += 1
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}: Train={train_acc:.3f} | "
                       f"Test Acc={test_metrics['accuracy']:.3f} F1={test_metrics['f1']:.3f}")
        
        if patience > 20:
            break
    
    # Final eval
    model.load_state_dict(torch.load(output_dir / 'best_model.pth'))
    final = evaluate(model, test_loader, criterion, device)
    
    logger.info("\n" + "="*70)
    logger.info(f"Final Accuracy: {final['accuracy']:.3f}")
    logger.info(f"Final F1-Score: {final['f1']:.3f}")
    logger.info(f"Final AUC-ROC:  {final['auc']:.3f}")
    logger.info("="*70)
    
    torch.save(model.state_dict(), output_dir / 'final_model.pth')
    logger.info(f"✅ Model saved to {output_dir}")


if __name__ == '__main__':
    main()
