#!/usr/bin/env python3
"""
Ploutos V7 - Momentum Predictor
Binary Classification: Predict if price goes UP or DOWN tomorrow

Usage:
    python scripts/train_v7_momentum.py --data data/historical_daily.csv --output models/v7_momentum --epochs 100
"""

import os
import sys
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
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

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
    """Extract 30 technical features for momentum prediction"""
    
    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices):
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram
    
    def extract_features(self, df):
        """Extract 30 features from OHLCV data"""
        df = df.copy()
        
        # Price features (6)
        df['returns'] = df['Close'].pct_change()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()
        df['price_position'] = (df['Close'] - df['sma_20']) / (df['sma_20'] + 1e-6)
        df['high_low_ratio'] = (df['High'] - df['Low']) / df['Close']
        df['close_open_ratio'] = (df['Close'] - df['Open']) / (df['Open'] + 1e-6)
        
        # Momentum features (9)
        df['rsi_14'] = self.calculate_rsi(df['Close'], 14)
        df['rsi_7'] = self.calculate_rsi(df['Close'], 7)
        macd, signal, hist = self.calculate_macd(df['Close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_histogram'] = hist
        df['momentum_10'] = df['Close'] - df['Close'].shift(10)
        df['rate_of_change'] = (df['Close'] - df['Close'].shift(1)) / (df['Close'].shift(1) + 1e-6)
        df['stoch_k'] = ((df['Close'] - df['Low'].rolling(14).min()) / (df['High'].rolling(14).max() - df['Low'].rolling(14).min() + 1e-6)) * 100
        
        # Volatility features (6)
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['atr'] = (df['High'] - df['Low']).rolling(14).mean()
        df['atr_ratio'] = df['atr'] / (df['Close'] + 1e-6)
        upper = df['Close'].rolling(20).mean() + df['Close'].rolling(20).std() * 2
        lower = df['Close'].rolling(20).mean() - df['Close'].rolling(20).std() * 2
        df['bb_position'] = (df['Close'] - lower) / (upper - lower + 1e-6)
        df['bb_width'] = (upper - lower) / (df['Close'] + 1e-6)
        
        # Volume features (5)
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / (df['volume_sma'] + 1e-6)
        df['price_volume_trend'] = df['Close'].pct_change() * df['Volume']
        df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        df['obv_sma'] = df['obv'].rolling(20).mean()
        
        # Trend features (4)
        df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['ema_ratio'] = df['ema_12'] / (df['ema_26'] + 1e-6)
        df['trend_strength'] = (df['Close'] - df['Close'].rolling(20).min()) / (df['Close'].rolling(20).max() - df['Close'].rolling(20).min() + 1e-6)
        
        feature_cols = [
            'returns', 'sma_20', 'sma_50', 'price_position', 'high_low_ratio', 'close_open_ratio',
            'rsi_14', 'rsi_7', 'macd', 'macd_signal', 'macd_histogram', 'momentum_10', 'rate_of_change', 'stoch_k',
            'volatility_20', 'atr', 'atr_ratio', 'bb_position', 'bb_width',
            'volume_sma', 'volume_ratio', 'price_volume_trend', 'obv', 'obv_sma',
            'ema_12', 'ema_26', 'ema_ratio', 'trend_strength'
        ]
        
        # Target: 1 if price goes UP tomorrow, 0 if DOWN
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        df = df.dropna()
        
        X = df[feature_cols].values
        y = df['target'].values
        
        logger.info(f"✅ Features extracted: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"📊 UP: {np.sum(y)} | DOWN: {len(y) - np.sum(y)}")
        
        return X, y, feature_cols


class MomentumClassifier(nn.Module):
    """Simple neural network for binary classification"""
    
    def __init__(self, input_dim=27):
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
            nn.Linear(32, 2),  # 2 classes: DOWN or UP
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
        'precision': precision_score(all_targets, all_preds, zero_division=0),
        'recall': recall_score(all_targets, all_preds, zero_division=0),
        'f1': f1_score(all_targets, all_preds, zero_division=0),
        'auc': auc,
        'predictions': np.array(all_preds),
        'targets': np.array(all_targets),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/historical_daily.csv')
    parser.add_argument('--output', default='models/v7_momentum')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_split', type=float, default=0.2)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("🚀 PLOUTOS V7 - MOMENTUM PREDICTOR")
    logger.info("="*70)
    logger.info(f"Device: {device}\n")
    
    # Load data
    logger.info("📊 Loading data...")
    df = pd.read_csv(args.data)
    if 'Ticker' in df.columns:
        ticker = df['Ticker'].iloc[0]
        df = df[df['Ticker'] == ticker].copy()
        logger.info(f"Ticker: {ticker}")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index(drop=True)
    
    # Extract features
    logger.info("\n📈 Extracting features...")
    extractor = MomentumFeatureExtractor()
    X, y, feature_cols = extractor.extract_features(df)
    
    # Normalize
    logger.info("\n🔧 Normalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler
    with open(output_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"✅ Scaler saved")
    
    # Split
    logger.info(f"\n📋 Splitting data (80/20)...")
    split_idx = int(len(X) * (1 - args.test_split))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
    
    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
        batch_size=args.batch_size
    )
    
    # Create model
    logger.info(f"\n🧠 Creating model...")
    model = MomentumClassifier(input_dim=X_scaled.shape[1]).to(device)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training
    logger.info(f"\n⚙️  Training...\n")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    
    best_f1 = 0
    patience_counter = 0
    best_metrics = None
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_metrics = evaluate(model, test_loader, criterion, device)
        
        scheduler.step(test_metrics['loss'])
        
        if test_metrics['f1'] > best_f1:
            best_f1 = test_metrics['f1']
            patience_counter = 0
            best_metrics = test_metrics
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f} Acc={train_acc:.3f} | "
                       f"Test Acc={test_metrics['accuracy']:.3f} F1={test_metrics['f1']:.3f}")
        
        if patience_counter > 20:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model and final evaluation
    logger.info("\n" + "="*70)
    logger.info("📊 FINAL RESULTS")
    logger.info("="*70)
    
    model.load_state_dict(torch.load(output_dir / 'best_model.pth'))
    final_metrics = evaluate(model, test_loader, criterion, device)
    
    logger.info(f"\nAccuracy:  {final_metrics['accuracy']:.3f}")
    logger.info(f"Precision: {final_metrics['precision']:.3f}")
    logger.info(f"Recall:    {final_metrics['recall']:.3f}")
    logger.info(f"F1-Score:  {final_metrics['f1']:.3f}")
    logger.info(f"AUC-ROC:   {final_metrics['auc']:.3f}")
    
    cm = confusion_matrix(final_metrics['targets'], final_metrics['predictions'])
    logger.info(f"\nConfusion Matrix:\n{cm}")
    
    # Save metadata
    metadata = {
        'model_type': 'MomentumClassifier',
        'input_dim': X_scaled.shape[1],
        'feature_columns': feature_cols,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'metrics': {
            'accuracy': float(final_metrics['accuracy']),
            'precision': float(final_metrics['precision']),
            'recall': float(final_metrics['recall']),
            'f1': float(final_metrics['f1']),
            'auc': float(final_metrics['auc']),
        },
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save model
    torch.save(model.state_dict(), output_dir / 'final_model.pth')
    
    logger.info(f"\n✅ Model saved to {output_dir}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
