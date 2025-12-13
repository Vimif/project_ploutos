#!/usr/bin/env python3
"""
Ploutos V7 - Multi-Ticker Momentum Predictor
Trains on 15k+ rows from 12 different tickers for robust momentum prediction.

Expected improvements:
- 50-60% accuracy (vs 49% with single ticker)
- Better generalization across sectors
- More robust patterns

Usage:
    python scripts/train_v7_multiticker.py \
        --data data/multi_ticker_history.csv \
        --output models/v7_multiticker \
        --epochs 200
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
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_v7_multiticker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MultiTickerFeatureExtractor:
    """Extract features optimized for multi-ticker dataset"""
    
    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def calculate_macd(self, prices):
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram
    
    def extract_features(self, df):
        """Extract 35 features optimized for multi-ticker learning"""
        features_list = []
        targets_list = []
        
        # Group by ticker for consistent feature extraction
        for ticker in df['Ticker'].unique():
            ticker_df = df[df['Ticker'] == ticker].copy().reset_index(drop=True)
            
            if len(ticker_df) < 50:  # Skip if not enough data
                continue
            
            # === PRICE FEATURES (6) ===
            ticker_df['returns'] = ticker_df['Close'].pct_change()
            ticker_df['sma_20'] = ticker_df['Close'].rolling(20).mean()
            ticker_df['sma_50'] = ticker_df['Close'].rolling(50).mean()
            ticker_df['price_position'] = (ticker_df['Close'] - ticker_df['sma_20']) / (ticker_df['sma_20'] + 1e-6)
            ticker_df['high_low_ratio'] = (ticker_df['High'] - ticker_df['Low']) / ticker_df['Close']
            ticker_df['close_open_ratio'] = (ticker_df['Close'] - ticker_df['Open']) / (ticker_df['Open'] + 1e-6)
            
            # === MOMENTUM FEATURES (10) ===
            ticker_df['rsi_14'] = self.calculate_rsi(ticker_df['Close'], 14)
            ticker_df['rsi_7'] = self.calculate_rsi(ticker_df['Close'], 7)
            macd, signal, hist = self.calculate_macd(ticker_df['Close'])
            ticker_df['macd'] = macd
            ticker_df['macd_signal'] = signal
            ticker_df['macd_histogram'] = hist
            ticker_df['momentum_10'] = ticker_df['Close'] - ticker_df['Close'].shift(10)
            ticker_df['momentum_5'] = ticker_df['Close'] - ticker_df['Close'].shift(5)
            ticker_df['rate_of_change'] = (ticker_df['Close'] - ticker_df['Close'].shift(1)) / (ticker_df['Close'].shift(1) + 1e-6)
            ticker_df['stoch_k'] = ((ticker_df['Close'] - ticker_df['Low'].rolling(14).min()) / 
                                   (ticker_df['High'].rolling(14).max() - ticker_df['Low'].rolling(14).min() + 1e-6)) * 100
            
            # === VOLATILITY FEATURES (6) ===
            ticker_df['volatility_20'] = ticker_df['returns'].rolling(20).std()
            ticker_df['volatility_5'] = ticker_df['returns'].rolling(5).std()
            ticker_df['atr'] = (ticker_df['High'] - ticker_df['Low']).rolling(14).mean()
            ticker_df['atr_ratio'] = ticker_df['atr'] / (ticker_df['Close'] + 1e-6)
            upper = ticker_df['Close'].rolling(20).mean() + ticker_df['Close'].rolling(20).std() * 2
            lower = ticker_df['Close'].rolling(20).mean() - ticker_df['Close'].rolling(20).std() * 2
            ticker_df['bb_position'] = (ticker_df['Close'] - lower) / (upper - lower + 1e-6)
            
            # === VOLUME FEATURES (5) ===
            ticker_df['volume_sma'] = ticker_df['Volume'].rolling(20).mean()
            ticker_df['volume_ratio'] = ticker_df['Volume'] / (ticker_df['volume_sma'] + 1e-6)
            ticker_df['price_volume_trend'] = ticker_df['Close'].pct_change() * ticker_df['Volume']
            ticker_df['obv'] = (np.sign(ticker_df['Close'].diff()) * ticker_df['Volume']).cumsum()
            ticker_df['obv_momentum'] = ticker_df['obv'] - ticker_df['obv'].shift(5)
            
            # === TREND FEATURES (4) ===
            ticker_df['ema_12'] = ticker_df['Close'].ewm(span=12, adjust=False).mean()
            ticker_df['ema_26'] = ticker_df['Close'].ewm(span=26, adjust=False).mean()
            ticker_df['ema_ratio'] = ticker_df['ema_12'] / (ticker_df['ema_26'] + 1e-6)
            ticker_df['trend_strength'] = (ticker_df['Close'] - ticker_df['Close'].rolling(20).min()) / \
                                         (ticker_df['Close'].rolling(20).max() - ticker_df['Close'].rolling(20).min() + 1e-6)
            
            # === TARGET ===
            ticker_df['target'] = (ticker_df['Close'].shift(-1) > ticker_df['Close']).astype(int)
            
            # Clean
            ticker_df = ticker_df.dropna()
            
            feature_cols = [
                'returns', 'sma_20', 'sma_50', 'price_position', 'high_low_ratio', 'close_open_ratio',
                'rsi_14', 'rsi_7', 'macd', 'macd_signal', 'macd_histogram', 'momentum_10', 'momentum_5',
                'rate_of_change', 'stoch_k',
                'volatility_20', 'volatility_5', 'atr', 'atr_ratio', 'bb_position',
                'volume_sma', 'volume_ratio', 'price_volume_trend', 'obv', 'obv_momentum',
                'ema_12', 'ema_26', 'ema_ratio', 'trend_strength'
            ]
            
            X = ticker_df[feature_cols].values
            y = ticker_df['target'].values
            
            features_list.append(X)
            targets_list.append(y)
            
            logger.info(f"  ✅ {ticker}: {len(X)} samples")
        
        # Merge all tickers
        X = np.vstack(features_list)
        y = np.hstack(targets_list)
        
        logger.info(f"\n✅ Total: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"📊 UP: {np.sum(y)} ({np.sum(y)/len(y)*100:.1f}%) | DOWN: {len(y) - np.sum(y)} ({(1-np.sum(y)/len(y))*100:.1f}%)")
        
        return X, y, feature_cols


class RobustMomentumClassifier(nn.Module):
    """Robust classifier for multi-ticker momentum prediction"""
    
    def __init__(self, input_dim=28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
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
        'precision': precision_score(all_targets, all_preds, zero_division=0),
        'recall': recall_score(all_targets, all_preds, zero_division=0),
        'f1': f1_score(all_targets, all_preds, zero_division=0),
        'auc': auc,
        'predictions': np.array(all_preds),
        'targets': np.array(all_targets),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/multi_ticker_history.csv')
    parser.add_argument('--output', default='models/v7_multiticker')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_split', type=float, default=0.2)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("🚀 PLOUTOS V7 - MULTI-TICKER MOMENTUM PREDICTOR")
    logger.info("="*70)
    logger.info(f"Device: {device}\n")
    
    # Load data
    logger.info("📊 Loading multi-ticker data...")
    df = pd.read_csv(args.data)
    logger.info(f"Loaded {len(df)} rows from {df['Ticker'].nunique()} tickers")
    
    # Extract features
    logger.info("\n📈 Extracting features for all tickers...")
    extractor = MultiTickerFeatureExtractor()
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
    
    # Class weights
    logger.info("\n⚡ Calculating class weights...")
    class_weights = torch.FloatTensor([
        len(y_train) / (2 * np.sum(y_train == 0)),
        len(y_train) / (2 * np.sum(y_train == 1))
    ])
    logger.info(f"Class weights: DOWN={class_weights[0]:.3f}, UP={class_weights[1]:.3f}")
    
    # DataLoaders
    sample_weights = class_weights[y_train].numpy()
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=args.batch_size, sampler=sampler
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
        batch_size=args.batch_size
    )
    
    # Create model
    logger.info(f"\n🧠 Creating model...")
    model = RobustMomentumClassifier(input_dim=X_scaled.shape[1]).to(device)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training
    logger.info(f"\n⚙️  Training...\n")
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_metrics = evaluate(model, test_loader, criterion, device)
        
        scheduler.step()
        
        if test_metrics['f1'] > best_f1:
            best_f1 = test_metrics['f1']
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f} Acc={train_acc:.3f} | "
                       f"Test Acc={test_metrics['accuracy']:.3f} F1={test_metrics['f1']:.3f} AUC={test_metrics['auc']:.3f}")
        
        if patience_counter > 40:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Final evaluation
    logger.info("\n" + "="*70)
    logger.info("🎉 FINAL RESULTS - MULTI-TICKER TRAINING")
    logger.info("="*70)
    
    model.load_state_dict(torch.load(output_dir / 'best_model.pth'))
    final_metrics = evaluate(model, test_loader, criterion, device)
    
    logger.info(f"\nAccuracy:  {final_metrics['accuracy']:.3f} (Target: > 55%)")
    logger.info(f"Precision: {final_metrics['precision']:.3f}")
    logger.info(f"Recall:    {final_metrics['recall']:.3f}")
    logger.info(f"F1-Score:  {final_metrics['f1']:.3f}")
    logger.info(f"AUC-ROC:   {final_metrics['auc']:.3f}")
    
    cm = confusion_matrix(final_metrics['targets'], final_metrics['predictions'])
    logger.info(f"\nConfusion Matrix:\n[[TN={cm[0,0]:4d}  FP={cm[0,1]:4d}]\n [FN={cm[1,0]:4d}  TP={cm[1,1]:4d}]]")
    
    # Save metadata
    metadata = {
        'model_type': 'RobustMomentumClassifier',
        'version': 'v7_multiticker',
        'input_dim': X_scaled.shape[1],
        'feature_columns': feature_cols,
        'total_samples': len(X),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'tickers': sorted(df['Ticker'].unique().tolist()),
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
