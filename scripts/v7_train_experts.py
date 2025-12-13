#!/usr/bin/env python3
"""
Ploutos V7 - Expert Trainer
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Model Definitions

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
    def __init__(self, input_dim, hidden_dims, dropout, use_attention=False):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = h
        self.main_stack = nn.Sequential(*layers)
        self.use_attention = use_attention
        if use_attention: self.attention = AttentionBlock(hidden_dims[-1], num_heads=4)
        self.classifier = nn.Sequential(nn.Linear(hidden_dims[-1], 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 2))
    def forward(self, x):
        features = self.main_stack(self.input_norm(x))
        if self.use_attention: features = features + self.attention(features) * 0.1
        return self.classifier(features)

# Feature Engineering

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
    return features

def train_expert(expert_type, tickers, epochs):
    logger.info(f"--- Training {expert_type.upper()} Expert ---")
    
    # 1. Data Loading and Alignment
    all_X = []
    all_y = []
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="5y", progress=False)
            if len(df) < 252: 
                continue

            # Calculate features
            features = calculate_momentum_features(df)
            
            # Calculate future returns (5-day forward return)
            future_returns = df['Close'].pct_change(5).shift(-5)
            
            # Align by truncating last 5 rows
            features = features[:-5]
            future_returns = future_returns[:-5]
            
            # Dropna and convert to numpy immediately to avoid index issues
            features_clean = features.dropna()
            future_returns_clean = future_returns.loc[features_clean.index]
            
            X = features_clean.values
            y = (future_returns_clean > 0).astype(int).values.flatten()
            
            if len(X) < 100:
                logger.warning(f"Not enough samples for {ticker}: {len(X)}")
                continue
            
            all_X.append(X)
            all_y.append(y)
            logger.info(f"Loaded {ticker}: {len(X)} samples")
            
        except Exception as e:
            logger.warning(f"Error loading {ticker}: {e}")
            continue
    
    if not all_X:
        logger.error("No data loaded!")
        return
    
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    logger.info(f"\nFinal data shapes: X={X.shape}, y={y.shape}")
    logger.info(f"NaN in X: {np.isnan(X).any()}, NaN in y: {np.isnan(y).any()}")
    logger.info(f"Inf in X: {np.isinf(X).any()}, Inf in y: {np.isinf(y).any()}")
    logger.info(f"Class distribution: {np.bincount(y)}")

    # 2. Data Splitting and Scaling
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError as e:
        logger.error(f"Cannot stratify: {e}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")

    # 3. DataLoader
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled), 
        torch.LongTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    # 4. Model, Loss, Optimizer
    input_dim = X_train_scaled.shape[1]
    model = EnhancedMomentumClassifier(input_dim, [128, 64], 0.3, use_attention=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    # 5. Training Loop
    logger.info(f"\nStarting training on {device}...\n")
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            test_outputs = model(torch.FloatTensor(X_test_scaled).to(device))
            _, predicted = torch.max(test_outputs, 1)
            accuracy = (predicted == torch.LongTensor(y_test).to(device)).float().mean().item()

        scheduler.step(accuracy)
        
        if accuracy > best_acc:
            best_acc = accuracy

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Val Acc: {accuracy:.4f}, Best: {best_acc:.4f}")

    # 6. Save Model and Scaler
    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), f"models/v7_{expert_type}_expert_final.pth")
    joblib.dump(scaler, f"models/v7_{expert_type}_scaler_final.pkl")
    logger.info(f"\n✅ Model and scaler for {expert_type} saved. Best Val Acc: {best_acc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", type=str, default="momentum", help="Expert to train")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    args = parser.parse_args()

    tickers = ["NVDA", "MSFT", "AAPL", "GOOGL", "AMZN", "SPY", "QQQ", "XOM", "JPM"]
    train_expert(args.expert, tickers, args.epochs)
