#!/usr/bin/env python3
"""
🧠 PLOUTOS V7.2 - Final Model Trainer from Config

Ce script charge les meilleurs hyperparamètres depuis un fichier de configuration JSON
(généré par v7_hyperparameter_optimizer_fixed.py) et entraîne le modèle final sur
l'intégralité du jeu de données.

Commande:
    python scripts/v7_train_from_config.py --config logs/v7_reversion_optimization_FIXED.json

Auteur: Ploutos AI Team
Date: Dec 2025
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
import joblib
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== MODELS (doivent être identiques à ceux de l'optimizer) =========

class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0
        
        self.scale = self.head_dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        b = x.shape[0]
        x = x.unsqueeze(1)
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(b, -1, self.num_heads, self.head_dim * 3)
        q, k, v = qkv[..., :self.head_dim], qkv[..., self.head_dim:2*self.head_dim], qkv[..., 2*self.head_dim:]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.reshape(b, -1, self.dim)
        return self.to_out(out).squeeze(1)

class EnhancedMomentumClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout, use_attention=True):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.main_stack = nn.Sequential(*layers)
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionBlock(hidden_dims[-1], num_heads=4)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 2)
        )
    def forward(self, x):
        x = self.input_norm(x)
        features = self.main_stack(x)
        if self.use_attention:
            features = features + self.attention(features) * 0.1
        return self.classifier(features)

class EnhancedReversionModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.main_stack = nn.Sequential(*layers)
        self.attention = AttentionBlock(hidden_dims[-1], num_heads=2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.Tanh(), nn.Linear(32, 2)
        )
    def forward(self, x):
        x = self.input_norm(x)
        features = self.main_stack(x)
        features = features + self.attention(features) * 0.05
        return self.classifier(features)

class EnhancedVolatilityModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.main_stack = nn.Sequential(*layers)
        self.attention = AttentionBlock(hidden_dims[-1], num_heads=2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.ReLU(), nn.Linear(32, 2)
        )
    def forward(self, x):
        x = self.input_norm(x)
        features = self.main_stack(x)
        features = features + self.attention(features) * 0.05
        return self.classifier(features)


class WeightedFocalLoss(nn.Module):
    def __init__(self, class_weights, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
    
    def forward(self, logits, targets):
        p = torch.softmax(logits, dim=1)
        ce_loss = nn.functional.cross_entropy(logits, targets, weight=self.class_weights, reduction='none')
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t).pow(self.gamma)
        focal_loss = self.alpha * focal_weight * ce_loss
        return focal_loss.mean()

# ========== FEATURE ENGINEERING (identique à l'optimizer) =========

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

def calculate_reversion_features(df):
    features = pd.DataFrame(index=df.index)
    close = df['Close'].values.flatten()
    sma_20 = pd.Series(close).rolling(20).mean().values
    sma_50 = pd.Series(close).rolling(50).mean().values
    features['sma_20'] = sma_20
    features['sma_50'] = sma_50
    features['dist_sma20'] = close - sma_20
    features['dist_sma50'] = close - sma_50
    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features['rsi'] = (100 - (100 / (1 + rs))).values
    features['volatility'] = pd.Series(close).pct_change().rolling(20).std().values
    features['returns'] = pd.Series(close).pct_change().values
    return features.dropna()

def calculate_volatility_features(df):
    features = pd.DataFrame(index=df.index)
    log_returns = np.log(df['Close'] / df['Close'].shift(1))
    features['volatility_10'] = log_returns.rolling(10).std() * np.sqrt(252)
    features['volatility_20'] = log_returns.rolling(20).std() * np.sqrt(252)
    features['volatility_50'] = log_returns.rolling(50).std() * np.sqrt(252)
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    features['atr'] = true_range.rolling(14).mean()
    features['return_vol'] = (df['Close'].pct_change()).rolling(20).std()
    features['volume_ma'] = df['Volume'].rolling(20).mean()
    return features.dropna()

# ========== MAIN TRAINING FUNCTION =========

def train_final_model(config_path, tickers):
    logger.info(f"💾 Chargement de la configuration: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    expert_type = config['expert_type']
    params = config['best_params']
    
    logger.info(f"🧠 Expert: {expert_type.upper()}")
    logger.info(f"⚙️ Hyperparamètres: {params}")

    logger.info(f"🔍 Téléchargement et feature engineering pour {len(tickers)} tickers...")
    
    feature_calculator = {
        'momentum': calculate_momentum_features,
        'reversion': calculate_reversion_features,
        'volatility': calculate_volatility_features
    }
    
    all_X = []
    all_y = []
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="3y", progress=False) # On prend plus de données pour le modèle final
            if df.empty or len(df) < 100:
                continue
            
            features = feature_calculator[expert_type](df)
            if len(features) < 50:
                continue
            
            df_aligned = df.loc[features.index]
            returns_5d = df_aligned['Close'].pct_change(5).shift(-5)
            target = (returns_5d > 0).astype(int).fillna(0)
            target_np = target.values.flatten()
            X_np = features.values
            
            if len(X_np) != len(target_np):
                min_len = min(len(X_np), len(target_np))
                X_np = X_np[:min_len]
                target_np = target_np[:min_len]
            
            X_np = X_np[:-5]
            target_np = target_np[:-5]
            
            if len(X_np) < 50:
                continue
            
            all_X.append(X_np)
            all_y.append(target_np)
            logger.info(f"   ✅ {ticker}: {len(X_np)} samples")
            
        except Exception as e:
            logger.warning(f"   ⚠️  {ticker}: {str(e)[:100]}")
    
    if not all_X:
        logger.error("❌ Aucune donnée pour l'entraînement!")
        return

    X = np.vstack(all_X)
    y = np.concatenate(all_y).flatten()
    
    logger.info(f"📊 Dataset complet: X={X.shape}, y={y.shape}")
    logger.info(f"📈 Classes: {np.bincount(y)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train = torch.FloatTensor(X_scaled)
    y_train = torch.LongTensor(y)

    # --- Instanciation du Modèle ---
    if expert_type == 'momentum':
        hidden_dims = [params['hidden1'], params['hidden2'], params['hidden3']]
        model = EnhancedMomentumClassifier(X_train.shape[1], hidden_dims, params['dropout'])
    elif expert_type == 'reversion':
        hidden_dims = [params['hidden1'], params['hidden2']]
        model = EnhancedReversionModel(X_train.shape[1], hidden_dims, params['dropout'])
    else: # volatility
        hidden_dims = [params['hidden1'], params['hidden2']]
        model = EnhancedVolatilityModel(X_train.shape[1], hidden_dims, params['dropout'])
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    train_set = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    class_weights = torch.tensor([0.4, 0.6]).to(device)
    criterion = WeightedFocalLoss(class_weights)

    logger.info(f"🚀 Début de l'entraînement pour {params['epochs']} epochs...")
    for epoch in range(params['epochs']):
        model.train()
        total_loss = 0
        for i, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"   Epoch {epoch+1}/{params['epochs']}, Loss: {avg_loss:.6f}")

    # --- Sauvegarde ---
    output_dir = Path('models')
    output_dir.mkdir(exist_ok=True)
    
    model_filename = f"v7_{expert_type}_expert_final.pth"
    scaler_filename = f"v7_{expert_type}_scaler_final.pkl"
    
    torch.save(model.state_dict(), output_dir / model_filename)
    joblib.dump(scaler, output_dir / scaler_filename)

    logger.info("\n" + "="*70)
    logger.info("🎉 ENTRAÎNEMENT TERMINÉ 🎉")
    logger.info(f"   - Modèle: {output_dir / model_filename}")
    logger.info(f"   - Scaler: {output_dir / scaler_filename}")
    logger.info("="*70 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Chemin vers le fichier de config JSON de l'optimisation")
    parser.add_argument('--tickers', default='NVDA,AAPL,MSFT,GOOGL,AMZN,TSLA,META,NFLX,SPY,QQQ,VOO,XOM,CVX,JPM,BAC', help="Liste des tickers pour l'entraînement final")
    
    args = parser.parse_args()
    tickers = args.tickers.split(',')
    
    print("\n" + "="*70)
    print("🧠 PLOUTOS V7.2 - Final Model Trainer")
    print("="*70)
    print(f"🌟 GPU: {torch.cuda.is_available()}")
    print(f"📅 Tickers: {len(tickers)}")
    print(f"🔌 Config: {args.config}")
    print("="*70 + "\n")
    
    train_final_model(args.config, tickers)
