#!/usr/bin/env python3
"""
🧠 PLOUTOS V7.1 - Hyperparameter Optimizer (FIXED v2)

Utilise Optuna + vraies données financières (MACD, RSI, Bollinger Bands)
Pas de données aléatoires.

Commande:
    python scripts/v7_hyperparameter_optimizer_fixed.py --expert momentum --trials 50

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
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== MODELS ==========

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
    def __init__(self, input_dim=28, hidden_dims=[512, 256, 128], dropout=0.3, use_attention=True):
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
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        x = self.input_norm(x)
        features = self.main_stack(x)
        if self.use_attention:
            features = features + self.attention(features) * 0.1
        return self.classifier(features)

class EnhancedReversionModel(nn.Module):
    def __init__(self, input_dim=9, hidden_dims=[256, 128], dropout=0.2):
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
            nn.Tanh(),
            nn.Linear(32, 2)
        )
    
    def forward(self, x):
        x = self.input_norm(x)
        features = self.main_stack(x)
        features = features + self.attention(features) * 0.05
        return self.classifier(features)

class EnhancedVolatilityModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dims=[128, 64], dropout=0.15):
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
            nn.ReLU(),
            nn.Linear(32, 2)
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

# ========== FEATURE ENGINEERING ==========

def calculate_momentum_features(df):
    """Calcule les features pour Momentum Expert"""
    features = pd.DataFrame(index=df.index)
    
    # Basic
    features['returns'] = df['Close'].pct_change()
    features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving averages
    features['sma_10'] = df['Close'].rolling(10).mean()
    features['sma_20'] = df['Close'].rolling(20).mean()
    features['sma_50'] = df['Close'].rolling(50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    features['macd'] = exp1 - exp2
    features['signal'] = features['macd'].ewm(span=9, adjust=False).mean()
    features['macd_hist'] = features['macd'] - features['signal']
    
    # Momentum
    features['momentum_10'] = df['Close'] - df['Close'].shift(10)
    features['momentum_20'] = df['Close'] - df['Close'].shift(20)
    
    features = features.dropna()
    return features

def calculate_reversion_features(df):
    """Calcule les features pour Reversion Expert"""
    features = pd.DataFrame(index=df.index)
    
    close = df['Close'].values
    
    # SMA
    features['sma_20'] = df['Close'].rolling(20).mean()
    features['sma_50'] = df['Close'].rolling(50).mean()
    
    # Bollinger Bands
    bb_middle = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    bb_upper = bb_middle + (bb_std * 2)
    bb_lower = bb_middle - (bb_std * 2)
    
    features['bb_upper'] = bb_upper
    features['bb_lower'] = bb_lower
    features['bb_width'] = bb_upper - bb_lower
    
    # BB Position: normalize price within bands
    bb_width_safe = features['bb_width'].values + 1e-8
    bb_lower_vals = features['bb_lower'].values
    close_vals = df['Close'].values
    
    features['bb_position'] = (close_vals - bb_lower_vals) / bb_width_safe
    
    # Z-score
    sma_20_vals = features['sma_20'].values
    std_safe = (df['Close'].rolling(20).std().values + 1e-8)
    features['z_score'] = (close_vals - sma_20_vals) / std_safe
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))
    
    features = features.dropna()
    return features

def calculate_volatility_features(df):
    """Calcule les features pour Volatility Expert"""
    features = pd.DataFrame(index=df.index)
    
    # Volatility
    log_returns = np.log(df['Close'] / df['Close'].shift(1))
    features['volatility_10'] = log_returns.rolling(10).std() * np.sqrt(252)
    features['volatility_20'] = log_returns.rolling(20).std() * np.sqrt(252)
    features['volatility_50'] = log_returns.rolling(50).std() * np.sqrt(252)
    
    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    features['atr'] = true_range.rolling(14).mean()
    
    # Return volatility
    features['return_vol'] = (df['Close'].pct_change()).rolling(20).std()
    
    # Volume
    features['volume_ma'] = df['Volume'].rolling(20).mean()
    
    features = features.dropna()
    return features

# ========== OPTIMIZATION OBJECTIVE ==========

class OptunaObjective:
    def __init__(self, X_train, y_train, X_val, y_val, expert_type='momentum'):
        y_train = np.asarray(y_train).flatten()  # CRITICAL: Flatten to 1D
        y_val = np.asarray(y_val).flatten()      # CRITICAL: Flatten to 1D
        assert X_train.shape[0] == y_train.shape[0], f"Shapes mismatch: {X_train.shape[0]} != {y_train.shape[0]}"
        assert X_val.shape[0] == y_val.shape[0], f"Val shapes mismatch: {X_val.shape[0]} != {y_val.shape[0]}"
        
        self.X_train = torch.FloatTensor(X_train)
        self.y_train = torch.LongTensor(y_train)
        self.X_val = torch.FloatTensor(X_val)
        self.y_val = torch.LongTensor(y_val)
        self.expert_type = expert_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __call__(self, trial):
        lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        epochs = trial.suggest_int('epochs', 20, 100, step=10)
        
        # Dynamic input dims based on features
        if self.expert_type == 'momentum':
            input_dim = self.X_train.shape[1]
            h1 = trial.suggest_int('hidden1', 128, 512, step=64)
            h2 = trial.suggest_int('hidden2', 64, 256, step=64)
            h3 = trial.suggest_int('hidden3', 32, 128, step=32)
            hidden_dims = [h1, h2, h3]
            model = EnhancedMomentumClassifier(input_dim, hidden_dims, dropout)
        elif self.expert_type == 'reversion':
            input_dim = self.X_train.shape[1]
            h1 = trial.suggest_int('hidden1', 64, 256, step=64)
            h2 = trial.suggest_int('hidden2', 32, 128, step=32)
            hidden_dims = [h1, h2]
            model = EnhancedReversionModel(input_dim, hidden_dims, dropout)
        else:  # volatility
            input_dim = self.X_train.shape[1]
            h1 = trial.suggest_int('hidden1', 32, 128, step=32)
            h2 = trial.suggest_int('hidden2', 16, 64, step=16)
            hidden_dims = [h1, h2]
            model = EnhancedVolatilityModel(input_dim, hidden_dims, dropout)
        
        model = model.to(self.device)
        
        train_set = TensorDataset(self.X_train, self.y_train)
        val_set = TensorDataset(self.X_val, self.y_val)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        class_weights = torch.tensor([0.4, 0.6]).to(self.device)
        criterion = WeightedFocalLoss(class_weights)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        return best_val_loss

# ========== MAIN ==========

def optimize_expert(expert_type, tickers, trials=50, timeout=3600):
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
            df = yf.download(ticker, period="2y", progress=False)
            if df.empty or len(df) < 100:
                continue
            
            features = feature_calculator[expert_type](df)
            if len(features) < 50:
                continue
            
            # Target: Create 1D numpy array EXPLICITLY
            df_aligned = df.loc[features.index]
            returns_5d = df_aligned['Close'].pct_change(5).shift(-5)
            target_np = (returns_5d.values > 0).astype(int)  # Force to 1D numpy array
            
            # Ensure both are same length
            X_np = features.values
            
            if len(X_np) != len(target_np):
                min_len = min(len(X_np), len(target_np))
                X_np = X_np[:min_len]
                target_np = target_np[:min_len]
            
            if len(X_np) < 50:
                continue
            
            all_X.append(X_np)
            all_y.append(target_np)
            logger.info(f"   ✅ {ticker}: {len(X_np)} samples")
            
        except Exception as e:
            logger.warning(f"   ⚠️  {ticker}: {str(e)[:50]}")
    
    if not all_X:
        logger.error("❌ No data!")
        return None
    
    X = np.vstack(all_X)
    y = np.concatenate(all_y).flatten()
    
    logger.info(f"📊 Raw: X={X.shape}, y={y.shape}")
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    logger.info(f"✅ Train: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"📈 Classes: {np.bincount(y_train)}")
    
    logger.info(f"\n🔬 Optuna: {trials} trials, {timeout}s timeout\n")
    
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction='minimize')
    
    objective = OptunaObjective(X_train, y_train, X_val, y_val, expert_type)
    study.optimize(objective, n_trials=trials, timeout=timeout, show_progress_bar=True)
    
    best_trial = study.best_trial
    logger.info("\n" + "="*70)
    logger.info(f"🎉 {expert_type.upper()} COMPLETE")
    logger.info("="*70)
    logger.info(f"📈 Best Val Loss: {best_trial.value:.6f}")
    logger.info(f"🔊 Best Trial: {best_trial.number}\n")
    for key, value in best_trial.params.items():
        logger.info(f"   {key}: {value}")
    
    results = {
        'expert_type': expert_type,
        'best_loss': float(best_trial.value),
        'best_trial': best_trial.number,
        'best_params': best_trial.params,
        'n_trials': len(study.trials),
        'timestamp': datetime.now().isoformat()
    }
    
    output_file = Path(f'logs/v7_{expert_type}_optimization_FIXED.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n💾 {output_file}\n")
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert', choices=['momentum', 'reversion', 'volatility', 'all'], default='momentum')
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--timeout', type=int, default=3600)
    parser.add_argument('--tickers', default='NVDA,AAPL,MSFT,GOOGL,AMZN,TSLA,META,NFLX')
    
    args = parser.parse_args()
    tickers = args.tickers.split(',')
    
    print("\n" + "="*70)
    print("🧠 PLOUTOS V7.1 - Hyperparameter Optimizer (FIXED v2)")
    print("="*70)
    print(f"🌟 GPU: {torch.cuda.is_available()}")
    print(f"📅 Tickers: {len(tickers)}")
    print(f"🔌 Trials: {args.trials}")
    print("="*70 + "\n")
    
    if args.expert == 'all':
        for expert in ['momentum', 'reversion', 'volatility']:
            optimize_expert(expert, tickers, args.trials, args.timeout)
    else:
        optimize_expert(args.expert, tickers, args.trials, args.timeout)
