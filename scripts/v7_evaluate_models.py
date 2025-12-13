#!/usr/bin/env python3
"""
🧠 PLOUTOS V7.3 - Model Evaluator (The Audit)

Ce script teste les modèles entraînés sur des données récentes (Out-of-Sample)
pour vérifier leur performance réelle avant intégration dans le PPO.

Commande:
    python scripts/v7_evaluate_models.py

Auteur: Ploutos AI Team
Date: Dec 2025
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
from sklearn.metrics import accuracy_score, precision_score, classification_report

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

class EnhancedReversionModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.Tanh(), nn.Dropout(dropout)])
            prev_dim = h
        self.main_stack = nn.Sequential(*layers)
        self.attention = AttentionBlock(hidden_dims[-1], num_heads=2)
        self.classifier = nn.Sequential(nn.Linear(hidden_dims[-1], 32), nn.Tanh(), nn.Linear(32, 2))
    def forward(self, x):
        features = self.main_stack(self.input_norm(x))
        features = features + self.attention(features) * 0.05
        return self.classifier(features)

class EnhancedVolatilityModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = h
        self.main_stack = nn.Sequential(*layers)
        self.attention = AttentionBlock(hidden_dims[-1], num_heads=2)
        self.classifier = nn.Sequential(nn.Linear(hidden_dims[-1], 32), nn.ReLU(), nn.Linear(32, 2))
    def forward(self, x):
        features = self.main_stack(self.input_norm(x))
        features = features + self.attention(features) * 0.05
        return self.classifier(features)

# ========== FEATURE ENGINEERING ==========

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

# ========== EVALUATION ==========

def load_model(expert_type, device):
    """Load trained model and scaler. Detect input dimension from state_dict."""
    model_path = Path(f"models/v7_{expert_type}_expert_final.pth")
    scaler_path = Path(f"models/v7_{expert_type}_scaler_final.pkl")
    config_path = Path(f"logs/v7_{expert_type}_optimization_FIXED.json")
    
    if not model_path.exists() or not scaler_path.exists() or not config_path.exists():
        logger.warning(f"❌ Fichiers manquants pour {expert_type}. Ignoré.")
        return None, None
    
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
        params = config['best_params']
    
    scaler = joblib.load(scaler_path)
    state_dict = torch.load(model_path, map_location=device)
    
    # Find the input_norm.weight to get actual input dimension
    input_norm_weight_key = 'input_norm.weight'
    if input_norm_weight_key in state_dict:
        actual_input_dim = state_dict[input_norm_weight_key].shape[0]
    else:
        logger.error(f"{expert_type}: Cannot find input_norm.weight in state_dict")
        return None, None
    
    # Rebuild model with correct input dimension
    if expert_type == 'momentum':
        hidden_dims = [params['hidden1'], params['hidden2'], params['hidden3']]
        model = EnhancedMomentumClassifier(actual_input_dim, hidden_dims, params['dropout'])
    elif expert_type == 'reversion':
        hidden_dims = [params['hidden1'], params['hidden2']]
        model = EnhancedReversionModel(actual_input_dim, hidden_dims, params['dropout'])
    else:  # volatility
        hidden_dims = [params['hidden1'], params['hidden2']]
        model = EnhancedVolatilityModel(actual_input_dim, hidden_dims, params['dropout'])
    
    try:
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        logger.info(f"✅ {expert_type.upper()}: input_dim={actual_input_dim}, hidden={hidden_dims}")
        return model, scaler
    except Exception as e:
        logger.error(f"Erreur lors du load_state_dict pour {expert_type}: {e}")
        return None, None

def evaluate(tickers):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🧪 Démarrage de l'évaluation sur {device}...\n")
    
    experts = ['momentum', 'reversion', 'volatility']
    models = {}
    scalers = {}
    
    for expert in experts:
        m, s = load_model(expert, device)
        if m:
            models[expert] = m
            scalers[expert] = s
            
    if not models:
        logger.error("❌ Aucun modèle chargé !")
        return

    logger.info(f"📉 Téléchargement des données de TEST (6 derniers mois)...\n")
    
    feature_calculators = {
        'momentum': calculate_momentum_features,
        'reversion': calculate_reversion_features,
        'volatility': calculate_volatility_features
    }
    
    results = []
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="6mo", progress=False)
            if len(df) < 50:
                continue
            
            for expert_name, model in models.items():
                calc = feature_calculators[expert_name]
                features = calc(df)
                
                # Log feature dimensions
                n_features = features.shape[1]
                logger.debug(f"{ticker} - {expert_name}: {n_features} features")
                
                df_aligned = df.loc[features.index]
                future_returns = df_aligned['Close'].pct_change(5).shift(-5)
                
                valid_mask = ~future_returns.isna()
                X = features.values[valid_mask]
                y_true = (future_returns[valid_mask] > 0).astype(int).values
                actual_returns = future_returns[valid_mask].values
                
                if len(X) < 10:
                    logger.debug(f"{ticker} - {expert_name}: Insuffisant ({len(X)} samples)")
                    continue
                
                # Check dimensions before scaling
                expected_dim = list(models.values())[0].input_norm.weight.shape[0] if expert_name == list(models.keys())[0] else None
                if X.shape[1] != model.input_norm.weight.shape[0]:
                    logger.warning(f"{ticker} - {expert_name}: Shape mismatch: got {X.shape[1]}, expected {model.input_norm.weight.shape[0]}")
                    continue
                
                X_scaled = scalers[expert_name].transform(X)
                X_tensor = torch.FloatTensor(X_scaled).to(device)
                
                with torch.no_grad():
                    logits = model(X_tensor)
                    probs = torch.softmax(logits, dim=1)
                    predictions = torch.argmax(probs, dim=1).cpu().numpy()
                
                acc = accuracy_score(y_true, predictions)
                algo_return = np.sum(actual_returns[predictions == 1])
                buy_hold_return = np.sum(actual_returns)
                
                results.append({
                    'Ticker': ticker,
                    'Expert': expert_name,
                    'Accuracy': acc,
                    'Algo_Return': algo_return,
                    'BuyHold_Return': buy_hold_return,
                    'Outperform': algo_return > buy_hold_return
                })
                logger.debug(f"{ticker} - {expert_name}: ✓ OK")
                
        except Exception as e:
            logger.debug(f"⚠️  Erreur sur {ticker}: {str(e)[:80]}")

    # Summary
    if results:
        df_res = pd.DataFrame(results)
        print("\n" + "="*80)
        print("🏆 RÉSULTATS DE L'ÉVALUATION (6 derniers mois)")
        print("="*80)
        
        summary = df_res.groupby('Expert').agg({
            'Accuracy': 'mean',
            'Algo_Return': 'mean',
            'BuyHold_Return': 'mean',
            'Outperform': 'mean'
        }).round(4)
        
        print("\n📊 MOYENNES PAR EXPERT :")
        print(summary)
        
        print("\n💡 INTERPRÉTATION :")
        for expert in summary.index:
            acc = summary.loc[expert, 'Accuracy']
            out = summary.loc[expert, 'Outperform']
            algo_ret = summary.loc[expert, 'Algo_Return']
            buy_hold = summary.loc[expert, 'BuyHold_Return']
            edge = algo_ret - buy_hold
            print(f"   {expert.upper():12} | Précision: {acc:6.1%} | Surperf: {out:6.1%} | Edge: {edge:+.2%}")
            
        print("\n" + "="*80)
    else:
        logger.error("❌ Aucun résultat généré.")
        logger.error("💡 Conseil: Relancez avec --debug pour voir les détails.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers', default='NVDA,AAPL,MSFT,TSLA,AMZN,GOOGL,META,NFLX,AMD,INTC', help="Tickers de test")
    parser.add_argument('--debug', action='store_true', help="Mode debug (affiche plus de détails)")
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
    
    evaluate(args.tickers.split(','))
