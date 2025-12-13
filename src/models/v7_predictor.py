#!/usr/bin/env python3
"""
Ploutos V7 - Predictor Wrapper
"""

import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

# Model Architecture (must match training)

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
            layers.extend([nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = h
        self.main_stack = nn.Sequential(*layers)
        self.use_attention = use_attention
        if use_attention: self.attention = AttentionBlock(hidden_dims[-1], num_heads=4)
        self.classifier = nn.Sequential(nn.Linear(hidden_dims[-1], 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2))
    
    def forward(self, x):
        features = self.main_stack(self.input_norm(x))
        if self.use_attention: features = features + self.attention(features) * 0.2
        return self.classifier(features)

# Feature Engineering (must match training)

def calculate_enhanced_features(df):
    """Calculate 26 features matching training"""
    features = pd.DataFrame(index=df.index)
    
    # Price features
    features['returns'] = df['Close'].pct_change()
    features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    features['returns_5d'] = df['Close'].pct_change(5)
    features['returns_10d'] = df['Close'].pct_change(10)
    features['returns_20d'] = df['Close'].pct_change(20)
    
    # Moving averages
    features['sma_5'] = df['Close'].rolling(5).mean()
    features['sma_10'] = df['Close'].rolling(10).mean()
    features['sma_20'] = df['Close'].rolling(20).mean()
    features['sma_50'] = df['Close'].rolling(50).mean()
    
    # EMA for MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    features['ema_12'] = ema12
    features['ema_26'] = ema26
    
    # Volatility
    features['volatility_5'] = df['Close'].pct_change().rolling(5).std()
    features['volatility_10'] = df['Close'].pct_change().rolling(10).std()
    features['volatility_20'] = df['Close'].pct_change().rolling(20).std()
    features['atr'] = (df['High'] - df['Low']).rolling(14).mean()
    
    # Momentum indicators
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))
    features['rsi_sma'] = features['rsi'].rolling(14).mean()
    
    # MACD
    macd = ema12 - ema26
    features['macd'] = macd
    features['signal'] = macd.ewm(span=9).mean()
    features['macd_hist'] = features['macd'] - features['signal']
    
    # Volume
    vol_sma = df['Volume'].rolling(20).mean()
    features['volume_ratio'] = df['Volume'] / vol_sma
    
    # Price patterns
    features['high_low_range'] = (df['High'] - df['Low']) / df['Close']
    features['close_open_diff'] = (df['Close'] - df['Open']) / df['Open']
    
    # Momentum
    features['momentum_5'] = df['Close'] - df['Close'].shift(5)
    features['momentum_10'] = df['Close'] - df['Close'].shift(10)
    features['momentum_20'] = df['Close'] - df['Close'].shift(20)
    
    return features

class V7Predictor:
    """V7 Enhanced Momentum Predictor"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.loaded = False
        
    def load(self, expert: str = "momentum"):
        """Load model and scaler"""
        try:
            model_path = self.model_dir / f"v7_{expert}_enhanced_best.pth"
            scaler_path = self.model_dir / f"v7_{expert}_enhanced_scaler.pkl"
            
            if not model_path.exists() or not scaler_path.exists():
                logger.error(f"Model files not found: {model_path}")
                return False
            
            # Load scaler
            self.scaler = joblib.load(scaler_path)
            
            # Load model (26 features, not 27!)
            input_dim = 26
            self.model = EnhancedMomentumClassifier(
                input_dim=input_dim,
                hidden_dims=[512, 256, 128],
                dropout=0.4,
                use_attention=True
            )
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            self.loaded = True
            logger.info(f"✅ V7 {expert} model loaded (68.35% accuracy)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading V7 model: {e}")
            return False
    
    def predict(self, ticker: str, period: str = "3mo") -> Dict:
        """Predict if ticker will go up in next 5 days"""
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            # Download data
            df = yf.download(ticker, period=period, progress=False)
            if len(df) < 60:
                return {"error": "Not enough data", "ticker": ticker}
            
            # Calculate features
            features = calculate_enhanced_features(df)
            features_clean = features.dropna()
            
            if len(features_clean) == 0:
                return {"error": "No valid features", "ticker": ticker}
            
            # Get last row (current state)
            X = features_clean.iloc[-1:].values
            X_scaled = self.scaler.transform(X)
            
            # Predict
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled).to(self.device)
                outputs = self.model(X_tensor)
                probas = torch.softmax(outputs, dim=1)
                pred_class = torch.argmax(probas, dim=1).item()
                confidence = probas[0][pred_class].item()
            
            prediction = "UP" if pred_class == 1 else "DOWN"
            
            return {
                "ticker": ticker,
                "prediction": prediction,
                "confidence": confidence,
                "signal_strength": confidence - 0.5,  # 0 to 0.5 scale
                "probabilities": {
                    "down": probas[0][0].item(),
                    "up": probas[0][1].item()
                }
            }
            
        except Exception as e:
            logger.error(f"Error predicting {ticker}: {e}")
            return {"error": str(e), "ticker": ticker}
    
    def predict_batch(self, tickers: List[str]) -> List[Dict]:
        """Predict multiple tickers"""
        return [self.predict(ticker) for ticker in tickers]

if __name__ == '__main__':
    # Test
    logging.basicConfig(level=logging.INFO)
    
    predictor = V7Predictor()
    if predictor.load("momentum"):
        result = predictor.predict("NVDA")
        print(f"\n{result['ticker']}: {result['prediction']} (confidence: {result['confidence']:.2%})")
