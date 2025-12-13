#!/usr/bin/env python3
import numpy as np
import pandas as pd
import yfinance as yf

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

df = yf.download("NVDA", period="5y", progress=False)
features = calculate_momentum_features(df)

print("\n=== DIAGNOSTIC DES FEATURES ===")
print(f"Shape: {features.shape}")
print(f"\nNaN count per column:")
print(features.isna().sum())
print(f"\nInf count per column:")
print(np.isinf(features).sum())
print(f"\nStats:")
print(features.describe())

# Check for problematic values
if features.isna().any().any():
    print("\n⚠️ WARNING: NaN detected in features!")
    print(features[features.isna().any(axis=1)].head())

if np.isinf(features.values).any():
    print("\n⚠️ WARNING: Inf detected in features!")
    print(features[np.isinf(features).any(axis=1)].head())

print("\n=== CHECKING LABELS ===")
future_returns = df['Close'].pct_change(5).shift(-5)
common_index = features.index.intersection(future_returns.index)
X = features.loc[common_index]
y = (future_returns.loc[common_index] > 0).astype(int)
valid_mask = ~y.isna()
X = X[valid_mask]
y = y[valid_mask]

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"NaN in X: {X.isna().any().any()}")
print(f"NaN in y: {y.isna().any()}")
print(f"Inf in X: {np.isinf(X.values).any()}")
print(f"\nClass balance: {np.bincount(y.values.flatten())}")
