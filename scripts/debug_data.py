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

print("=== TESTING CONCATENATION ===")
tickers = ["NVDA", "MSFT", "AAPL"]
all_X = []
all_y = []

for ticker in tickers:
    df = yf.download(ticker, period="5y", progress=False)
    features = calculate_momentum_features(df)
    future_returns = df['Close'].pct_change(5).shift(-5)
    common_index = features.index.intersection(future_returns.index)
    X = features.loc[common_index]
    y = (future_returns.loc[common_index] > 0).astype(int)
    valid_mask = ~(y.isna() | X.isna().any(axis=1))
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"\n{ticker}:")
    print(f"  Shape: {X.shape}")
    print(f"  NaN per column: {X.isna().sum()}")
    print(f"  Total rows with NaN: {X.isna().any(axis=1).sum()}")
    
    all_X.append(X.values)
    all_y.append(y.values.flatten())

print("\n=== AFTER CONCATENATION ===")
X_concat = np.concatenate(all_X, axis=0)
y_concat = np.concatenate(all_y, axis=0)

print(f"Shape: {X_concat.shape}")
print(f"Total NaN count: {np.isnan(X_concat).sum()}")
print(f"Rows with NaN: {np.isnan(X_concat).any(axis=1).sum()}")
print(f"\nNaN count per column:")
for i in range(X_concat.shape[1]):
    nan_count = np.isnan(X_concat[:, i]).sum()
    print(f"  Column {i}: {nan_count} NaN")

print(f"\n=== AFTER NaN REMOVAL ===")
nan_mask = ~np.isnan(X_concat).any(axis=1)
X_clean = X_concat[nan_mask]
y_clean = y_concat[nan_mask]
print(f"Final shape: X={X_clean.shape}, y={y_clean.shape}")
print(f"NaN in X: {np.isnan(X_clean).any()}")
