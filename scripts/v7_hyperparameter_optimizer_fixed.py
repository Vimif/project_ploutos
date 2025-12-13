def calculate_reversion_features(df):
    """Calcule les features pour Reversion Expert (SIMPLIFIED)"""
    features = pd.DataFrame(index=df.index)
    
    # SMA
    features['sma_20'] = df['Close'].rolling(20).mean()
    features['sma_50'] = df['Close'].rolling(50).mean()
    
    # Price distance from SMA
    features['dist_sma20'] = df['Close'] - features['sma_20']
    features['dist_sma50'] = df['Close'] - features['sma_50']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # Volatility
    features['volatility'] = df['Close'].pct_change().rolling(20).std()
    
    # Return (simple)
    features['returns'] = df['Close'].pct_change()
    
    features = features.dropna()
    return features
