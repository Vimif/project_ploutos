def calculate_reversion_features(df):
    """Calcule les features pour Reversion Expert (FIXED avec .values)"""
    features = pd.DataFrame(index=df.index)
    features['sma_20'] = df['Close'].rolling(20).mean()
    features['sma_50'] = df['Close'].rolling(50).mean()
    
    # Use .values to avoid pandas shape mismatch
    features['dist_sma20'] = (df['Close'].values - features['sma_20'].values)
    features['dist_sma50'] = (df['Close'].values - features['sma_50'].values)
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))
    features['volatility'] = df['Close'].pct_change().rolling(20).std()
    features['returns'] = df['Close'].pct_change()
    features = features.dropna()
    return features
