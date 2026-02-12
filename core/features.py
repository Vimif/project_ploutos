# core/features.py
"""Calcul des features pour le trading"""

import numpy as np
import pandas as pd
import yfinance as yf
from config.settings import N_FEATURES

class FeatureCalculator:
    """Calcul centralisé des features"""
    
    @staticmethod
    def calculate(ticker, n_features=30, lookback_days=90):
        """
        Calculer les features pour un ticker
        
        Args:
            ticker: Symbole du ticker
            n_features: Nombre de features à générer
            lookback_days: Nombre de jours historiques
        
        Returns:
            np.array de shape (n_features,) ou None si erreur
        """
        try:
            # Télécharger données
            df = yf.download(ticker, period=f'{lookback_days}d', interval='1d', progress=False)
            
            if df is None or len(df) < 50:
                return None
            
            # Flatten MultiIndex si nécessaire
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Initialiser
            features = np.zeros(n_features, dtype=np.float32)
            idx = 0
            
            # Prix actuel
            current_price = float(df['Close'].iloc[-1])
            
            # 1. Prix normalisés (5 jours)
            if idx + 5 <= n_features:
                close = df['Close'].values[-5:]
                close = close.flatten() if close.ndim > 1 else close
                close_norm = (close - close.mean()) / (close.std() + 1e-8)
                features[idx:idx+5] = close_norm.astype(np.float32)
                idx += 5
            
            # 2. Volumes normalisés (5 jours)
            if idx + 5 <= n_features:
                volume = df['Volume'].values[-5:]
                volume = volume.flatten() if volume.ndim > 1 else volume
                vol_norm = (volume - volume.mean()) / (volume.std() + 1e-8)
                features[idx:idx+5] = vol_norm.astype(np.float32)
                idx += 5
            
            # 3. Returns (5 jours)
            if idx + 5 <= n_features:
                returns = df['Close'].pct_change().fillna(0).values[-5:]
                returns = returns.flatten() if returns.ndim > 1 else returns
                features[idx:idx+5] = returns.astype(np.float32)
                idx += 5
            
            # 4. RSI
            if idx < n_features:
                features[idx] = FeatureCalculator._calculate_rsi(df)
                idx += 1
            
            # 5. MA 20
            if idx < n_features:
                ma20 = df['Close'].rolling(20).mean().iloc[-1]
                features[idx] = float((current_price - ma20) / current_price) if not pd.isna(ma20) else 0.0
                idx += 1
            
            # 6. MA 50
            if idx < n_features:
                ma50 = df['Close'].rolling(50).mean().iloc[-1]
                features[idx] = float((current_price - ma50) / current_price) if not pd.isna(ma50) else 0.0
                idx += 1
            
            # 7. Volatilité
            if idx < n_features:
                vol = df['Close'].pct_change().rolling(20).std().iloc[-1]
                features[idx] = float(vol) if not pd.isna(vol) else 0.0
                idx += 1
            
            # 8. Position (placeholder)
            if idx < n_features:
                features[idx] = 0.0
                idx += 1
            
            # 9. P&L (placeholder)
            if idx < n_features:
                features[idx] = 0.0
                idx += 1
            
            return features
            
        except Exception as e:
            print(f"❌ Erreur features pour {ticker}: {e}")
            return None
    
    @staticmethod
    def _calculate_rsi(df, period=14):
        """Calculer RSI"""
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        rsi_val = rsi.iloc[-1]
        return float(rsi_val / 100.0) if not pd.isna(rsi_val) else 0.5
