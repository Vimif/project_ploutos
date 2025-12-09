# core/advanced_features.py
"""Features Engineering Avancé pour Trading - 50+ Indicateurs ROBUSTE"""

import numpy as np
import pandas as pd
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureEngineering:
    """Calculateur de features techniques avancées avec gestion NaN robuste"""
    
    def __init__(self):
        self.feature_names = []
    
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculer toutes les features pour un DataFrame OHLCV
        ✅ AVEC GESTION ROBUSTE DES NaN/Inf
        """
        features = df.copy()
        
        try:
            # 1. TREND INDICATORS
            features = self._add_trend_indicators(features)
            
            # 2. MOMENTUM INDICATORS  
            features = self._add_momentum_indicators(features)
            
            # 3. VOLATILITY INDICATORS
            features = self._add_volatility_indicators(features)
            
            # 4. VOLUME INDICATORS
            features = self._add_volume_indicators(features)
            
            # 5. PRICE ACTION
            features = self._add_price_action(features)
            
            # ✅ NETTOYAGE GLOBAL
            features = self._clean_features(features)
            
        except Exception as e:
            print(f"⚠️  Erreur features: {e}")
            # Fallback: retourner features basiques
            return self._get_basic_features(df)
        
        return features
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """✅ Nettoyage robuste de toutes les features"""
        # Colonnes à ne pas toucher
        protected = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        for col in df.columns:
            if col not in protected:
                # 1. Remplacer Inf
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                
                # 2. Forward fill puis backward fill
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                
                # 3. Si encore NaN, remplir avec 0
                df[col] = df[col].fillna(0)
                
                # 4. Winsorization (clip outliers)
                q01 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)
                df[col] = df[col].clip(q01, q99)
                
                # 5. Normalisation robuste (IQR)
                median = df[col].median()
                q25 = df[col].quantile(0.25)
                q75 = df[col].quantile(0.75)
                iqr = q75 - q25
                
                if iqr > 0:
                    df[col] = (df[col] - median) / (iqr + 1e-8)
                
                # 6. Clip final
                df[col] = df[col].clip(-5, 5)
        
        return df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Indicateurs de tendance"""
        close = df['Close']
        
        # SMAs
        df['sma_10'] = close.rolling(10, min_periods=5).mean()
        df['sma_20'] = close.rolling(20, min_periods=10).mean()
        df['sma_50'] = close.rolling(50, min_periods=25).mean()
        
        # EMAs
        df['ema_10'] = close.ewm(span=10, min_periods=5).mean()
        df['ema_20'] = close.ewm(span=20, min_periods=10).mean()
        
        # MACD
        ema_12 = close.ewm(span=12, min_periods=6).mean()
        ema_26 = close.ewm(span=26, min_periods=13).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, min_periods=5).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # ADX (simplifié)
        high = df['High']
        low = df['Low']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(14, min_periods=7).mean()
        plus_di = 100 * (plus_dm.rolling(14, min_periods=7).mean() / (atr + 1e-8))
        minus_di = 100 * (minus_dm.rolling(14, min_periods=7).mean() / (atr + 1e-8))
        
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8))
        df['adx'] = dx.rolling(14, min_periods=7).mean()
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Indicateurs de momentum"""
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=7).mean()
        loss = -delta.where(delta < 0, 0).rolling(14, min_periods=7).mean()
        rs = gain / (loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Stochastic
        low_14 = low.rolling(14, min_periods=7).min()
        high_14 = high.rolling(14, min_periods=7).max()
        df['stoch_k'] = 100 * ((close - low_14) / (high_14 - low_14 + 1e-8))
        df['stoch_d'] = df['stoch_k'].rolling(3, min_periods=2).mean()
        
        # ROC
        df['roc_10'] = 100 * (close / close.shift(10) - 1)
        df['roc_20'] = 100 * (close / close.shift(20) - 1)
        
        # Williams %R
        df['williams_r'] = -100 * ((high_14 - close) / (high_14 - low_14 + 1e-8))
        
        # CCI
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(20, min_periods=10).mean()
        mad = (tp - sma_tp).abs().rolling(20, min_periods=10).mean()
        df['cci'] = (tp - sma_tp) / (0.015 * mad + 1e-8)
        
        # MFI
        volume = df['Volume']
        mf = tp * volume
        pos_mf = mf.where(tp > tp.shift(), 0).rolling(14, min_periods=7).sum()
        neg_mf = mf.where(tp < tp.shift(), 0).rolling(14, min_periods=7).sum()
        mf_ratio = pos_mf / (neg_mf + 1e-8)
        df['mfi'] = 100 - (100 / (1 + mf_ratio))
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Indicateurs de volatilité"""
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # Bollinger Bands
        sma_20 = close.rolling(20, min_periods=10).mean()
        std_20 = close.rolling(20, min_periods=10).std()
        df['bb_upper'] = sma_20 + (2 * std_20)
        df['bb_lower'] = sma_20 - (2 * std_20)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (sma_20 + 1e-8)
        df['bb_pct'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        
        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14, min_periods=7).mean()
        
        # Normalized ATR
        df['natr'] = 100 * (df['atr'] / (close + 1e-8))
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Indicateurs de volume"""
        close = df['Close']
        volume = df['Volume']
        high = df['High']
        low = df['Low']
        
        # OBV
        obv = (volume * ((close - close.shift()).apply(np.sign))).fillna(0).cumsum()
        df['obv'] = obv
        df['obv_ema'] = obv.ewm(span=20, min_periods=10).mean()
        
        # CMF
        mf_multiplier = ((close - low) - (high - close)) / (high - low + 1e-8)
        mf_volume = mf_multiplier * volume
        df['cmf'] = mf_volume.rolling(20, min_periods=10).sum() / (volume.rolling(20, min_periods=10).sum() + 1e-8)
        
        # Volume ratio
        vol_sma = volume.rolling(20, min_periods=10).mean()
        df['volume_ratio'] = volume / (vol_sma + 1e-8)
        
        # VWAP (approximation)
        tp = (high + low + close) / 3
        df['vwap'] = (tp * volume).rolling(20, min_periods=10).sum() / (volume.rolling(20, min_periods=10).sum() + 1e-8)
        
        return df
    
    def _add_price_action(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price action patterns"""
        open_ = df['Open']
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # Returns
        df['returns'] = close.pct_change()
        df['returns_5'] = close.pct_change(5)
        df['returns_10'] = close.pct_change(10)
        
        # Body
        df['body'] = (close - open_).abs() / (close + 1e-8)
        
        # Shadows
        df['upper_shadow'] = (high - close.combine(open_, max)) / (high + 1e-8)
        df['lower_shadow'] = (close.combine(open_, min) - low) / (low + 1e-8)
        
        # Gap
        df['gap'] = (open_ - close.shift()) / (close.shift() + 1e-8)
        
        # Range
        df['range'] = (high - low) / (low + 1e-8)
        
        # Trend strength
        df['trend'] = (close - close.rolling(50, min_periods=25).mean()) / (close + 1e-8)
        
        return df
    
    def _get_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback: features basiques seulement"""
        close = df['Close']
        
        df['returns'] = close.pct_change().fillna(0)
        df['sma_10'] = close.rolling(10, min_periods=5).mean()
        df['sma_20'] = close.rolling(20, min_periods=10).mean()
        df['volatility'] = df['returns'].rolling(20, min_periods=10).std().fillna(0)
        
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=7).mean()
        loss = -delta.where(delta < 0, 0).rolling(14, min_periods=7).mean()
        rs = gain / (loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        df = self._clean_features(df)
        
        return df
    
    def get_feature_names(self) -> list:
        """Retourner la liste des noms de features"""
        return [
            # Trend
            'sma_10', 'sma_20', 'sma_50', 'ema_10', 'ema_20',
            'macd', 'macd_signal', 'macd_diff', 'adx',
            # Momentum
            'rsi', 'stoch_k', 'stoch_d', 'roc_10', 'roc_20',
            'williams_r', 'cci', 'mfi',
            # Volatility
            'bb_upper', 'bb_lower', 'bb_width', 'bb_pct',
            'atr', 'natr',
            # Volume
            'obv', 'obv_ema', 'cmf', 'volume_ratio', 'vwap',
            # Price Action
            'returns', 'returns_5', 'returns_10',
            'body', 'upper_shadow', 'lower_shadow',
            'gap', 'range', 'trend'
        ]
