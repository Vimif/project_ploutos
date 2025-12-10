# core/advanced_features_v2.py
"""🚀 FEATURES V2 - Optimisées pour détecter BONS POINTS D'ENTRÉE

Problème identifié: IA achète trop tard (85% buy high)

Solution: Features qui détectent:
- Support/Resistance dynamiques
- Début de mouvement (pas fin)
- Mean reversion opportunities
- Volume confirmation
- Divergences RSI/Prix

Auteur: Ploutos AI Team
Date: Dec 2025
"""

import numpy as np
import pandas as pd
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeaturesV2:
    """
    Features avancées V2 pour timing optimal
    """
    
    def __init__(self):
        self.features_calculated = []
    
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule TOUTES les features optimisées
        """
        df = df.copy()
        
        # 1. Support/Resistance
        df = self._calculate_support_resistance(df)
        
        # 2. Mean Reversion
        df = self._calculate_mean_reversion(df)
        
        # 3. Volume Patterns
        df = self._calculate_volume_patterns(df)
        
        # 4. Price Action
        df = self._calculate_price_action(df)
        
        # 5. Divergences
        df = self._calculate_divergences(df)
        
        # 6. Bollinger Patterns
        df = self._calculate_bollinger_patterns(df)
        
        # 7. Entry Score Composite
        df = self._calculate_entry_score(df)
        
        # 8. Momentum (amélioré)
        df = self._calculate_enhanced_momentum(df)
        
        # 9. Trend Strength
        df = self._calculate_trend_strength(df)
        
        # 10. Volatility Regime
        df = self._calculate_volatility_regime(df)
        
        # Cleanup
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(0)
        
        return df
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """✅ Support/Resistance DYNAMIQUES
        
        Détecte niveaux clés pour identifier bons points d'entrée
        """
        # Lookback windows
        windows = [20, 50, 100]
        
        for w in windows:
            if len(df) < w:
                df[f'support_{w}'] = df['Low'].min()
                df[f'resistance_{w}'] = df['High'].max()
            else:
                # Support = min local
                df[f'support_{w}'] = df['Low'].rolling(w, min_periods=1).min()
                # Resistance = max local
                df[f'resistance_{w}'] = df['High'].rolling(w, min_periods=1).max()
            
            # Distance actuelle vs support/resistance
            df[f'dist_support_{w}'] = (df['Close'] - df[f'support_{w}']) / df['Close']
            df[f'dist_resistance_{w}'] = (df[f'resistance_{w}'] - df['Close']) / df['Close']
            
            # Signal: proche support = BUY opportunity
            df[f'near_support_{w}'] = (df[f'dist_support_{w}'] < 0.02).astype(int)  # <2%
            df[f'near_resistance_{w}'] = (df[f'dist_resistance_{w}'] < 0.02).astype(int)
        
        return df
    
    def _calculate_mean_reversion(self, df: pd.DataFrame) -> pd.DataFrame:
        """✅ MEAN REVERSION signals
        
        Détecte quand prix s'éloigne trop de la moyenne = opportunité
        """
        windows = [20, 50]
        
        for w in windows:
            # Moving average
            df[f'ma_{w}'] = df['Close'].rolling(w, min_periods=1).mean()
            
            # Std dev
            df[f'std_{w}'] = df['Close'].rolling(w, min_periods=1).std()
            
            # Z-score (distance en écart-types)
            df[f'zscore_{w}'] = (df['Close'] - df[f'ma_{w}']) / (df[f'std_{w}'] + 1e-8)
            
            # Signal: z-score < -1.5 = oversold = BUY
            df[f'oversold_{w}'] = (df[f'zscore_{w}'] < -1.5).astype(int)
            df[f'overbought_{w}'] = (df[f'zscore_{w}'] > 1.5).astype(int)
            
            # Reversion signal: prix commence à revenir vers moyenne
            df[f'reverting_{w}'] = (
                (df[f'zscore_{w}'].shift(1) < df[f'zscore_{w}'].shift(2)) &  # Z-score remonte
                (df[f'zscore_{w}'] < -1.0)  # Toujours oversold
            ).astype(int)
        
        return df
    
    def _calculate_volume_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """✅ VOLUME confirmation
        
        Volume confirme la force du mouvement
        """
        # Volume moyenne
        df['vol_ma_20'] = df['Volume'].rolling(20, min_periods=1).mean()
        df['vol_ma_50'] = df['Volume'].rolling(50, min_periods=1).mean()
        
        # Volume ratio
        df['vol_ratio'] = df['Volume'] / (df['vol_ma_20'] + 1e-8)
        
        # Volume spike = signal fort
        df['vol_spike'] = (df['vol_ratio'] > 1.5).astype(int)
        
        # Volume + prix monte = bullish confirmation
        df['vol_bullish'] = (
            (df['vol_ratio'] > 1.2) & 
            (df['Close'] > df['Close'].shift(1))
        ).astype(int)
        
        # Volume + prix baisse = bearish confirmation
        df['vol_bearish'] = (
            (df['vol_ratio'] > 1.2) & 
            (df['Close'] < df['Close'].shift(1))
        ).astype(int)
        
        # Low volume = manque conviction
        df['vol_low'] = (df['vol_ratio'] < 0.7).astype(int)
        
        return df
    
    def _calculate_price_action(self, df: pd.DataFrame) -> pd.DataFrame:
        """✅ PRICE ACTION patterns
        
        Patterns de chandeliers pour détecter reversals
        """
        # Body size
        df['body'] = abs(df['Close'] - df['Open'])
        df['body_pct'] = df['body'] / df['Open']
        
        # Wicks
        df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        
        # Hammer (bullish reversal)
        df['hammer'] = (
            (df['lower_wick'] > df['body'] * 2) &  # Long lower wick
            (df['upper_wick'] < df['body'] * 0.5) &  # Small upper wick
            (df['Close'] > df['Open'])  # Green candle
        ).astype(int)
        
        # Shooting star (bearish reversal)
        df['shooting_star'] = (
            (df['upper_wick'] > df['body'] * 2) &
            (df['lower_wick'] < df['body'] * 0.5) &
            (df['Close'] < df['Open'])
        ).astype(int)
        
        # Doji (indecision)
        df['doji'] = (df['body_pct'] < 0.001).astype(int)
        
        # Engulfing patterns
        df['bullish_engulfing'] = (
            (df['Close'] > df['Open']) &  # Green today
            (df['Close'].shift(1) < df['Open'].shift(1)) &  # Red yesterday
            (df['Close'] > df['Open'].shift(1)) &  # Close above yesterday open
            (df['Open'] < df['Close'].shift(1))  # Open below yesterday close
        ).astype(int)
        
        df['bearish_engulfing'] = (
            (df['Close'] < df['Open']) &
            (df['Close'].shift(1) > df['Open'].shift(1)) &
            (df['Close'] < df['Open'].shift(1)) &
            (df['Open'] > df['Close'].shift(1))
        ).astype(int)
        
        return df
    
    def _calculate_divergences(self, df: pd.DataFrame) -> pd.DataFrame:
        """✅ DIVERGENCES RSI/Prix
        
        Divergence = signal fort de reversal
        """
        # RSI
        period = 14
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period, min_periods=1).mean()
        rs = gain / (loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Lookback pour divergence
        lookback = 10
        
        # Bullish divergence: prix fait lower low, RSI fait higher low
        df['price_lower_low'] = (
            df['Close'] < df['Close'].shift(lookback)
        ).astype(int)
        
        df['rsi_higher_low'] = (
            df['rsi'] > df['rsi'].shift(lookback)
        ).astype(int)
        
        df['bullish_divergence'] = (
            df['price_lower_low'] & df['rsi_higher_low']
        ).astype(int)
        
        # Bearish divergence: prix fait higher high, RSI fait lower high
        df['price_higher_high'] = (
            df['Close'] > df['Close'].shift(lookback)
        ).astype(int)
        
        df['rsi_lower_high'] = (
            df['rsi'] < df['rsi'].shift(lookback)
        ).astype(int)
        
        df['bearish_divergence'] = (
            df['price_higher_high'] & df['rsi_lower_high']
        ).astype(int)
        
        return df
    
    def _calculate_bollinger_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """✅ BOLLINGER BANDS patterns
        
        Squeeze, breakout, etc.
        """
        period = 20
        std_mult = 2
        
        # Bollinger Bands
        df['bb_mid'] = df['Close'].rolling(period, min_periods=1).mean()
        df['bb_std'] = df['Close'].rolling(period, min_periods=1).std()
        df['bb_upper'] = df['bb_mid'] + (std_mult * df['bb_std'])
        df['bb_lower'] = df['bb_mid'] - (std_mult * df['bb_std'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        
        # Position dans les bandes
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        
        # Touche bande basse = BUY signal
        df['touch_lower_bb'] = (df['Close'] <= df['bb_lower'] * 1.01).astype(int)
        
        # Touche bande haute = SELL signal
        df['touch_upper_bb'] = (df['Close'] >= df['bb_upper'] * 0.99).astype(int)
        
        # Squeeze: bandes se resserrent = breakout imminent
        df['bb_squeeze'] = (
            df['bb_width'] < df['bb_width'].rolling(50, min_periods=1).quantile(0.2)
        ).astype(int)
        
        return df
    
    def _calculate_entry_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """✅ ENTRY SCORE composite
        
        Combine tous les signaux en un score d'entrée
        """
        # BUY score
        buy_signals = [
            'near_support_20', 'near_support_50',
            'oversold_20', 'oversold_50',
            'reverting_20', 'reverting_50',
            'vol_bullish',
            'hammer', 'bullish_engulfing',
            'bullish_divergence',
            'touch_lower_bb'
        ]
        
        # Compter signaux BUY présents
        df['buy_score'] = sum([df.get(col, 0) for col in buy_signals if col in df.columns])
        
        # Normaliser 0-1
        max_signals = len([col for col in buy_signals if col in df.columns])
        if max_signals > 0:
            df['buy_score_norm'] = df['buy_score'] / max_signals
        else:
            df['buy_score_norm'] = 0
        
        # SELL score
        sell_signals = [
            'near_resistance_20', 'near_resistance_50',
            'overbought_20', 'overbought_50',
            'shooting_star', 'bearish_engulfing',
            'bearish_divergence',
            'touch_upper_bb'
        ]
        
        df['sell_score'] = sum([df.get(col, 0) for col in sell_signals if col in df.columns])
        
        max_signals = len([col for col in sell_signals if col in df.columns])
        if max_signals > 0:
            df['sell_score_norm'] = df['sell_score'] / max_signals
        else:
            df['sell_score_norm'] = 0
        
        # Signal net: buy - sell
        df['entry_signal'] = df['buy_score_norm'] - df['sell_score_norm']
        
        return df
    
    def _calculate_enhanced_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """✅ Momentum AMÉLIORÉ
        
        Détecte DÉBUT de momentum (pas fin)
        """
        # Rate of change
        periods = [5, 10, 20]
        for p in periods:
            df[f'roc_{p}'] = df['Close'].pct_change(p)
            
            # Accélération momentum
            df[f'momentum_accel_{p}'] = df[f'roc_{p}'].diff()
            
            # Début momentum = accélération positive + momentum faible
            df[f'momentum_start_{p}'] = (
                (df[f'momentum_accel_{p}'] > 0) &
                (abs(df[f'roc_{p}']) < 0.05)  # Pas encore trop haut
            ).astype(int)
        
        return df
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """✅ Force du TREND"""
        # ADX (Average Directional Index)
        period = 14
        
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period, min_periods=1).mean()
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        df['plus_di'] = 100 * pd.Series(plus_dm).rolling(period, min_periods=1).mean() / (atr + 1e-8)
        df['minus_di'] = 100 * pd.Series(minus_dm).rolling(period, min_periods=1).mean() / (atr + 1e-8)
        
        # ADX
        dx = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'] + 1e-8)
        df['adx'] = dx.rolling(period, min_periods=1).mean()
        
        # Trend fort = ADX > 25
        df['strong_trend'] = (df['adx'] > 25).astype(int)
        df['weak_trend'] = (df['adx'] < 20).astype(int)
        
        return df
    
    def _calculate_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """✅ Régime de VOLATILITÉ"""
        # ATR
        period = 14
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(period, min_periods=1).mean()
        df['atr_pct'] = df['atr'] / df['Close']
        
        # Régime volatility
        df['high_vol'] = (df['atr_pct'] > df['atr_pct'].rolling(50, min_periods=1).quantile(0.7)).astype(int)
        df['low_vol'] = (df['atr_pct'] < df['atr_pct'].rolling(50, min_periods=1).quantile(0.3)).astype(int)
        
        return df


if __name__ == '__main__':
    # Test
    print("🧪 Test Features V2...")
    
    # Créer données test
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    df_test = pd.DataFrame({
        'Open': 100 + np.random.randn(100).cumsum(),
        'High': 101 + np.random.randn(100).cumsum(),
        'Low': 99 + np.random.randn(100).cumsum(),
        'Close': 100 + np.random.randn(100).cumsum(),
        'Volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Calculer features
    fe = AdvancedFeaturesV2()
    df_result = fe.calculate_all_features(df_test)
    
    print(f"\n✅ Features calculées: {len(df_result.columns)}")
    print(f"\nColonnes: {list(df_result.columns)}")
    print(f"\nEntry signals:")
    print(df_result[['Close', 'buy_score', 'sell_score', 'entry_signal']].tail(10))
