# core/advanced_features.py
"""Features Engineering Avancé pour Trading - 50+ Indicateurs"""

import numpy as np
import pandas as pd
from typing import Dict
import ta  # Technical Analysis library
from scipy.stats import skew, kurtosis


class AdvancedFeatureEngineering:
    """Calculateur de features techniques avancées"""
    
    def __init__(self):
        self.feature_names = []
    
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculer toutes les features pour un DataFrame OHLCV
        
        Args:
            df: DataFrame avec colonnes ['Open', 'High', 'Low', 'Close', 'Volume']
        
        Returns:
            DataFrame avec toutes les features
        """
        features = df.copy()
        
        # 1. TREND INDICATORS (10 features)
        features = self._add_trend_indicators(features)
        
        # 2. MOMENTUM INDICATORS (12 features)
        features = self._add_momentum_indicators(features)
        
        # 3. VOLATILITY INDICATORS (8 features)
        features = self._add_volatility_indicators(features)
        
        # 4. VOLUME INDICATORS (8 features)
        features = self._add_volume_indicators(features)
        
        # 5. SUPPORT/RESISTANCE (6 features)
        features = self._add_support_resistance(features)
        
        # 6. STATISTICAL FEATURES (8 features)
        features = self._add_statistical_features(features)
        
        # 7. PRICE ACTION (6 features)
        features = self._add_price_action(features)
        
        # Remplir NaN avec forward fill puis 0
        features = features.ffill().fillna(0)
        
        return features
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Indicateurs de tendance"""
        # SMA
        df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        
        # EMA
        df['EMA_10'] = ta.trend.ema_indicator(df['Close'], window=10)
        df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()
        
        # ADX (trend strength)
        df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Indicateurs de momentum"""
        # RSI
        df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
        df['RSI_7'] = ta.momentum.rsi(df['Close'], window=7)
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # ROC (Rate of Change)
        df['ROC_10'] = ta.momentum.roc(df['Close'], window=10)
        df['ROC_20'] = ta.momentum.roc(df['Close'], window=20)
        
        # Williams %R
        df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
        
        # TSI (True Strength Index)
        df['TSI'] = ta.momentum.tsi(df['Close'])
        
        # Ultimate Oscillator
        df['UO'] = ta.momentum.ultimate_oscillator(df['High'], df['Low'], df['Close'])
        
        # CCI (Commodity Channel Index)
        df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
        
        # MFI (Money Flow Index)
        df['MFI'] = ta.volume.money_flow_index(
            df['High'], df['Low'], df['Close'], df['Volume']
        )
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Indicateurs de volatilité"""
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_high'] = bollinger.bollinger_hband()
        df['BB_low'] = bollinger.bollinger_lband()
        df['BB_width'] = bollinger.bollinger_wband()
        df['BB_pct'] = bollinger.bollinger_pband()
        
        # ATR (Average True Range)
        df['ATR_14'] = ta.volatility.average_true_range(
            df['High'], df['Low'], df['Close'], window=14
        )
        
        # Keltner Channel
        keltner = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'])
        df['Keltner_high'] = keltner.keltner_channel_hband()
        df['Keltner_low'] = keltner.keltner_channel_lband()
        
        # Ulcer Index
        df['Ulcer_Index'] = ta.volatility.ulcer_index(df['Close'])
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Indicateurs de volume"""
        # OBV (On Balance Volume)
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        
        # CMF (Chaikin Money Flow)
        df['CMF'] = ta.volume.chaikin_money_flow(
            df['High'], df['Low'], df['Close'], df['Volume']
        )
        
        # Force Index
        df['Force_Index'] = ta.volume.force_index(df['Close'], df['Volume'])
        
        # Volume Price Trend
        df['VPT'] = ta.volume.volume_price_trend(df['Close'], df['Volume'])
        
        # Ease of Movement
        df['EOM'] = ta.volume.ease_of_movement(
            df['High'], df['Low'], df['Volume']
        )
        
        # Volume Ratio
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / (df['Volume_SMA'] + 1e-8)
        
        # VWAP (Volume Weighted Average Price) - approximation
        df['VWAP'] = (df['Close'] * df['Volume']).rolling(window=20).sum() / \
                     (df['Volume'].rolling(window=20).sum() + 1e-8)
        
        return df
    
    def _add_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Niveaux de support/résistance"""
        # Pivot Points
        df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['R1'] = 2 * df['Pivot'] - df['Low']
        df['S1'] = 2 * df['Pivot'] - df['High']
        df['R2'] = df['Pivot'] + (df['High'] - df['Low'])
        df['S2'] = df['Pivot'] - (df['High'] - df['Low'])
        
        # Distance from 52-week high/low
        df['High_52w'] = df['High'].rolling(window=252).max()
        df['Low_52w'] = df['Low'].rolling(window=252).min()
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features statistiques"""
        returns = df['Close'].pct_change()
        
        # Rolling Statistics
        df['Returns_Mean_20'] = returns.rolling(window=20).mean()
        df['Returns_Std_20'] = returns.rolling(window=20).std()
        df['Returns_Skew_20'] = returns.rolling(window=20).apply(skew, raw=True)
        df['Returns_Kurt_20'] = returns.rolling(window=20).apply(kurtosis, raw=True)
        
        # Z-Score
        df['Price_ZScore'] = (
            df['Close'] - df['Close'].rolling(window=20).mean()
        ) / (df['Close'].rolling(window=20).std() + 1e-8)
        
        # Autocorrelation
        df['Autocorr_1'] = returns.rolling(window=20).apply(
            lambda x: x.autocorr(lag=1), raw=False
        )
        
        # Hurst Exponent (trend persistence)
        df['Hurst_20'] = self._calculate_hurst(df['Close'], window=20)
        
        # Variance Ratio
        df['Var_Ratio'] = returns.rolling(window=10).var() / \
                         (returns.rolling(window=20).var() + 1e-8)
        
        return df
    
    def _add_price_action(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price action patterns"""
        # Candle body size
        df['Body_Size'] = abs(df['Close'] - df['Open']) / (df['Close'] + 1e-8)
        
        # Upper/Lower shadows
        df['Upper_Shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / \
                            (df['High'] + 1e-8)
        df['Lower_Shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / \
                            (df['Low'] + 1e-8)
        
        # Gap
        df['Gap'] = (df['Open'] - df['Close'].shift(1)) / (df['Close'].shift(1) + 1e-8)
        
        # True Range
        df['True_Range'] = df[['High', 'Close']].max(axis=1) - \
                          df[['Low', 'Close']].min(axis=1)
        
        # Daily range
        df['Daily_Range'] = (df['High'] - df['Low']) / (df['Low'] + 1e-8)
        
        return df
    
    def _calculate_hurst(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Calculer l'exposant de Hurst (simplifié)"""
        def hurst_window(x):
            if len(x) < 10:
                return 0.5
            try:
                lags = range(2, min(10, len(x)//2))
                tau = [np.std(np.subtract(x[lag:], x[:-lag])) for lag in lags]
                return np.polyfit(np.log(lags), np.log(tau), 1)[0]
            except:
                return 0.5
        
        return series.rolling(window=window).apply(hurst_window, raw=True)
    
    def get_feature_names(self) -> list:
        """Retourner la liste des noms de features"""
        return [
            # Trend
            'SMA_10', 'SMA_20', 'SMA_50', 'EMA_10', 'EMA_20',
            'MACD', 'MACD_signal', 'MACD_diff', 'ADX',
            # Momentum
            'RSI_14', 'RSI_7', 'Stoch_K', 'Stoch_D', 'ROC_10', 'ROC_20',
            'Williams_R', 'TSI', 'UO', 'CCI', 'MFI',
            # Volatility
            'BB_high', 'BB_low', 'BB_width', 'BB_pct', 'ATR_14',
            'Keltner_high', 'Keltner_low', 'Ulcer_Index',
            # Volume
            'OBV', 'CMF', 'Force_Index', 'VPT', 'EOM',
            'Volume_Ratio', 'VWAP',
            # Support/Resistance
            'Pivot', 'R1', 'S1', 'R2', 'S2', 'High_52w', 'Low_52w',
            # Statistical
            'Returns_Mean_20', 'Returns_Std_20', 'Returns_Skew_20',
            'Returns_Kurt_20', 'Price_ZScore', 'Autocorr_1', 'Hurst_20',
            'Var_Ratio',
            # Price Action
            'Body_Size', 'Upper_Shadow', 'Lower_Shadow', 'Gap',
            'True_Range', 'Daily_Range'
        ]


def add_market_regime_features(df: pd.DataFrame, vix_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    Ajouter features de régime de marché
    
    Args:
        df: DataFrame principal
        vix_data: DataFrame avec VIX (optionnel)
    
    Returns:
        DataFrame avec features de régime
    """
    # Volatilité réalisée
    returns = df['Close'].pct_change()
    df['Realized_Vol_20'] = returns.rolling(window=20).std() * np.sqrt(252)
    df['Realized_Vol_60'] = returns.rolling(window=60).std() * np.sqrt(252)
    
    # Trend strength
    df['Trend_Strength'] = abs(
        df['Close'].rolling(window=20).mean() - df['Close'].rolling(window=50).mean()
    ) / df['Close']
    
    # Market regime (bull/bear/sideways)
    sma_20 = df['Close'].rolling(window=20).mean()
    sma_50 = df['Close'].rolling(window=50).mean()
    
    df['Bull_Regime'] = (sma_20 > sma_50).astype(int)
    df['Bear_Regime'] = (sma_20 < sma_50).astype(int)
    
    # VIX si disponible
    if vix_data is not None:
        df = df.join(vix_data[['Close']].rename(columns={'Close': 'VIX'}), how='left')
        df['VIX'] = df['VIX'].ffill()
    
    return df
