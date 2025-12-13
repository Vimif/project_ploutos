#!/usr/bin/env python3
"""
🔥 PLOUTOS V8 ORACLE - XGBOOST WEEKLY PREDICTOR

Modèle moyen terme (5 jours) basé sur XGBoost
Optimisé pour les tendances hebdomadaires

Performance cible: 65-75% accuracy
Horizon: 5 jours (1 semaine de trading)

Auteur: Ploutos AI Team
Date: Dec 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Tuple, Optional
import yfinance as yf
from datetime import datetime
import pickle
import ta


class V8XGBoostWeekly:
    """
    Prédicteur moyen terme (5 jours) avec XGBoost
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée 35+ features pour analyse hebdomadaire
        """
        df = df.copy()
        
        # --- TREND INDICATORS (focus moyen terme) ---
        
        # Moving Averages (plus de périodes)
        for period in [10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = ta.trend.sma_indicator(df['Close'], window=period)
            df[f'ema_{period}'] = ta.trend.ema_indicator(df['Close'], window=period)
        
        # Distance to key MAs
        for period in [10, 20, 50, 100]:
            df[f'dist_sma_{period}'] = (df['Close'] - df[f'sma_{period}']) / df[f'sma_{period}']
        
        # MA crossovers
        df['sma_10_20_cross'] = (df['sma_10'] > df['sma_20']).astype(int)
        df['sma_20_50_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
        df['sma_50_200_cross'] = (df['sma_50'] > df['sma_200']).astype(int)
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        df['macd_positive'] = (df['macd'] > 0).astype(int)
        
        # ADX (Trend Strength)
        df['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
        df['adx_pos'] = ta.trend.adx_pos(df['High'], df['Low'], df['Close'], window=14)
        df['adx_neg'] = ta.trend.adx_neg(df['High'], df['Low'], df['Close'], window=14)
        
        # Ichimoku Cloud
        ichimoku = ta.trend.IchimokuIndicator(df['High'], df['Low'])
        df['ichimoku_a'] = ichimoku.ichimoku_a()
        df['ichimoku_b'] = ichimoku.ichimoku_b()
        df['ichimoku_base'] = ichimoku.ichimoku_base_line()
        df['ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
        
        # --- SUPPORT / RESISTANCE ---
        
        # Pivot points
        df['pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['resistance_1'] = 2 * df['pivot'] - df['Low']
        df['support_1'] = 2 * df['pivot'] - df['High']
        
        # Distance to pivot
        df['dist_pivot'] = (df['Close'] - df['pivot']) / df['pivot']
        
        # --- MOMENTUM (weekly focus) ---
        
        # RSI
        df['rsi_14'] = ta.momentum.rsi(df['Close'], window=14)
        df['rsi_21'] = ta.momentum.rsi(df['Close'], window=21)
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # --- VOLATILITY ---
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'], window=20)
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['Close']
        
        # ATR
        df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        df['atr_ratio'] = df['atr'] / df['Close']
        
        # --- VOLUME ---
        
        # VWAP approximation
        df['vwap'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        df['dist_vwap'] = (df['Close'] - df['vwap']) / df['vwap']
        
        # Volume trend
        df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
        
        # OBV
        df['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['obv_ema'] = ta.trend.ema_indicator(df['obv'], window=20)
        
        # --- PRICE ACTION ---
        
        # Weekly returns
        df['return_5d'] = df['Close'].pct_change(5)
        df['return_10d'] = df['Close'].pct_change(10)
        df['return_20d'] = df['Close'].pct_change(20)
        
        # Volatility (weekly)
        df['volatility_5d'] = df['Close'].pct_change().rolling(window=5).std()
        df['volatility_20d'] = df['Close'].pct_change().rolling(window=20).std()
        
        # High-Low range
        df['hl_range'] = (df['High'] - df['Low']) / df['Close']
        
        # --- TIME FEATURES ---
        
        df['day_of_week'] = df.index.dayofweek
        df['week_of_month'] = (df.index.day - 1) // 7
        df['month'] = df.index.month
        
        return df
    
    def create_labels(self, df: pd.DataFrame, horizon: int = 5) -> pd.Series:
        """
        Labels pour horizon 5 jours
        """
        future_price = df['Close'].shift(-horizon)
        current_price = df['Close']
        labels = (future_price > current_price).astype(int)
        return labels
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prépare features et labels
        """
        df_features = self.create_features(df)
        labels = self.create_labels(df_features, horizon=5)
        
        feature_cols = [
            col for col in df_features.columns 
            if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        ]
        
        X = df_features[feature_cols]
        y = labels
        
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        self.feature_names = feature_cols
        return X, y
    
    def train(self, tickers: list, start_date: str, end_date: str,
             test_size: float = 0.2, params: Optional[Dict] = None):
        """
        Entraîne XGBoost
        """
        print("\n" + "="*70)
        print("🔥 PLOUTOS V8 ORACLE - ENTRAINEMENT XGBOOST WEEKLY")
        print("="*70)
        print(f"\n📅 Période: {start_date} à {end_date}")
        print(f"🎯 Tickers: {', '.join(tickers)}")
        print(f"📊 Horizon: 5 jours\n")
        
        all_X = []
        all_y = []
        
        for ticker in tickers:
            print(f"📡 Chargement {ticker}...", end=" ")
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                if len(df) < 200:
                    print("⚠️  Données insuffisantes")
                    continue
                
                X, y = self.prepare_data(df)
                all_X.append(X)
                all_y.append(y)
                print(f"✅ {len(X)} samples")
                
            except Exception as e:
                print(f"❌ {e}")
        
        X_all = pd.concat(all_X, axis=0)
        y_all = pd.concat(all_y, axis=0)
        
        print(f"\n📊 Total: {len(X_all):,} samples")
        
        # Split temporel
        split_idx = int(len(X_all) * (1 - test_size))
        X_train = X_all.iloc[:split_idx]
        y_train = y_all.iloc[:split_idx]
        X_test = X_all.iloc[split_idx:]
        y_test = y_all.iloc[split_idx:]
        
        # Paramètres XGBoost
        if params is None:
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'learning_rate': 0.03,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'tree_method': 'hist',
                'verbosity': 0
            }
        
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=self.feature_names)
        
        evals = [(dtrain, 'train'), (dtest, 'test')]
        
        print("\n🚀 Entraînement...\n")
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=evals,
            early_stopping_rounds=30,
            verbose_eval=50
        )
        
        # Évaluation
        print("\n" + "="*70)
        print("📊 ÉVALUATION")
        print("="*70)
        
        y_train_pred = (self.model.predict(dtrain) > 0.5).astype(int)
        y_test_pred = (self.model.predict(dtest) > 0.5).astype(int)
        
        train_acc = (y_train_pred == y_train).mean()
        test_acc = (y_test_pred == y_test).mean()
        
        print(f"\n🎯 Accuracy Train: {train_acc*100:.2f}%")
        print(f"🎯 Accuracy Test: {test_acc*100:.2f}%")
        
        # Feature importance
        print("\n🔥 Top 10 Features:")
        importance = self.model.get_score(importance_type='gain')
        feature_importance = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance.items()
        ]).sort_values('importance', ascending=False)
        
        for idx, row in feature_importance.head(10).iterrows():
            print(f"   {row['feature']:25s}: {row['importance']:.0f}")
        
        print("\n✅ ENTRAINEMENT TERMINÉ\n")
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'feature_importance': feature_importance
        }
    
    def predict(self, ticker: str, period: str = "1y") -> Dict:
        """
        Prédit pour 5 jours
        """
        if self.model is None:
            return {'error': 'Modèle non entraîné'}
        
        try:
            df = yf.download(ticker, period=period, progress=False)
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            if len(df) < 200:
                return {'error': 'Données insuffisantes'}
            
            df_features = self.create_features(df)
            X_latest = df_features[self.feature_names].iloc[-1:]
            
            if X_latest.isna().any().any():
                return {'error': 'Features manquantes'}
            
            dlatest = xgb.DMatrix(X_latest, feature_names=self.feature_names)
            proba = self.model.predict(dlatest)[0]
            
            prediction = 'UP' if proba > 0.5 else 'DOWN'
            confidence = proba if prediction == 'UP' else (1 - proba)
            
            # Facteurs
            adx = df_features['adx'].iloc[-1]
            volume_ratio = df_features['volume_ratio'].iloc[-1]
            atr_ratio = df_features['atr_ratio'].iloc[-1]
            
            trend_bonus = min(0.15, max(0, (adx - 20) / 100))
            volume_bonus = min(0.1, max(0, (volume_ratio - 1) * 0.1))
            volatility_penalty = min(0.15, atr_ratio * 4)
            
            adjusted_confidence = confidence + trend_bonus + volume_bonus - volatility_penalty
            adjusted_confidence = np.clip(adjusted_confidence, 0, 1)
            
            return {
                'ticker': ticker,
                'horizon': '5 days',
                'prediction': prediction,
                'confidence': float(adjusted_confidence * 100),
                'raw_proba': float(proba),
                'factors': {
                    'model_confidence': float(confidence * 100),
                    'trend_adx': float(adx),
                    'volume_ratio': float(volume_ratio),
                    'volatility': float(atr_ratio)
                },
                'current_price': float(df['Close'].iloc[-1]),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def save(self, path: str = "models/v8_xgboost_weekly.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"💾 Sauvegardé: {path}")
    
    def load(self, path: str = "models/v8_xgboost_weekly.pkl") -> bool:
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            return True
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return False


if __name__ == '__main__':
    predictor = V8XGBoostWeekly()
    
    tickers = ['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'META', 'JPM', 'SPY', 'QQQ']
    
    results = predictor.train(
        tickers=tickers,
        start_date='2023-01-01',
        end_date='2024-12-01',
        test_size=0.2
    )
    
    predictor.save()
    
    print("\n🔮 Test NVDA:")
    print(predictor.predict('NVDA'))
