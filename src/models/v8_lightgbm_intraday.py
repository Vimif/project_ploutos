#!/usr/bin/env python3
"""
🔥 PLOUTOS V8 ORACLE - LIGHTGBM INTRADAY PREDICTOR

Modèle court terme (1 jour) basé sur LightGBM
Optimisé pour rapidité et robustesse

Performance cible: 65-75% accuracy
Inférence: <10ms par ticker

Auteur: Ploutos AI Team
Date: Dec 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Tuple, Optional
import yfinance as yf
from datetime import datetime, timedelta
import pickle
import ta


class V8LightGBMIntraday:
    """
    Prédicteur court terme (1 jour) avec LightGBM
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.scaler = None
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée 30+ features techniques
        """
        df = df.copy()
        
        # --- MOMENTUM INDICATORS ---
        
        # RSI (multiples périodes)
        df['rsi_7'] = ta.momentum.rsi(df['Close'], window=7)
        df['rsi_14'] = ta.momentum.rsi(df['Close'], window=14)
        df['rsi_21'] = ta.momentum.rsi(df['Close'], window=21)
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        df['williams_r'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
        
        # ROC (Rate of Change)
        df['roc'] = ta.momentum.roc(df['Close'], window=10)
        
        # --- TREND INDICATORS ---
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # ADX (Trend Strength)
        df['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
        
        # Moving Averages
        df['sma_10'] = ta.trend.sma_indicator(df['Close'], window=10)
        df['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['ema_10'] = ta.trend.ema_indicator(df['Close'], window=10)
        df['ema_20'] = ta.trend.ema_indicator(df['Close'], window=20)
        
        # Distance to MAs
        df['dist_sma_10'] = (df['Close'] - df['sma_10']) / df['sma_10']
        df['dist_sma_20'] = (df['Close'] - df['sma_20']) / df['sma_20']
        df['dist_sma_50'] = (df['Close'] - df['sma_50']) / df['sma_50']
        
        # --- VOLATILITY INDICATORS ---
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_mid'] = bollinger.bollinger_mavg()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        df['bb_position'] = (df['Close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
        
        # ATR (Average True Range)
        df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        df['atr_ratio'] = df['atr'] / df['Close']
        
        # --- VOLUME INDICATORS ---
        
        # OBV (On-Balance Volume)
        df['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['obv_ema'] = ta.trend.ema_indicator(df['obv'], window=20)
        
        # Volume ratio
        df['volume_sma'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        # Money Flow Index
        df['mfi'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Volume Price Trend
        df['vpt'] = ta.volume.volume_price_trend(df['Close'], df['Volume'])
        
        # --- PRICE ACTION ---
        
        # Returns
        df['return_1d'] = df['Close'].pct_change(1)
        df['return_5d'] = df['Close'].pct_change(5)
        df['return_10d'] = df['Close'].pct_change(10)
        
        # High-Low range
        df['hl_range'] = (df['High'] - df['Low']) / df['Close']
        
        # Close position in range
        df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # --- TIME FEATURES ---
        
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        
        return df
    
    def create_labels(self, df: pd.DataFrame, horizon: int = 1) -> pd.Series:
        """
        Crée les labels : 1 si prix monte dans 'horizon' jours, 0 sinon
        """
        future_price = df['Close'].shift(-horizon)
        current_price = df['Close']
        
        labels = (future_price > current_price).astype(int)
        return labels
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prépare les features et labels
        """
        # Créer features
        df_features = self.create_features(df)
        
        # Créer labels
        labels = self.create_labels(df_features, horizon=1)
        
        # Colonnes features (exclure OHLCV et labels)
        feature_cols = [
            col for col in df_features.columns 
            if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        ]
        
        X = df_features[feature_cols]
        y = labels
        
        # Supprimer NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        self.feature_names = feature_cols
        
        return X, y
    
    def train(self, tickers: list, start_date: str, end_date: str, 
             test_size: float = 0.2, params: Optional[Dict] = None):
        """
        Entraîne le modèle sur plusieurs tickers
        """
        print("\n" + "="*70)
        print("🔥 PLOUTOS V8 ORACLE - ENTRAINEMENT LIGHTGBM INTRADAY")
        print("="*70)
        print(f"\n📅 Période: {start_date} à {end_date}")
        print(f"🎯 Tickers: {', '.join(tickers)}")
        print(f"📊 Test size: {test_size*100}%\n")
        
        # Charger données
        all_X = []
        all_y = []
        
        for ticker in tickers:
            print(f"📡 Chargement {ticker}...", end=" ")
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                if len(df) < 100:
                    print("⚠️  Données insuffisantes")
                    continue
                
                X, y = self.prepare_data(df)
                all_X.append(X)
                all_y.append(y)
                
                print(f"✅ {len(X)} samples")
                
            except Exception as e:
                print(f"❌ Erreur: {e}")
        
        # Concaténer toutes les données
        X_all = pd.concat(all_X, axis=0)
        y_all = pd.concat(all_y, axis=0)
        
        print(f"\n📊 Total samples: {len(X_all):,}")
        print(f"🎯 Label distribution: UP={y_all.sum():,} ({y_all.mean()*100:.1f}%) / DOWN={len(y_all)-y_all.sum():,}")
        
        # Split temporel (pas shuffle!)
        split_idx = int(len(X_all) * (1 - test_size))
        X_train = X_all.iloc[:split_idx]
        y_train = y_all.iloc[:split_idx]
        X_test = X_all.iloc[split_idx:]
        y_test = y_all.iloc[split_idx:]
        
        print(f"\n🛠️ Train: {len(X_train):,} samples")
        print(f"🧪 Test: {len(X_test):,} samples")
        
        # Paramètres LightGBM
        if params is None:
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': 7,
                'min_data_in_leaf': 50,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'lambda_l1': 0.1,
                'lambda_l2': 1.0,
                'min_gain_to_split': 0.01,
                'verbose': -1
            }
        
        # Datasets LightGBM
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        test_data = lgb.Dataset(X_test, label=y_test, feature_name=self.feature_names, reference=train_data)
        
        # Entraînement
        print("\n🚀 Entraînement en cours...\n")
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, test_data],
            valid_names=['train', 'test'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50)
            ]
        )
        
        # Évaluation
        print("\n" + "="*70)
        print("📊 ÉVALUATION")
        print("="*70)
        
        # Prédictions
        y_train_pred = (self.model.predict(X_train) > 0.5).astype(int)
        y_test_pred = (self.model.predict(X_test) > 0.5).astype(int)
        
        train_acc = (y_train_pred == y_train).mean()
        test_acc = (y_test_pred == y_test).mean()
        
        print(f"\n🎯 Accuracy Train: {train_acc*100:.2f}%")
        print(f"🎯 Accuracy Test: {test_acc*100:.2f}%")
        
        # Feature importance
        print("\n🔥 Top 10 Features:")
        importance = self.model.feature_importance(importance_type='gain')
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        for idx, row in feature_importance.head(10).iterrows():
            print(f"   {row['feature']:20s}: {row['importance']:.0f}")
        
        print("\n" + "="*70)
        print("✅ ENTRAINEMENT TERMINÉ")
        print("="*70 + "\n")
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'feature_importance': feature_importance
        }
    
    def predict(self, ticker: str, period: str = "1y") -> Dict:
        """
        Prédit la direction du prix pour 1 jour
        """
        if self.model is None:
            return {'error': 'Modèle non entraîné'}
        
        try:
            # Charger données
            df = yf.download(ticker, period=period, progress=False)
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            if len(df) < 60:
                return {'error': 'Données insuffisantes'}
            
            # Créer features
            df_features = self.create_features(df)
            
            # Dernière ligne (aujourd'hui)
            X_latest = df_features[self.feature_names].iloc[-1:]
            
            if X_latest.isna().any().any():
                return {'error': 'Features manquantes'}
            
            # Prédiction
            proba = self.model.predict(X_latest)[0]
            prediction = 'UP' if proba > 0.5 else 'DOWN'
            confidence = proba if prediction == 'UP' else (1 - proba)
            
            # Calculer facteurs de confiance
            latest_data = df.iloc[-1]
            
            # Volatility penalty
            atr_ratio = df_features['atr_ratio'].iloc[-1]
            volatility_penalty = min(0.2, atr_ratio * 5)
            
            # Volume bonus
            volume_ratio = df_features['volume_ratio'].iloc[-1]
            volume_bonus = min(0.1, max(0, (volume_ratio - 1) * 0.1))
            
            # Trend bonus (ADX)
            adx = df_features['adx'].iloc[-1]
            trend_bonus = min(0.15, max(0, (adx - 20) / 100))
            
            # Confidence ajustée
            adjusted_confidence = confidence - volatility_penalty + volume_bonus + trend_bonus
            adjusted_confidence = np.clip(adjusted_confidence, 0, 1)
            
            return {
                'ticker': ticker,
                'prediction': prediction,
                'confidence': float(adjusted_confidence * 100),
                'raw_proba': float(proba),
                'factors': {
                    'model_confidence': float(confidence * 100),
                    'volatility': float(atr_ratio),
                    'volume_ratio': float(volume_ratio),
                    'trend_adx': float(adx)
                },
                'current_price': float(latest_data['Close']),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def save(self, path: str = "models/v8_lightgbm_intraday.pkl"):
        """
        Sauvegarde le modèle
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Modèle sauvegardé: {path}")
    
    def load(self, path: str = "models/v8_lightgbm_intraday.pkl") -> bool:
        """
        Charge le modèle
        """
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            
            return True
        except Exception as e:
            print(f"❌ Erreur chargement: {e}")
            return False


if __name__ == '__main__':
    # Exemple d'utilisation
    predictor = V8LightGBMIntraday()
    
    # Entraîner
    tickers = ['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'META', 'TSLA', 'SPY', 'QQQ']
    
    results = predictor.train(
        tickers=tickers,
        start_date='2023-01-01',
        end_date='2024-12-01',
        test_size=0.2
    )
    
    # Sauvegarder
    predictor.save()
    
    # Tester prédiction
    print("\n🔮 Test prédiction NVDA:")
    result = predictor.predict('NVDA')
    print(result)
