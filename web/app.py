#!/usr/bin/env python3
"""
🌐 PLOUTOS WEB DASHBOARD + V7 ENHANCED

Dashboard Web moderne pour monitorer le bot de trading

Features:
- Vue temps réel du portfolio
- Graphiques de performances
- ★ Prédictions V7 Enhanced Momentum (68.35% accuracy)
- V7 Ensemble (Momentum + Reversion + Volatility)
- Health Score et auto-amélioration
- Historique des trades
- Alertes et suggestions

Auteur: Ploutos AI Team
Date: Dec 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
import torch
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

# Import modules Ploutos
try:
    from trading.alpaca_client import AlpacaClient
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

try:
    from core.self_improvement import SelfImprovementEngine
    SELF_IMPROVEMENT_AVAILABLE = True
except ImportError:
    SELF_IMPROVEMENT_AVAILABLE = False

# ★ IMPORT V7 ENHANCED PREDICTOR
try:
    from src.models.v7_predictor import V7Predictor
    V7_ENHANCED_AVAILABLE = True
except ImportError:
    V7_ENHANCED_AVAILABLE = False

# ========== V7 ENSEMBLE MODELS (ancien système) ==========

class RobustMomentumClassifier(torch.nn.Module):
    def __init__(self, input_dim=28):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256), torch.nn.ReLU(), torch.nn.BatchNorm1d(256), torch.nn.Dropout(0.4),
            torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.BatchNorm1d(128), torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.BatchNorm1d(64), torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(32, 2),
        )
    def forward(self, x): return self.net(x)

class ReversionModel(torch.nn.Module):
    def __init__(self, input_dim=9):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64), torch.nn.Tanh(), torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 32), torch.nn.Tanh(), torch.nn.Linear(32, 2)
        )
    def forward(self, x): return self.net(x)

class VolatilityModel(torch.nn.Module):
    def __init__(self, input_dim=6):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64), torch.nn.ReLU(), torch.nn.BatchNorm1d(64),
            torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2)
        )
    def forward(self, x): return self.net(x)

class SimpleFeatureExtractor:
    def extract_features(self, df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df = df.reset_index(drop=True)
        
        df['returns'] = df['Close'].pct_change()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()
        df['price_position'] = (df['Close'] - df['sma_20']) / (df['sma_20'] + 1e-6)
        df['high_low_ratio'] = (df['High'] - df['Low']) / df['Close']
        df['close_open_ratio'] = (df['Close'] - df['Open']) / (df['Open'] + 1e-6)
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-6)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['rsi_7'] = self._rsi(df['Close'], 7)
        
        macd, signal, hist = self._macd(df['Close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_histogram'] = hist
        df['momentum_10'] = df['Close'] - df['Close'].shift(10)
        df['momentum_5'] = df['Close'] - df['Close'].shift(5)
        df['rate_of_change'] = (df['Close'] - df['Close'].shift(1)) / (df['Close'].shift(1) + 1e-6)
        
        low_14 = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        df['stoch_k'] = 100 * (df['Close'] - low_14) / (high_14 - low_14 + 1e-6)
        
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['volatility_5'] = df['returns'].rolling(5).std()
        df['atr'] = (df['High'] - df['Low']).rolling(14).mean()
        df['atr_ratio'] = df['atr'] / (df['Close'] + 1e-6)
        
        upper = df['Close'].rolling(20).mean() + df['Close'].rolling(20).std() * 2
        lower = df['Close'].rolling(20).mean() - df['Close'].rolling(20).std() * 2
        df['bb_position'] = (df['Close'] - lower) / (upper - lower + 1e-6)
        
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / (df['volume_sma'] + 1e-6)
        df['price_volume_trend'] = df['Close'].pct_change() * df['Volume']
        df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        df['obv_momentum'] = df['obv'] - df['obv'].shift(5)
        
        df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['ema_ratio'] = df['ema_12'] / (df['ema_26'] + 1e-6)
        df['trend_strength'] = (df['Close'] - df['Close'].rolling(20).min()) / (df['Close'].rolling(20).max() - df['Close'].rolling(20).min() + 1e-6)
        
        cols = ['returns', 'sma_20', 'sma_50', 'price_position', 'high_low_ratio', 'close_open_ratio',
                'rsi_14', 'rsi_7', 'macd', 'macd_signal', 'macd_histogram', 'momentum_10', 'momentum_5',
                'rate_of_change', 'stoch_k', 'volatility_20', 'volatility_5', 'atr', 'atr_ratio', 'bb_position',
                'volume_sma', 'volume_ratio', 'price_volume_trend', 'obv', 'obv_momentum',
                'ema_12', 'ema_26', 'ema_ratio', 'trend_strength']
        df = df.dropna()
        return df[cols].values, cols
    
    def _rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-6)
        return 100 - (100 / (1 + rs))
    
    def _macd(self, prices):
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal, macd - signal

class ReversionFeatureExtractor:
    def extract_features(self, df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df = df.reset_index(drop=True)
        
        sma_20 = df['Close'].rolling(20).mean()
        std_20 = df['Close'].rolling(20).std()
        upper = sma_20 + 2 * std_20
        lower = sma_20 - 2 * std_20
        
        df['bb_pct'] = (df['Close'] - lower) / (upper - lower + 1e-6)
        df['bb_width'] = (upper - lower) / (sma_20 + 1e-6)
        df['dist_sma_20'] = (df['Close'] - sma_20) / sma_20
        df['dist_sma_50'] = (df['Close'] - df['Close'].rolling(50).mean()) / df['Close'].rolling(50).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-6)))
        
        low_14 = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        df['stoch_k'] = 100 * (df['Close'] - low_14) / (high_14 - low_14 + 1e-6)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        df['williams_r'] = -100 * (high_14 - df['Close']) / (high_14 - low_14 + 1e-6)
        df['z_score'] = (df['Close'] - sma_20) / (std_20 + 1e-6)
        
        cols = ['bb_pct', 'bb_width', 'dist_sma_20', 'dist_sma_50', 'rsi', 'stoch_k', 'stoch_d', 'williams_r', 'z_score']
        df = df.dropna()
        return df[cols].values, cols

class VolatilityFeatureExtractor:
    def extract_features(self, df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df = df.reset_index(drop=True)
        
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        df['atr_pct'] = atr / (df['Close'] + 1e-6)
        
        returns = df['Close'].pct_change()
        df['vol_20'] = returns.rolling(20).std() * np.sqrt(252)
        df['vol_5'] = returns.rolling(5).std() * np.sqrt(252)
        
        high_14 = df['High'].rolling(14).max()
        low_14 = df['Low'].rolling(14).min()
        df['chop_idx'] = 100 * np.log10(tr.rolling(14).sum() / (high_14 - low_14 + 1e-6)) / np.log10(14)
        df['vol_volatility'] = df['Volume'].pct_change().rolling(20).std()
        
        net_change = np.abs(df['Close'] - df['Close'].shift(10))
        total_path = np.abs(df['Close'].diff()).rolling(10).sum()
        df['efficiency'] = net_change / (total_path + 1e-6)
        
        cols = ['atr_pct', 'vol_20', 'vol_5', 'chop_idx', 'vol_volatility', 'efficiency']
        df = df.dropna()
        return df[cols].values, cols

def convert_to_native_python(obj):
    """Convertit les types numpy en types Python natifs pour JSON"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_native_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_python(item) for item in obj]
    return obj

# Setup
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation clients
alpaca_client = None
if ALPACA_AVAILABLE:
    try:
        alpaca_client = AlpacaClient(paper_trading=True)
    except Exception as e:
        logger.warning(f"⚠️  Alpaca non disponible: {e}")

# ★ INITIALISER V7 ENHANCED PREDICTOR
v7_enhanced = None
if V7_ENHANCED_AVAILABLE:
    try:
        v7_enhanced = V7Predictor()
        if v7_enhanced.load("momentum"):
            logger.info("✅ V7 Enhanced Predictor chargé (68.35% accuracy)")
        else:
            logger.warning("⚠️  V7 Enhanced non chargé")
            v7_enhanced = None
    except Exception as e:
        logger.warning(f"⚠️  Erreur chargement V7 Enhanced: {e}")
        v7_enhanced = None

# V7 Ensemble Models Cache (ancien système)
v7_models_cache = {}

def load_v7_models():
    """Charge les 3 modèles V7 Ensemble en cache"""
    global v7_models_cache
    
    if v7_models_cache:
        return v7_models_cache
    
    try:
        import pickle
        models_dir = Path('models')
        
        # Load Momentum
        mom_meta = json.load(open(models_dir / 'v7_multiticker/metadata.json'))
        mom_scaler = pickle.load(open(models_dir / 'v7_multiticker/scaler.pkl', 'rb'))
        mom_model = RobustMomentumClassifier(mom_meta['input_dim'])
        mom_model.load_state_dict(torch.load(models_dir / 'v7_multiticker/best_model.pth', map_location='cpu'))
        mom_model.eval()
        
        # Load Reversion
        rev_meta = json.load(open(models_dir / 'v7_mean_reversion/metadata.json'))
        rev_scaler = pickle.load(open(models_dir / 'v7_mean_reversion/scaler.pkl', 'rb'))
        rev_model = ReversionModel(len(rev_meta['features']))
        rev_model.load_state_dict(torch.load(models_dir / 'v7_mean_reversion/best_model.pth', map_location='cpu'))
        rev_model.eval()
        
        # Load Volatility
        vol_meta = json.load(open(models_dir / 'v7_volatility/metadata.json'))
        vol_scaler = pickle.load(open(models_dir / 'v7_volatility/scaler.pkl', 'rb'))
        vol_model = VolatilityModel(len(vol_meta['features']))
        vol_model.load_state_dict(torch.load(models_dir / 'v7_volatility/best_model.pth', map_location='cpu'))
        vol_model.eval()
        
        v7_models_cache = {
            'mom': {'model': mom_model, 'scaler': mom_scaler, 'meta': mom_meta},
            'rev': {'model': rev_model, 'scaler': rev_scaler, 'meta': rev_meta},
            'vol': {'model': vol_model, 'scaler': vol_scaler, 'meta': vol_meta}
        }
        
        logger.info("✅ V7 Ensemble Models loaded successfully")
        return v7_models_cache
        
    except Exception as e:
        logger.error(f"❌ Erreur chargement V7 Ensemble: {e}")
        return None

# Cache simple
cache = {
    'account': None,
    'positions': None,
    'trades': None,
    'improvement_report': None,
    'v7_enhanced_cache': None,
    'last_update': None,
    'v7_enhanced_last_update': None
}


# ========== ROUTES HTML ==========

@app.route('/')
def index():
    """Page principale"""
    return render_template('index.html')


# ========== API ENDPOINTS ==========

@app.route('/api/status')
def api_status():
    """Status général du système"""
    return jsonify({
        'status': 'online',
        'timestamp': datetime.now().isoformat(),
        'alpaca_connected': alpaca_client is not None,
        'self_improvement_available': SELF_IMPROVEMENT_AVAILABLE,
        'v7_enhanced_available': v7_enhanced is not None,
        'v7_ensemble_available': bool(v7_models_cache)
    })


# ★ V7 ENHANCED ENDPOINTS (nouveau système)

@app.route('/api/v7/enhanced/predict/<ticker>')
def api_v7_enhanced_single(ticker):
    """Prédiction V7 Enhanced pour un ticker"""
    if not v7_enhanced:
        return jsonify({'error': 'V7 Enhanced non disponible'}), 503
    
    try:
        result = v7_enhanced.predict(ticker.upper(), period="3mo")
        
        if "error" not in result:
            return jsonify({
                'ticker': ticker.upper(),
                'timestamp': datetime.now().isoformat(),
                'model': 'V7 Enhanced Momentum',
                'accuracy': '68.35%',
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'signal_strength': result['signal_strength'],
                'probabilities': result['probabilities']
            })
        else:
            return jsonify({'error': result['error']}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v7/enhanced/batch')
def api_v7_enhanced_batch():
    """
    Prédictions V7 Enhanced pour plusieurs tickers
    Cache de 5 minutes
    """
    if not v7_enhanced:
        return jsonify({'error': 'V7 Enhanced non disponible'}), 503
    
    # Cache 5 minutes
    if cache['v7_enhanced_cache'] and cache['v7_enhanced_last_update']:
        if (datetime.now() - cache['v7_enhanced_last_update']).seconds < 300:
            return jsonify(cache['v7_enhanced_cache'])
    
    # Tickers principaux
    tickers = request.args.get('tickers', 'NVDA,MSFT,AAPL,GOOGL,AMZN,SPY,QQQ').split(',')
    
    try:
        logger.info(f"🔮 Génération prédictions V7 Enhanced pour {len(tickers)} tickers...")
        
        predictions = {}
        for ticker in tickers:
            ticker = ticker.strip().upper()
            try:
                result = v7_enhanced.predict(ticker, period="3mo")
                
                if "error" not in result:
                    predictions[ticker] = {
                        'prediction': result['prediction'],
                        'confidence': round(result['confidence'], 4),
                        'signal_strength': round(result['signal_strength'], 4),
                        'probabilities': {
                            'up': round(result['probabilities']['up'], 4),
                            'down': round(result['probabilities']['down'], 4)
                        }
                    }
                else:
                    predictions[ticker] = {'error': result['error']}
                    
            except Exception as e:
                predictions[ticker] = {'error': str(e)}
        
        response = {
            'timestamp': datetime.now().isoformat(),
            'model': 'V7 Enhanced Momentum',
            'accuracy': '68.35%',
            'predictions': predictions
        }
        
        cache['v7_enhanced_cache'] = response
        cache['v7_enhanced_last_update'] = datetime.now()
        
        logger.info(f"✅ Prédictions V7 Enhanced: {len(predictions)} tickers")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Erreur prédictions V7 Enhanced: {e}")
        return jsonify({'error': str(e)}), 500


# ========== V7 ENSEMBLE ENDPOINTS (ancien système) ==========

@app.route('/api/v7/analysis')
def api_v7_analysis():
    """Analyse d'un ticker avec le système V7 Ensemble"""
    ticker = request.args.get('ticker', '').upper()
    
    if not ticker:
        return jsonify({'error': 'Ticker requis'}), 400
    
    # Charger modèles
    models = load_v7_models()
    if not models:
        return jsonify({'error': 'V7 Ensemble non disponible'}), 503
    
    try:
        # Télécharger données
        df = yf.download(ticker, period="2y", progress=False)
        if df.empty:
            return jsonify({'error': f'Données indisponibles pour {ticker}'}), 404
        
        # 1. MOMENTUM
        X_mom, _ = SimpleFeatureExtractor().extract_features(df)
        X_mom_scaled = models['mom']['scaler'].transform(X_mom[-1:])
        with torch.no_grad():
            logits = models['mom']['model'](torch.FloatTensor(X_mom_scaled))
            probs = torch.softmax(logits, 1).numpy()[0]
            mom_pred = 1 if probs[1] > 0.5 else 0
            mom_conf = float(probs[1])
        
        # 2. REVERSION
        X_rev, _ = ReversionFeatureExtractor().extract_features(df)
        X_rev_scaled = models['rev']['scaler'].transform(X_rev[-1:])
        with torch.no_grad():
            logits = models['rev']['model'](torch.FloatTensor(X_rev_scaled))
            probs = torch.softmax(logits, 1).numpy()[0]
            rev_pred = 1 if probs[1] > 0.5 else 0
            rev_conf = float(probs[1])
        
        # 3. VOLATILITY
        X_vol, _ = VolatilityFeatureExtractor().extract_features(df)
        X_vol_scaled = models['vol']['scaler'].transform(X_vol[-1:])
        with torch.no_grad():
            logits = models['vol']['model'](torch.FloatTensor(X_vol_scaled))
            probs = torch.softmax(logits, 1).numpy()[0]
            vol_pred = 1 if probs[1] > 0.5 else 0
            vol_conf = float(probs[1])
        
        # VOTING LOGIC
        score = mom_pred + rev_pred - 1
        
        if score >= 1: signal = "BUY"
        elif score <= -1: signal = "SELL"
        else: signal = "HOLD"
        
        # Volatility filter
        strength = "STRONG" if vol_pred == 1 else "WEAK"
        
        return jsonify({
            'ticker': ticker,
            'signal': signal,
            'strength': strength,
            'experts': {
                'momentum': {'prediction': 'UP' if mom_pred else 'DOWN', 'confidence': mom_conf * 100},
                'reversion': {'prediction': 'UP' if rev_pred else 'DOWN', 'confidence': rev_conf * 100},
                'volatility': {'prediction': 'HIGH' if vol_pred else 'LOW', 'confidence': vol_conf * 100}
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erreur V7 Ensemble: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v7/batch')
def api_v7_batch():
    """Analyse plusieurs tickers en batch (V7 Ensemble)"""
    tickers = request.args.get('tickers', 'NVDA,AAPL,MSFT').split(',')
    
    models = load_v7_models()
    if not models:
        return jsonify({'error': 'V7 Ensemble non disponible'}), 503
    
    results = []
    
    for ticker in tickers:
        ticker = ticker.strip().upper()
        try:
            df = yf.download(ticker, period="2y", progress=False)
            if df.empty:
                continue
            
            X_mom, _ = SimpleFeatureExtractor().extract_features(df)
            X_mom_scaled = models['mom']['scaler'].transform(X_mom[-1:])
            with torch.no_grad():
                logits = models['mom']['model'](torch.FloatTensor(X_mom_scaled))
                mom_conf = torch.softmax(logits, 1).numpy()[0][1]
                mom_pred = 1 if mom_conf > 0.5 else 0
            
            X_rev, _ = ReversionFeatureExtractor().extract_features(df)
            X_rev_scaled = models['rev']['scaler'].transform(X_rev[-1:])
            with torch.no_grad():
                logits = models['rev']['model'](torch.FloatTensor(X_rev_scaled))
                rev_conf = torch.softmax(logits, 1).numpy()[0][1]
                rev_pred = 1 if rev_conf > 0.5 else 0
            
            X_vol, _ = VolatilityFeatureExtractor().extract_features(df)
            X_vol_scaled = models['vol']['scaler'].transform(X_vol[-1:])
            with torch.no_grad():
                logits = models['vol']['model'](torch.FloatTensor(X_vol_scaled))
                vol_conf = torch.softmax(logits, 1).numpy()[0][1]
                vol_pred = 1 if vol_conf > 0.5 else 0
            
            score = mom_pred + rev_pred - 1
            signal = "BUY" if score >= 1 else "SELL" if score <= -1 else "HOLD"
            strength = "STRONG" if vol_pred == 1 else "WEAK"
            
            results.append({
                'ticker': ticker,
                'signal': signal,
                'strength': strength,
                'momentum_conf': float(mom_conf * 100)
            })
        except Exception as e:
            logger.error(f"Erreur V7 Ensemble batch {ticker}: {e}")
            pass
    
    results = convert_to_native_python(results)
    
    return jsonify({'results': results, 'timestamp': datetime.now().isoformat()})


# ========== STANDARD ENDPOINTS ==========

@app.route('/api/account')
def api_account():
    """Informations du compte"""
    if not alpaca_client:
        return jsonify({'error': 'Alpaca non disponible'}), 503
    
    # Cache 30 secondes
    if cache['account'] and cache['last_update']:
        if (datetime.now() - cache['last_update']).seconds < 30:
            return jsonify(cache['account'])
    
    account = alpaca_client.get_account()
    if account:
        cache['account'] = account
        cache['last_update'] = datetime.now()
        return jsonify(account)
    
    return jsonify({'error': 'Impossible de récupérer le compte'}), 500


@app.route('/api/positions')
def api_positions():
    """Positions actuelles"""
    if not alpaca_client:
        return jsonify({'error': 'Alpaca non disponible'}), 503
    
    positions = alpaca_client.get_positions()
    cache['positions'] = positions
    
    return jsonify(positions)


@app.route('/api/trades')
def api_trades():
    """Historique des trades (depuis JSON)"""
    days = request.args.get('days', 7, type=int)
    
    trades_dir = Path('logs/trades')
    all_trades = []
    
    for i in range(days):
        date = datetime.now() - timedelta(days=i)
        filename = trades_dir / f"trades_{date.strftime('%Y-%m-%d')}.json"
        
        if filename.exists():
            try:
                with open(filename, 'r') as f:
                    daily_trades = json.load(f)
                    all_trades.extend(daily_trades)
            except:
                pass
    
    all_trades.sort(key=lambda t: t.get('timestamp', ''), reverse=True)
    
    return jsonify(all_trades)


@app.route('/api/performance')
def api_performance():
    """Statistiques de performance"""
    days = request.args.get('days', 7, type=int)
    
    trades_dir = Path('logs/trades')
    trades = []
    
    for i in range(days):
        date = datetime.now() - timedelta(days=i)
        filename = trades_dir / f"trades_{date.strftime('%Y-%m-%d')}.json"
        
        if filename.exists():
            try:
                with open(filename, 'r') as f:
                    daily_trades = json.load(f)
                    trades.extend(daily_trades)
            except:
                pass
    
    buys = [t for t in trades if t['action'] == 'BUY']
    sells = [t for t in trades if t['action'] == 'SELL']
    
    total_invested = sum(t['amount'] for t in buys)
    total_proceeds = sum(t['amount'] for t in sells)
    
    return jsonify({
        'total_trades': len(trades),
        'buy_count': len(buys),
        'sell_count': len(sells),
        'total_invested': total_invested,
        'total_proceeds': total_proceeds,
        'net_pnl': total_proceeds - total_invested,
        'days_analyzed': days
    })


@app.route('/api/improvement')
def api_improvement():
    """Rapport d'auto-amélioration"""
    if not SELF_IMPROVEMENT_AVAILABLE:
        return jsonify({'error': 'Self-Improvement non disponible'}), 503
    
    if cache['improvement_report']:
        report_time = datetime.fromisoformat(cache['improvement_report']['timestamp'])
        if (datetime.now() - report_time).seconds < 300:
            return jsonify(cache['improvement_report'])
    
    report_file = Path('logs/self_improvement_report.json')
    
    if report_file.exists():
        try:
            with open(report_file, 'r') as f:
                report = json.load(f)
                cache['improvement_report'] = report
                return jsonify(report)
        except:
            pass
    
    try:
        engine = SelfImprovementEngine()
        result = engine.analyze_recent_performance(days=7)
        
        if result['status'] == 'analyzed':
            report = engine.export_report()
            cache['improvement_report'] = report
            return jsonify(report)
    except Exception as e:
        logger.error(f"Erreur analyse: {e}")
    
    return jsonify({'error': 'Impossible de générer le rapport'}), 500


@app.route('/api/chart/portfolio')
def api_chart_portfolio():
    """Données pour graphique portfolio"""
    days = request.args.get('days', 30, type=int)
    
    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') 
             for i in range(days-1, -1, -1)]
    
    return jsonify({
        'dates': dates,
        'values': [100000] * days
    })


@app.route('/api/health')
def api_health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200


# ========== GESTION ERREURS ==========

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ========== MAIN ==========

if __name__ == '__main__':
    import os
    
    host = os.getenv('DASHBOARD_HOST', '0.0.0.0')
    port = int(os.getenv('DASHBOARD_PORT', 5000))
    debug = os.getenv('DASHBOARD_DEBUG', 'false').lower() == 'true'
    
    # Pre-load V7 Ensemble models
    load_v7_models()
    
    print("\n" + "="*60)
    print("🌐 PLOUTOS WEB DASHBOARD + V7 ENHANCED")
    print("="*60)
    print(f"\n🚀 Démarrage sur http://{host}:{port}")
    print(f"🔧 Mode debug: {debug}")
    print(f"📊 Alpaca: {'Actif' if alpaca_client else 'Inactif'}")
    print(f"🧠 Self-Improvement: {'Actif' if SELF_IMPROVEMENT_AVAILABLE else 'Inactif'}")
    print(f"⭐ V7 Enhanced: {'Actif (68.35% accuracy)' if v7_enhanced else 'Inactif'}")
    print(f"🤖 V7 Ensemble: {'Actif' if v7_models_cache else 'Inactif'}")
    print("\n" + "="*60 + "\n")
    
    app.run(host=host, port=port, debug=debug)
