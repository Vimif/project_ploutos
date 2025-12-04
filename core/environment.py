import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
import os

# NOUVEAU : Import des indicateurs techniques
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands

class TradingEnv(gym.Env):
    """Environnement avec indicateurs techniques"""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, csv_path=None, ticker=None, initial_balance=10000, lookback_window=50):
        super(TradingEnv, self).__init__()
        
        # Chargement données
        if csv_path and os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        elif ticker:
            self.df = yf.download(ticker, period="730d", interval="1h", auto_adjust=True, progress=False)
            if isinstance(self.df.columns, pd.MultiIndex):
                self.df = self.df.xs(ticker, axis=1, level=1)
        else:
            raise ValueError("csv_path ou ticker requis")
        
        # NOUVEAU : Calcul des indicateurs techniques (UNE FOIS au début)
        self.df = self._add_technical_indicators(self.df)
        self.df = self.df.dropna()  # Les indicateurs créent des NaN au début
        
        if len(self.df) < lookback_window + 10:
            raise ValueError(f"Pas assez de données: {len(self.df)}")
        
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        
        # NOUVEAU : Espace d'observation élargi
        # Avant: 5 features × 50 = 250
        # Après: 13 features × 50 = 650 (+ 3 portfolio = 653)
        obs_size = 13 * lookback_window + 3
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
    
    def _add_technical_indicators(self, df):
        """Ajoute les indicateurs techniques au DataFrame"""
        
        # 1. RSI (Relative Strength Index) - Détecte surachat/survente
        rsi = RSIIndicator(close=df['Close'], window=14)
        df['RSI'] = rsi.rsi()
        
        # 2. MACD (Moving Average Convergence Divergence) - Momentum
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()  # Histogramme
        
        # 3. Bollinger Bands - Volatilité
        bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_middle'] = bb.bollinger_mavg()
        
        # 4. Moyennes Mobiles (EMA rapide, SMA lente)
        df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
        
        return df
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.shares = 0
        self.current_step = self.lookback_window
        
        return self._next_observation(), {}
    
    def _next_observation(self):
        """Construit l'observation AVEC les indicateurs"""
        
        frame = self.df.iloc[self.current_step - self.lookback_window:self.current_step]
        
        # NOUVEAU : 13 features au lieu de 5
        features = ['Open', 'High', 'Low', 'Close', 'Volume',
                   'RSI', 'MACD', 'MACD_signal', 'MACD_diff',
                   'BB_upper', 'BB_lower', 'BB_middle', 'EMA_20']
        
        obs = frame[features].values.flatten()
        
        # Normalisation par le prix actuel
        current_close = frame['Close'].iloc[-1] + 1e-8
        obs = obs / current_close
        
        # Portfolio state (inchangé)
        current_price = self.df.iloc[self.current_step]['Close']
        portfolio_state = np.array([
            self.balance / self.initial_balance,
            (self.shares * current_price) / self.initial_balance,
            current_price / 1000.0
        ])
        
        return np.concatenate([obs, portfolio_state]).astype(np.float32)
    
    def step(self, action):
        """Step function (identique à avant)"""
        current_price = self.df.iloc[self.current_step]['Close']
        prev_price = self.df.iloc[self.current_step - 1]['Close']
        prev_value = self.balance + self.shares * prev_price
        
        if action == 1:  # BUY
            shares_to_buy = int(self.balance // current_price)
            if shares_to_buy > 0:
                self.shares += shares_to_buy
                self.balance -= shares_to_buy * current_price
        elif action == 2:  # SELL
            if self.shares > 0:
                self.balance += self.shares * current_price
                self.shares = 0
        
        self.current_step += 1
        new_price = self.df.iloc[self.current_step]['Close']
        current_value = self.balance + self.shares * new_price
        
        # Reward (inchangé)
        reward = (current_value - prev_value) / (prev_value + 1e-8)
        if action != 0:
            reward += 0.0005
        else:
            reward -= 0.0001
        
        terminated = False
        if self.current_step >= len(self.df) - 1:
            terminated = True
            final_pnl = (current_value - self.initial_balance) / self.initial_balance
            reward += final_pnl * 10
        
        if current_value <= 0:
            terminated = True
            reward = -10.0
        
        truncated = False
        info = {"total_value": float(current_value)}
        
        return self._next_observation(), reward, terminated, truncated, info
