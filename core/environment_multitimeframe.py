import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
import os

class TradingEnvMultiTimeframe(gym.Env):
    """Environnement avec données multi-timeframe (1h + 1d)"""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, csv_path=None, ticker=None, initial_balance=10000, lookback_window=50):
        super(TradingEnvMultiTimeframe, self).__init__()
        
        # CHARGEMENT DONNÉES HORAIRES ROBUSTE
        if csv_path and os.path.exists(csv_path):
            try:
                # 1. Lecture brute sans parser les dates au début
                self.df = pd.read_csv(csv_path, index_col=0)
                
                # 2. Forcer l'index en Datetime
                self.df.index = pd.to_datetime(self.df.index, utc=True)
                
                # 3. Nettoyer Timezone
                if self.df.index.tz is not None:
                    self.df.index = self.df.index.tz_localize(None)
                
                # 4. Forcer les colonnes en numérique (au cas où c'est des strings)
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in self.df.columns:
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                
                self.df = self.df.dropna()
                
            except Exception as e:
                raise ValueError(f"Erreur lecture CSV {csv_path}: {e}")
                
        elif ticker:
            print(f"⚠️ Téléchargement {ticker}...")
            self.df = yf.download(ticker, period="730d", interval="1h", auto_adjust=True, progress=False)
            if isinstance(self.df.columns, pd.MultiIndex):
                self.df = self.df.xs(ticker, axis=1, level=1)
            if self.df.index.tz is not None:
                self.df.index = self.df.index.tz_localize(None)
        else:
            raise ValueError("Doit recevoir 'csv_path' ou 'ticker'")
        
        # AJOUT : Indicateurs techniques
        self.df['RSI'] = self._calculate_rsi(self.df['Close'])
        self.df['MACD'], self.df['MACD_signal'] = self._calculate_macd(self.df['Close'])
        self.df['BB_upper'], self.df['BB_lower'] = self._calculate_bollinger_bands(self.df['Close'])
        self.df['EMA_20'] = self.df['Close'].ewm(span=20).mean()
        self.df = self.df.dropna()
        
        # NOUVEAU : Données DAILY simplifiées (Agrégation)
        # print(f"📊 Agrégation données daily...")
        df_daily = self.df.resample('D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        # Ajouter colonne daily au df principal
        self.df['Close_daily'] = self.df.index.map(
            lambda x: df_daily.loc[:x.date()].iloc[-1]['Close'] if len(df_daily.loc[:x.date()]) > 0 else np.nan
        )
        self.df = self.df.dropna()
        
        if len(self.df) < lookback_window + 10:
            raise ValueError(f"Pas assez de données: {len(self.df)} lignes")
        
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        
        # Obs: 11 features * window + 3 portfolio
        obs_size = 11 * lookback_window + 3
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
    
    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices):
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return macd, signal
    
    def _calculate_bollinger_bands(self, prices, period=20):
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        return upper, lower
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.shares = 0
        self.current_step = self.lookback_window
        return self._next_observation(), {}
    
    def _next_observation(self):
        frame = self.df.iloc[self.current_step - self.lookback_window:self.current_step]
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                   'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 
                   'Close_daily']
        obs = frame[features].values.flatten()
        current_close = frame['Close'].iloc[-1] + 1e-8
        obs = obs / current_close
        
        current_price = self.df.iloc[self.current_step]['Close']
        portfolio_state = np.array([
            self.balance / self.initial_balance,
            (self.shares * current_price) / self.initial_balance,
            current_price / 1000.0
        ])
        return np.concatenate([obs, portfolio_state]).astype(np.float32)
    
    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        prev_value = self.balance + self.shares * current_price
        
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
        
        reward = (current_value - prev_value) / (prev_value + 1e-8)
        
        if action != 0: reward += 0.0005
        else: reward -= 0.0002
        
        terminated = False
        if self.current_step >= len(self.df) - 1:
            terminated = True
            final_pnl = (current_value - self.initial_balance) / self.initial_balance
            reward += final_pnl * 10
        
        if current_value <= 0:
            terminated = True
            reward = -10.0
        
        return self._next_observation(), reward, terminated, False, {"total_value": float(current_value)}
