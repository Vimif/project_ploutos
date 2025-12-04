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
        
        # CHARGEMENT DONNÉES HORAIRES
        if csv_path and os.path.exists(csv_path):
            try:
                self.df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                # FIX : Retirer timezone
                if self.df.index.tz is not None:
                    self.df.index = self.df.index.tz_localize(None)
                
                # Extraire ticker name du path
                ticker_name = os.path.basename(csv_path).replace(".csv", "")
            except Exception as e:
                raise ValueError(f"Erreur lecture CSV {csv_path}: {e}")
        elif ticker:
            ticker_name = ticker
            print(f"⚠️ Téléchargement {ticker}...")
            self.df = yf.download(ticker, period="730d", interval="1h", auto_adjust=True, progress=False)
            if isinstance(self.df.columns, pd.MultiIndex):
                self.df = self.df.xs(ticker, axis=1, level=1)
            if self.df.index.tz is not None:
                self.df.index = self.df.index.tz_localize(None)
        else:
            raise ValueError("Doit recevoir 'csv_path' ou 'ticker'")
        
        self.df = self.df.dropna()
        
        # AJOUT : Indicateurs techniques (comme baseline)
        self.df['RSI'] = self._calculate_rsi(self.df['Close'])
        self.df['MACD'], self.df['MACD_signal'] = self._calculate_macd(self.df['Close'])
        self.df['BB_upper'], self.df['BB_lower'] = self._calculate_bollinger_bands(self.df['Close'])
        self.df['EMA_20'] = self.df['Close'].ewm(span=20).mean()
        self.df = self.df.dropna()
        
        # NOUVEAU : Données DAILY simplifiées
        # Au lieu de télécharger, on agrège les données horaires
        print(f"📊 Agrégation données daily...")
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
        
        print(f"✅ {len(self.df)} bougies horaires chargées avec contexte daily")
        
        if len(self.df) < lookback_window + 10:
            raise ValueError(f"Pas assez de données: {len(self.df)} lignes (min {lookback_window + 10})")
        
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        
        # Observation space : OHLCV + indicateurs + daily close + portfolio
        # 5 (OHLCV) + 5 (indicateurs) + 1 (daily) = 11 features × 50 steps + 3 portfolio
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
        """Observation avec données horaires + indicateurs + daily"""
        
        frame = self.df.iloc[self.current_step - self.lookback_window:self.current_step]
        
        # Features : OHLCV + indicateurs + daily close
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                   'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 
                   'Close_daily']
        
        obs = frame[features].values.flatten()
        
        # Normalisation
        current_close = frame['Close'].iloc[-1] + 1e-8
        obs = obs / current_close
        
        # Portfolio state
        current_price = self.df.iloc[self.current_step]['Close']
        portfolio_state = np.array([
            self.balance / self.initial_balance,
            (self.shares * current_price) / self.initial_balance,
            current_price / 1000.0
        ])
        
        return np.concatenate([obs, portfolio_state]).astype(np.float32)
    
    def step(self, action):
        """Step standard"""
        
        current_price = self.df.iloc[self.current_step]['Close']
        prev_value = self.balance + self.shares * current_price
        
        # Actions
        if action == 1:  # BUY
            shares_to_buy = int(self.balance // current_price)
            if shares_to_buy > 0:
                self.shares += shares_to_buy
                self.balance -= shares_to_buy * current_price
                
        elif action == 2:  # SELL
            if self.shares > 0:
                self.balance += self.shares * current_price
                self.shares = 0
        
        # Avancer
        self.current_step += 1
        new_price = self.df.iloc[self.current_step]['Close']
        current_value = self.balance + self.shares * new_price
        
        # Reward
        reward = (current_value - prev_value) / (prev_value + 1e-8)
        
        # Bonus diversification
        if action != 0:
            reward += 0.0005
        else:
            reward -= 0.0002
        
        # Fin
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
    
    def render(self, mode='human'):
        if self.current_step < len(self.df):
            current_price = self.df.iloc[self.current_step]['Close']
            total_value = self.balance + self.shares * current_price
            pnl = ((total_value - self.initial_balance) / self.initial_balance) * 100
            print(f"Step {self.current_step} | Value: ${total_value:.2f} | P&L: {pnl:+.2f}%")