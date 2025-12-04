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
                self.df_hourly = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                ticker_name = csv_path.split("/")[-1].replace(".csv", "")
            except Exception as e:
                raise ValueError(f"Erreur lecture CSV {csv_path}: {e}")
        elif ticker:
            ticker_name = ticker
            print(f"⚠️ Warning: Téléchargement direct de {ticker} (LENT)")
            self.df_hourly = yf.download(ticker, period="730d", interval="1h", auto_adjust=True, progress=False)
            if isinstance(self.df_hourly.columns, pd.MultiIndex):
                self.df_hourly = self.df_hourly.xs(ticker, axis=1, level=1)
        else:
            raise ValueError("TradingEnvMultiTimeframe doit recevoir 'csv_path' ou 'ticker'")
        
        # NOUVEAU : TÉLÉCHARGEMENT DONNÉES DAILY
        print(f"📥 Téléchargement données daily pour {ticker_name}...")
        self.df_daily = yf.download(ticker_name, period="730d", interval="1d", auto_adjust=True, progress=False)
        
        if isinstance(self.df_daily.columns, pd.MultiIndex):
            self.df_daily = self.df_daily.xs(ticker_name, axis=1, level=1)
        
        # Garder seulement Close pour daily
        self.df_daily = self.df_daily[['Close']].rename(columns={'Close': 'Close_daily'})
        
        # MERGE : Broadcast daily sur hourly
        self.df = self.df_hourly.join(self.df_daily, how='left')
        self.df['Close_daily'] = self.df['Close_daily'].ffill()  # Forward fill
        self.df = self.df.dropna()
        
        print(f"✅ {len(self.df)} bougies horaires + daily data chargées")
        
        if len(self.df) < lookback_window + 10:
            raise ValueError(f"Pas assez de données: {len(self.df)} lignes")
        
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        
        # NOUVEAU : Espace d'observation élargi
        # Hourly: 5 features × 50 = 250
        # Daily: 10 derniers jours = 10
        # Portfolio: 3
        # TOTAL: 263
        obs_size = (5 * lookback_window) + 10 + 3
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.shares = 0
        self.current_step = self.lookback_window
        
        return self._next_observation(), {}
    
    def _next_observation(self):
        """Construit l'observation MULTI-TIMEFRAME"""
        
        # 1. DONNÉES HORAIRES (lookback_window dernières heures)
        frame_hourly = self.df.iloc[self.current_step - self.lookback_window:self.current_step]
        obs_hourly = frame_hourly[['Open', 'High', 'Low', 'Close', 'Volume']].values.flatten()
        
        # 2. DONNÉES DAILY (10 derniers jours)
        current_date = self.df.index[self.current_step]
        
        # Filtrer toutes les lignes avant current_date
        df_before = self.df[self.df.index <= current_date]
        
        # Prendre les valeurs daily uniques (1 par jour)
        daily_closes = df_before['Close_daily'].drop_duplicates().tail(10).values
        
        # Padding si moins de 10 jours disponibles
        if len(daily_closes) < 10:
            padding = np.full(10 - len(daily_closes), daily_closes[0] if len(daily_closes) > 0 else 0)
            daily_closes = np.concatenate([padding, daily_closes])
        
        obs_daily = daily_closes
        
        # 3. NORMALISATION (par prix actuel)
        current_close = frame_hourly['Close'].iloc[-1] + 1e-8
        obs_hourly_norm = obs_hourly / current_close
        obs_daily_norm = obs_daily / current_close
        
        # 4. PORTFOLIO STATE
        current_price = self.df.iloc[self.current_step]['Close']
        portfolio_state = np.array([
            self.balance / self.initial_balance,
            (self.shares * current_price) / self.initial_balance,
            current_price / 1000.0
        ])
        
        # CONCATÉNATION FINALE
        return np.concatenate([obs_hourly_norm, obs_daily_norm, portfolio_state]).astype(np.float32)
    
    def step(self, action):
        """Step classique (reward simple)"""
        
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
        
        # Reward simple
        reward = (current_value - prev_value) / (prev_value + 1e-8)
        
        if action != 0:
            reward += 0.0005
        else:
            reward -= 0.0001
        
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
