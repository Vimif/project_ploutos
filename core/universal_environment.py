"""
Environnement de trading universel
Peut trader N'IMPORTE QUEL asset avec les mêmes features
Supporte portfolio multi-assets
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from core.data_loader import load_market_data

class UniversalTradingEnv(gym.Env):
    """Environnement générique pour trader un portfolio multi-assets"""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, regime_detector, enable_market_scan=False):
        """
        Args:
            regime_detector: Instance de MarketRegimeDetector
            enable_market_scan: Si True, active le scan complet du marché
        """
        self.regime_detector = regime_detector
        self.last_selection = None
        self.selection_history = []
        self.enable_market_scan = enable_market_scan
        
        # Initialiser scanner si activé
        self.fetcher = None
        self.scanner = None
        
        if enable_market_scan:
            try:
                from core.data_fetcher import UniversalDataFetcher
                from core.market_scanner import MarketScanner
                
                self.fetcher = UniversalDataFetcher()
                self.scanner = MarketScanner(self.fetcher)
                print("🔍 Market Scanner activé")
            except ImportError as e:
                print(f"⚠️ Market Scanner non disponible : {e}")
                print("  → Installer : pip install alpaca-trade-api requests")
                self.enable_market_scan = False
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.positions = {ticker: 0 for ticker in self.data.keys()}
        self.current_step = self.lookback_window
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        """Construit l'observation multi-assets"""
        
        obs_parts = []
        
        # 1. Features de chaque asset (OHLCV normalisé)
        for ticker in self.data.keys():
            df = self.data[ticker]
            
            # Protection bounds
            start_idx = max(0, self.current_step - self.lookback_window)
            end_idx = min(self.current_step, len(df))
            
            frame = df.iloc[start_idx:end_idx]
            
            if len(frame) < self.lookback_window:
                # Padding si pas assez de données
                padding = np.zeros((self.lookback_window - len(frame), 5))
                features = np.vstack([padding, frame[['Open', 'High', 'Low', 'Close', 'Volume']].values])
            else:
                features = frame[['Open', 'High', 'Low', 'Close', 'Volume']].values
            
            # Normalisation par le dernier prix de clôture
            last_close = frame['Close'].iloc[-1] if len(frame) > 0 else 1.0
            features = features / (last_close + 1e-8)
            
            obs_parts.append(features.flatten())
        
        # 2. Régime de marché (one-hot encoding)
        if self.regime_detector:
            try:
                regime_info = self.regime_detector.detect()
                regime_encoded = self._encode_regime(regime_info['regime'])
            except:
                regime_encoded = np.zeros(5)
                regime_encoded[2] = 1  # SIDEWAYS par défaut
        else:
            regime_encoded = np.zeros(5)
            regime_encoded[2] = 1
        
        obs_parts.append(regime_encoded)
        
        # 3. État du portfolio
        portfolio_state = [self.balance / self.initial_balance]
        
        for ticker in self.data.keys():
            df = self.data[ticker]
            if self.current_step < len(df):
                price = df.iloc[self.current_step]['Close']
                position_value = self.positions[ticker] * price
                portfolio_state.append(position_value / self.initial_balance)
        
        obs_parts.append(np.array(portfolio_state))
        
        # Concaténer tout
        return np.concatenate(obs_parts).astype(np.float32)
    
    def _encode_regime(self, regime):
        """Encode le régime en vecteur one-hot"""
        regimes = ['BULL', 'BEAR', 'SIDEWAYS', 'HIGH_VOLATILITY', 'UNKNOWN']
        encoded = np.zeros(5)
        if regime in regimes:
            encoded[regimes.index(regime)] = 1
        else:
            encoded[4] = 1  # UNKNOWN
        return encoded
    
    def step(self, actions):
        """
        Exécute les actions sur tous les assets
        
        Args:
            actions: Array d'actions [action_asset1, action_asset2, ...]
        
        Returns:
            obs, reward, terminated, truncated, info
        """
        
        # Calculer valeur portfolio avant action
        prev_value = self.balance
        for ticker in self.data.keys():
            df = self.data[ticker]
            if self.current_step < len(df):
                price = df.iloc[self.current_step]['Close']
                prev_value += self.positions[ticker] * price
        
        # Exécuter chaque action
        for idx, ticker in enumerate(self.data.keys()):
            action = actions[idx]
            df = self.data[ticker]
            
            if self.current_step >= len(df):
                continue
            
            price = df.iloc[self.current_step]['Close']
            
            if action == 1:  # BUY
                # Diviser le capital disponible par le nombre d'assets
                available = self.balance / len(self.data)
                shares = int(available / (price * 1.001))  # Commission 0.1%
                
                if shares > 0:
                    cost = shares * price * 1.001
                    self.positions[ticker] += shares
                    self.balance -= cost
            
            elif action == 2:  # SELL
                if self.positions[ticker] > 0:
                    revenue = self.positions[ticker] * price * 0.999  # Commission
                    self.balance += revenue
                    self.positions[ticker] = 0
        
        # Avancer d'un step
        self.current_step += 1
        
        # Calculer valeur portfolio après action
        current_value = self.balance
        for ticker in self.data.keys():
            df = self.data[ticker]
            if self.current_step < len(df):
                price = df.iloc[self.current_step]['Close']
                current_value += self.positions[ticker] * price
        
        # Reward = Log return (plus stable)
        reward = np.log((current_value + 1e-8) / (prev_value + 1e-8))
        
        # Bonus pour diversification
        n_positions = sum(1 for pos in self.positions.values() if pos > 0)
        if n_positions > 1:
            reward += 0.0001 * n_positions
        
        # Pénalité si tout le capital est bloqué (pas de liquidité)
        if self.balance < self.initial_balance * 0.05:
            reward -= 0.001
        
        # Condition de terminaison
        terminated = False
        if self.current_step >= self.min_length - 1:
            terminated = True
        
        # Stop loss global (-70%)
        if current_value < self.initial_balance * 0.30:
            terminated = True
            reward = -5.0
        
        info = {
            'total_value': float(current_value),
            'balance': float(self.balance),
            'positions': {k: int(v) for k, v in self.positions.items()},
            'n_active_positions': n_positions
        }
        
        return self._get_obs(), reward, terminated, False, info
    
    def render(self, mode='human'):
        """Affiche l'état actuel"""
        print(f"\nStep {self.current_step}")
        print(f"Balance: ${self.balance:,.2f}")
        for ticker, shares in self.positions.items():
            if shares > 0:
                df = self.data[ticker]
                price = df.iloc[min(self.current_step, len(df)-1)]['Close']
                value = shares * price
                print(f"  {ticker}: {shares} shares @ ${price:.2f} = ${value:,.2f}")
