"""
Environnement de trading universel pour multi-assets
Avec pré-calcul des features pour performance maximale
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

# Import du modèle de coûts réalistes
try:
    from core.transaction_costs import AdvancedTransactionModel
    REALISTIC_COSTS = True
except ImportError:
    REALISTIC_COSTS = False

class UniversalTradingEnv(gym.Env):
    """Environnement générique pour trader un portfolio multi-assets"""

    metadata = {'render_modes': ['human']}

    def __init__(self, data, initial_balance=100000, commission=0.001, max_steps=2000, realistic_costs=True):
        """
        Args:
            data (dict): {ticker: DataFrame} avec colonnes [Open, High, Low, Close, Volume]
            initial_balance (float): Capital initial
            commission (float): Commission par trade
            max_steps (int): Nombre max de steps par épisode (2000 = ~83 jours)
            realistic_costs (bool): Utiliser modèle de coûts avancé
        """
        super().__init__()
        
        self.data = data
        self.tickers = list(data.keys())
        self.n_assets = len(self.tickers)
        self.initial_balance = initial_balance
        self.commission = commission
        self.max_steps = max_steps
        self.realistic_costs = realistic_costs and REALISTIC_COSTS
        
        # ✅ Modèle de coûts
        if self.realistic_costs:
            self.transaction_model = AdvancedTransactionModel(
                base_commission=commission,
                min_slippage=0.0005,
                max_slippage=0.005,
                market_impact_coef=0.0001,
                latency_std=0.0002
            )
        else:
            self.transaction_model = None
        
        # Longueur des données
        self.data_length = min(len(df) for df in data.values())
        
        # ✅ ★★★ PRÉ-CALCUL DES FEATURES ★★★
        print("\n⚡ Pré-calcul des features (accélération 10x)...")
        self._precompute_features()
        print("✅ Features pré-calculées !\n")
        
        # Observation space
        n_features_per_asset = 6  # [close_norm, volume_norm, rsi, macd, returns_1d, returns_5d]
        n_portfolio_features = 3  # [cash_ratio, total_value_norm, n_positions]
        
        obs_size = self.n_assets * n_features_per_asset + n_portfolio_features
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_size,), 
            dtype=np.float32
        )
        
        # Action space
        self.action_space = spaces.Box(
            low=-1, 
            high=1, 
            shape=(self.n_assets,), 
            dtype=np.float32
        )
        
        # État interne
        self.current_step = 0
        self.reset_step = 0
        self.balance = initial_balance
        self.positions = {ticker: 0 for ticker in self.tickers}
        self.portfolio_value = initial_balance
        self.trades_history = []
        self.transaction_costs_history = []
        
    def _precompute_features(self):
        """
        ✅ PRÉ-CALCULE TOUTES LES FEATURES EN NUMPY
        Élimine le bottleneck pandas .iloc dans la boucle d'entraînement
        """
        self.precomputed = {}
        
        for ticker in self.tickers:
            df = self.data[ticker]
            
            # Convertir en numpy arrays
            close = df['Close'].values
            volume = df['Volume'].values
            high = df['High'].values if 'High' in df.columns else close
            low = df['Low'].values if 'Low' in df.columns else close
            
            # Close normalisé (rolling mean/std sur 20 périodes)
            close_norm = np.zeros_like(close)
            for i in range(20, len(close)):
                window = close[max(0, i-20):i]
                mean = np.mean(window)
                std = np.std(window)
                close_norm[i] = (close[i] - mean) / (std + 1e-8)
            
            # Volume normalisé
            volume_norm = np.zeros_like(volume, dtype=np.float32)
            for i in range(20, len(volume)):
                window = volume[max(0, i-20):i]
                mean = np.mean(window)
                std = np.std(window)
                volume_norm[i] = (volume[i] - mean) / (std + 1e-8)
            
            # Returns
            returns_1d = np.zeros_like(close)
            returns_1d[1:] = (close[1:] - close[:-1]) / close[:-1]
            
            returns_5d = np.zeros_like(close)
            returns_5d[5:] = (close[5:] - close[:-5]) / close[:-5]
            
            # RSI (14 périodes)
            rsi = np.zeros_like(close)
            diff = np.diff(close, prepend=close[0])
            gains = np.where(diff > 0, diff, 0)
            losses = np.where(diff < 0, -diff, 0)
            
            for i in range(14, len(close)):
                avg_gain = np.mean(gains[max(0, i-14):i])
                avg_loss = np.mean(losses[max(0, i-14):i])
                
                if avg_loss == 0:
                    rsi[i] = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi[i] = 100 - (100 / (1 + rs))
            
            rsi_norm = (rsi - 50) / 50
            
            # MACD (12, 26)
            macd = np.zeros_like(close)
            if len(close) > 26:
                ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
                ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
                macd = (ema12 - ema26) / close
            
            # Stocker tout
            self.precomputed[ticker] = {
                'close': close,
                'volume': volume,
                'close_norm': close_norm.astype(np.float32),
                'volume_norm': volume_norm.astype(np.float32),
                'rsi_norm': rsi_norm.astype(np.float32),
                'macd': macd.astype(np.float32),
                'returns_1d': returns_1d.astype(np.float32),
                'returns_5d': returns_5d.astype(np.float32)
            }
    
    def reset(self, seed=None, options=None):
        """Reset environnement"""
        super().reset(seed=seed)
        
        # Point de départ aléatoire
        self.current_step = np.random.randint(100, self.data_length - self.max_steps)
        self.reset_step = self.current_step
        
        self.balance = self.initial_balance
        self.positions = {ticker: 0 for ticker in self.tickers}
        self.portfolio_value = self.initial_balance
        self.trades_history = []
        self.transaction_costs_history = []
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Exécuter une action"""
        self.current_step += 1
        
        # Prix actuels (accès numpy direct)
        current_prices = {
            ticker: float(self.precomputed[ticker]['close'][self.current_step])
            for ticker in self.tickers
        }
        
        # Valeur portfolio AVANT trade
        positions_value = sum(
            self.positions[ticker] * current_prices[ticker]
            for ticker in self.tickers
        )
        previous_portfolio_value = self.balance + positions_value
        
        # Exécuter actions
        for i, ticker in enumerate(self.tickers):
            target_position = action[i]
            target_value = target_position * previous_portfolio_value * 0.95 / self.n_assets
            
            current_value = self.positions[ticker] * current_prices[ticker]
            trade_value = target_value - current_value
            
            if abs(trade_value) > previous_portfolio_value * 0.01:
                shares_to_trade = int(trade_value / current_prices[ticker])
                
                if shares_to_trade != 0:
                    # Coûts de transaction
                    if self.realistic_costs:
                        execution_price, costs = self.transaction_model.calculate_execution_price(
                            ticker=ticker,
                            intended_price=current_prices[ticker],
                            order_size=abs(shares_to_trade),
                            current_volume=float(self.precomputed[ticker]['volume'][self.current_step]),
                            side='buy' if shares_to_trade > 0 else 'sell',
                            recent_prices=self.precomputed[ticker]['close'][max(0, self.current_step-20):self.current_step]
                        )
                        total_cost = costs['total_cost_dollars']
                        
                        self.transaction_costs_history.append({
                            'step': self.current_step,
                            'ticker': ticker,
                            'shares': abs(shares_to_trade),
                            'intended_price': current_prices[ticker],
                            'execution_price': execution_price,
                            'slippage': costs['slippage'],
                            'market_impact': costs['market_impact'],
                            'total_cost': total_cost
                        })
                    else:
                        execution_price = current_prices[ticker]
                        cost = abs(shares_to_trade * current_prices[ticker])
                        total_cost = cost * self.commission
                    
                    # Exécuter trade
                    if shares_to_trade > 0:  # Achat
                        total_needed = abs(shares_to_trade * execution_price) + total_cost
                        
                        if self.balance >= total_needed:
                            self.positions[ticker] += shares_to_trade
                            self.balance -= total_needed
                            self.trades_history.append({
                                'step': self.current_step,
                                'ticker': ticker,
                                'action': 'BUY',
                                'shares': shares_to_trade,
                                'price': execution_price,
                                'cost': total_cost
                            })
                    else:  # Vente
                        if self.positions[ticker] >= abs(shares_to_trade):
                            self.positions[ticker] += shares_to_trade
                            proceeds = abs(shares_to_trade * execution_price) - total_cost
                            self.balance += proceeds
                            self.trades_history.append({
                                'step': self.current_step,
                                'ticker': ticker,
                                'action': 'SELL',
                                'shares': abs(shares_to_trade),
                                'price': execution_price,
                                'cost': total_cost
                            })
        
        # Recalculer valeur finale APRÈS trade
        positions_value = sum(
            self.positions[ticker] * current_prices[ticker]
            for ticker in self.tickers
        )
        new_portfolio_value = self.balance + positions_value
        
        # ✅ FIX REWARD FUNCTION (CRITIQUE)
        # Normaliser par initial_balance, pas portfolio_value
        reward = (new_portfolio_value - previous_portfolio_value) / self.initial_balance
        
        # ✅ SAFETY: Clip reward pour éviter divergence
        reward = np.clip(reward, -0.1, 0.1)
        
        self.portfolio_value = new_portfolio_value
        
        # ✅ SAFETY: Terminer si portfolio détruit
        if new_portfolio_value <= 0:
            reward = -1.0
            terminated = True
        else:
            terminated = (
                self.current_step >= self.data_length - 1 or
                new_portfolio_value <= self.initial_balance * 0.1
            )
        
        truncated = (self.current_step - self.reset_step) >= self.max_steps
        
        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'positions': self.positions.copy(),
            'n_trades': len(self.trades_history)
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self):
        """
        ✅ OBSERVATION ULTRA-RAPIDE (numpy pur)
        Plus de pandas .iloc dans la hot path !
        """
        obs = []
        idx = self.current_step
        
        # Features par asset (accès direct numpy)
        for ticker in self.tickers:
            precomp = self.precomputed[ticker]
            
            obs.extend([
                precomp['close_norm'][idx],
                precomp['volume_norm'][idx],
                precomp['rsi_norm'][idx],
                precomp['macd'][idx],
                precomp['returns_1d'][idx],
                precomp['returns_5d'][idx]
            ])
        
        # Portfolio features
        cash_ratio = self.balance / self.portfolio_value if self.portfolio_value > 0 else 0
        total_value_norm = (self.portfolio_value - self.initial_balance) / self.initial_balance
        n_positions = sum(1 for pos in self.positions.values() if pos > 0) / self.n_assets
        
        obs.extend([
            float(cash_ratio),
            float(total_value_norm),
            float(n_positions)
        ])
        
        # Convertir en array et remplacer NaN/Inf
        obs = np.array(obs, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs
    
    def get_transaction_costs_summary(self):
        """Récupère statistiques sur les coûts de transaction"""
        if len(self.transaction_costs_history) == 0:
            return {'avg_slippage': 0, 'avg_impact': 0, 'total_costs': 0}
        
        df = pd.DataFrame(self.transaction_costs_history)
        
        return {
            'avg_slippage_pct': df['slippage'].mean() * 100,
            'avg_market_impact_pct': df['market_impact'].mean() * 100,
            'total_costs_dollars': df['total_cost'].sum(),
            'n_trades': len(df),
            'avg_cost_per_trade': df['total_cost'].mean()
        }
    
    def render(self, mode='human'):
        """Afficher état"""
        if mode == 'human':
            print(f"\n{'='*60}")
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Cash: ${self.balance:,.2f}")
            print(f"Positions: {self.positions}")
            print(f"Total Trades: {len(self.trades_history)}")
            
            if self.realistic_costs and len(self.transaction_costs_history) > 0:
                costs = self.get_transaction_costs_summary()
                print(f"\nTransaction Costs:")
                print(f"  Avg Slippage: {costs['avg_slippage_pct']:.3f}%")
                print(f"  Avg Impact: {costs['avg_market_impact_pct']:.3f}%")
                print(f"  Total Costs: ${costs['total_costs_dollars']:.2f}")
            
            print(f"{'='*60}")
