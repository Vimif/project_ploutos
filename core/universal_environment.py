"""
Environnement de trading universel pour multi-assets
Avec coûts de transaction réalistes (slippage + impact marché)
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
    print("⚠️  transaction_costs.py non trouvé, utilisation coûts simplifiés")
    REALISTIC_COSTS = False

class UniversalTradingEnv(gym.Env):
    """Environnement générique pour trader un portfolio multi-assets"""

    metadata = {'render_modes': ['human']}

    def __init__(self, data, initial_balance=100000, commission=0.001, max_steps=1000, realistic_costs=True):
        """
        Args:
            data (dict): {ticker: DataFrame} avec colonnes [Open, High, Low, Close, Volume]
            initial_balance (float): Capital initial
            commission (float): Commission par trade (0.001 = 0.1%)
            max_steps (int): Nombre max de steps par épisode
            realistic_costs (bool): Utiliser modèle de coûts avancé (slippage + impact marché)
        """
        super().__init__()
        
        self.data = data
        self.tickers = list(data.keys())
        self.n_assets = len(self.tickers)
        self.initial_balance = initial_balance
        self.commission = commission
        self.max_steps = max_steps
        self.realistic_costs = realistic_costs and REALISTIC_COSTS
        
        # ✅ NOUVEAU : Modèle de coûts réalistes
        if self.realistic_costs:
            self.transaction_model = AdvancedTransactionModel(
                base_commission=commission,
                min_slippage=0.0005,   # 0.05%
                max_slippage=0.005,    # 0.5%
                market_impact_coef=0.0001,
                latency_std=0.0002
            )
            print("✅ Environnement avec coûts réalistes (slippage + impact marché)")
        else:
            self.transaction_model = None
            print("⚠️  Environnement avec coûts simplifiés")
        
        # Vérifier que tous les DataFrames ont la même longueur
        self.data_length = min(len(df) for df in data.values())
        
        # Observation space : features par asset + portfolio state
        # Features par asset : [close_norm, volume_norm, rsi, macd, returns_1d, returns_5d]
        n_features_per_asset = 6
        n_portfolio_features = 3  # [cash_ratio, total_value_norm, n_positions]
        
        obs_size = self.n_assets * n_features_per_asset + n_portfolio_features
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_size,), 
            dtype=np.float32
        )
        
        # Action space : position pour chaque asset [-1, 1]
        # -1 = short max, 0 = neutral, 1 = long max
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
        self.positions = {ticker: 0 for ticker in self.tickers}  # Nombre d'actions
        self.portfolio_value = initial_balance
        self.trades_history = []
        self.transaction_costs_history = []  # ✅ NOUVEAU : Historique coûts
        
    def reset(self, seed=None, options=None):
        """Reset environnement"""
        super().reset(seed=seed)
        
        # Choisir point de départ aléatoire (laisser 100 steps pour calcul features)
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
        
        # Récupérer prix actuels
        current_prices = {
            ticker: float(self.data[ticker].iloc[self.current_step]['Close'])
            for ticker in self.tickers
        }
        
        # Calculer valeur actuelle du portfolio
        positions_value = sum(
            self.positions[ticker] * current_prices[ticker]
            for ticker in self.tickers
        )
        self.portfolio_value = self.balance + positions_value
        
        # Exécuter actions (ajuster positions selon action)
        for i, ticker in enumerate(self.tickers):
            target_position = action[i]  # -1 à 1
            
            # Convertir en valeur monétaire cible
            target_value = target_position * self.portfolio_value * 0.95 / self.n_assets
            
            current_value = self.positions[ticker] * current_prices[ticker]
            trade_value = target_value - current_value
            
            # Exécuter trade si significatif
            if abs(trade_value) > self.portfolio_value * 0.01:  # Min 1% portfolio
                shares_to_trade = int(trade_value / current_prices[ticker])
                
                if shares_to_trade != 0:
                    # ✅ NOUVEAU : Coûts réalistes avec slippage
                    if self.realistic_costs:
                        execution_price, costs = self.transaction_model.calculate_execution_price(
                            ticker=ticker,
                            intended_price=current_prices[ticker],
                            order_size=abs(shares_to_trade),
                            current_volume=float(self.data[ticker].iloc[self.current_step]['Volume']),
                            side='buy' if shares_to_trade > 0 else 'sell',
                            recent_prices=self.data[ticker]['Close'].iloc[max(0, self.current_step-20):self.current_step]
                        )
                        
                        total_cost = costs['total_cost_dollars']
                        
                        # Logger coûts
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
                        # Coûts simplifiés (ancien système)
                        execution_price = current_prices[ticker]
                        cost = abs(shares_to_trade * current_prices[ticker])
                        total_cost = cost * self.commission
                    
                    # Vérifier si assez de cash
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
                            self.positions[ticker] += shares_to_trade  # shares_to_trade est négatif
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
        
        # Recalculer valeur finale
        positions_value = sum(
            self.positions[ticker] * current_prices[ticker]
            for ticker in self.tickers
        )
        new_portfolio_value = self.balance + positions_value
        
        # Calculer reward (variation en %)
        reward = (new_portfolio_value - self.portfolio_value) / self.portfolio_value
        
        self.portfolio_value = new_portfolio_value
        
        # Terminer si max_steps atteint ou portfolio <= 0
        terminated = (
            self.current_step >= self.data_length - 1 or
            self.portfolio_value <= self.initial_balance * 0.1
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
        """Construire observation"""
        obs = []
        
        # Features par asset
        for ticker in self.tickers:
            df = self.data[ticker]
            idx = self.current_step
            
            # Prix normalisé
            close = df.iloc[idx]['Close']
            close_norm = (close - df['Close'].iloc[max(0, idx-20):idx].mean()) / (df['Close'].iloc[max(0, idx-20):idx].std() + 1e-8)
            
            # Volume normalisé
            volume = df.iloc[idx]['Volume']
            volume_norm = (volume - df['Volume'].iloc[max(0, idx-20):idx].mean()) / (df['Volume'].iloc[max(0, idx-20):idx].std() + 1e-8)
            
            # Returns
            returns_1d = (df.iloc[idx]['Close'] - df.iloc[idx-1]['Close']) / df.iloc[idx-1]['Close'] if idx > 0 else 0
            returns_5d = (df.iloc[idx]['Close'] - df.iloc[idx-5]['Close']) / df.iloc[idx-5]['Close'] if idx > 5 else 0
            
            # RSI simplifié
            gains = df['Close'].diff().clip(lower=0).iloc[max(0, idx-14):idx].mean()
            losses = -df['Close'].diff().clip(upper=0).iloc[max(0, idx-14):idx].mean()
            rsi = 100 - (100 / (1 + gains / (losses + 1e-8)))
            rsi_norm = (rsi - 50) / 50
            
            # MACD approximatif
            ema12 = df['Close'].iloc[max(0, idx-12):idx].ewm(span=12).mean().iloc[-1] if idx > 12 else close
            ema26 = df['Close'].iloc[max(0, idx-26):idx].ewm(span=26).mean().iloc[-1] if idx > 26 else close
            macd = (ema12 - ema26) / close
            
            obs.extend([
                float(close_norm),
                float(volume_norm),
                float(rsi_norm),
                float(macd),
                float(returns_1d),
                float(returns_5d)
            ])
        
        # Portfolio features
        cash_ratio = self.balance / self.portfolio_value
        total_value_norm = (self.portfolio_value - self.initial_balance) / self.initial_balance
        n_positions = sum(1 for pos in self.positions.values() if pos > 0) / self.n_assets
        
        obs.extend([
            float(cash_ratio),
            float(total_value_norm),
            float(n_positions)
        ])
        
        # Remplacer NaN/Inf par 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return np.array(obs, dtype=np.float32)
    
    def get_transaction_costs_summary(self):
        """
        ✅ NOUVEAU : Récupère statistiques sur les coûts de transaction
        
        Returns:
            dict avec métriques de coûts
        """
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
