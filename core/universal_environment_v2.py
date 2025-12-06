#!/usr/bin/env python3
"""
🎯 UNIVERSAL TRADING ENVIRONMENT V2

VERSION FIXÉE avec les découvertes du debug (6 déc 2025)

✅ FIXES APPLIQUÉS:
1. Action space: Continuous → MultiDiscrete (BUY/HOLD/SELL)
2. Reward: Portfolio variation → PnL réalisé + PnL latent
3. Tracking entry prices par ticker (FIFO avec deque)
4. Vente forcée à la fin (truncation)

Résultats attendus:
- L'IA trade activement (BUY + SELL > 10%)
- Portfolio > $102k
- Sharpe > 0.5
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from collections import deque

# Import du modèle de coûts réalistes
try:
    from core.transaction_costs import AdvancedTransactionModel
    REALISTIC_COSTS = True
except ImportError:
    REALISTIC_COSTS = False

class UniversalTradingEnvV2(gym.Env):
    """
    Environnement multi-assets avec actions discrètes et reward PnL
    
    Actions par ticker:
    - 0 = HOLD  (ne rien faire)
    - 1 = BUY   (acheter 20% du portfolio)
    - 2 = SELL  (vendre toute la position)
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, data, initial_balance=100000, commission=0.0001, max_steps=2000, 
                 buy_pct=0.2, realistic_costs=False):
        """
        Args:
            data (dict): {ticker: DataFrame} avec colonnes [Open, High, Low, Close, Volume]
            initial_balance (float): Capital initial
            commission (float): Commission par trade (0.0001 = 0.01%)
            max_steps (int): Nombre max de steps par épisode
            buy_pct (float): % du portfolio à investir par BUY (0.2 = 20%)
            realistic_costs (bool): Utiliser modèle de coûts avancé
        """
        super().__init__()
        
        self.data = data
        self.tickers = list(data.keys())
        self.n_assets = len(self.tickers)
        self.initial_balance = initial_balance
        self.commission = commission
        self.max_steps = max_steps
        self.buy_pct = buy_pct
        self.realistic_costs = realistic_costs and REALISTIC_COSTS
        
        # Modèle de coûts
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
        
        # ✅ PRÉ-CALCUL DES FEATURES
        print("\n⚡ Pré-calcul des features V2...")
        self._precompute_features()
        print("✅ Features pré-calculées !\n")
        
        # Observation space (inchangé)
        n_features_per_asset = 6
        n_portfolio_features = 3
        obs_size = self.n_assets * n_features_per_asset + n_portfolio_features
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_size,), 
            dtype=np.float32
        )
        
        # ✅★★★ NOUVEAU: ACTION SPACE DISCRET ★★★
        self.action_space = spaces.MultiDiscrete([3] * self.n_assets)
        # [3, 3, 3, ...] = 3 actions possibles par ticker
        # 0 = HOLD, 1 = BUY, 2 = SELL
        
        # État interne
        self.current_step = 0
        self.reset_step = 0
        self.balance = initial_balance
        self.positions = {ticker: 0 for ticker in self.tickers}
        self.portfolio_value = initial_balance
        self.trades_history = []
        self.transaction_costs_history = []
        
        # ✅★★★ NOUVEAU: TRACKING PNL PAR TICKER ★★★
        self.entry_prices = {ticker: deque() for ticker in self.tickers}
        self.entry_steps = {ticker: None for ticker in self.tickers}
        self.total_pnl = 0.0
        
        # Tracking portfolio
        self.portfolio_history = []
        self.peak_portfolio_value = initial_balance
    
    def _precompute_features(self):
        """Pré-calcule toutes les features en numpy"""
        self.precomputed = {}
        
        for ticker in self.tickers:
            df = self.data[ticker]
            
            close = df['Close'].values
            volume = df['Volume'].values
            
            # Close normalisé
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
            returns_1d[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-8)
            
            returns_5d = np.zeros_like(close)
            returns_5d[5:] = (close[5:] - close[:-5]) / (close[:-5] + 1e-8)
            
            # RSI
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
            
            # MACD
            macd = np.zeros_like(close)
            if len(close) > 26:
                ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
                ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
                macd = (ema12 - ema26) / (close + 1e-8)
            
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
        
        # ✅ Reset tracking PnL
        for ticker in self.tickers:
            self.entry_prices[ticker].clear()
            self.entry_steps[ticker] = None
        self.total_pnl = 0.0
        
        self.portfolio_history = [self.initial_balance]
        self.peak_portfolio_value = self.initial_balance
        
        return self._get_observation(), {}
    
    def step(self, actions):
        """
        Exécute les actions discrètes pour chaque ticker
        
        Args:
            actions: array of ints, un par ticker (0=HOLD, 1=BUY, 2=SELL)
        """
        self.current_step += 1
        
        # Prix actuels
        current_prices = {
            ticker: float(self.precomputed[ticker]['close'][self.current_step])
            for ticker in self.tickers
        }
        
        # Valeur portfolio AVANT trades
        positions_value = sum(
            self.positions[ticker] * current_prices[ticker]
            for ticker in self.tickers
        )
        previous_portfolio_value = self.balance + positions_value
        
        # ★★★ EXÉCUTER ACTIONS POUR CHAQUE TICKER ★★★
        total_reward = 0.0
        
        for i, ticker in enumerate(self.tickers):
            action = int(actions[i])
            current_price = current_prices[ticker]
            
            # Calculer reward pour ce ticker
            reward_ticker = self._execute_action(
                ticker, action, current_price, previous_portfolio_value
            )
            total_reward += reward_ticker
        
        # Recalculer valeur APRÈS trades
        positions_value = sum(
            self.positions[ticker] * current_prices[ticker]
            for ticker in self.tickers
        )
        new_portfolio_value = self.balance + positions_value
        self.portfolio_value = new_portfolio_value
        
        # Update tracking
        self.portfolio_history.append(new_portfolio_value)
        self.peak_portfolio_value = max(self.peak_portfolio_value, new_portfolio_value)
        
        # Clip reward total
        total_reward = np.clip(total_reward, -0.5, 0.5)
        
        # Termination
        if new_portfolio_value <= 0:
            total_reward = -1.0
            terminated = True
        elif new_portfolio_value <= self.initial_balance * 0.1:
            total_reward = -0.5
            terminated = True
        else:
            terminated = self.current_step >= self.data_length - 1
        
        truncated = (self.current_step - self.reset_step) >= self.max_steps
        
        # ✅★★★ VENTE FORCÉE À LA FIN ★★★
        if (terminated or truncated):
            for ticker in self.tickers:
                if self.positions[ticker] > 0 and len(self.entry_prices[ticker]) > 0:
                    # Calculer PnL final
                    current_price = current_prices[ticker]
                    avg_entry = np.mean(list(self.entry_prices[ticker]))
                    final_pnl = (current_price - avg_entry) / avg_entry
                    
                    # Ajouter au reward
                    total_reward += final_pnl
                    
                    # Vendre (simulation)
                    proceeds = self.positions[ticker] * current_price
                    proceeds *= (1 - self.commission)
                    self.balance += proceeds
                    self.positions[ticker] = 0
                    self.portfolio_value = self.balance
        
        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'positions': self.positions.copy(),
            'n_trades': len(self.trades_history),
            'peak_value': self.peak_portfolio_value,
            'total_pnl': self.total_pnl
        }
        
        return self._get_observation(), total_reward, terminated, truncated, info
    
    def _execute_action(self, ticker, action, current_price, portfolio_value):
        """
        Exécute une action pour un ticker spécifique
        
        Returns:
            reward (float): Reward pour cette action
        """
        reward = 0.0
        
        # ★ ACTION 1: BUY ★
        if action == 1:
            # Investir buy_pct% du portfolio dans ce ticker
            investment = self.balance * self.buy_pct
            
            if investment > 0 and current_price > 0:
                shares_to_buy = int(investment / current_price)
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    fee = cost * self.commission
                    total = cost + fee
                    
                    if self.balance >= total:
                        # Exécuter achat
                        self.positions[ticker] += shares_to_buy
                        self.balance -= total
                        
                        # Enregistrer prix d'entrée
                        for _ in range(shares_to_buy):
                            self.entry_prices[ticker].append(current_price)
                        
                        if self.entry_steps[ticker] is None:
                            self.entry_steps[ticker] = self.current_step
                        
                        # ✅ Reward = 0 lors du BUY
                        reward = 0.0
                        
                        self.trades_history.append({
                            'step': self.current_step,
                            'ticker': ticker,
                            'action': 'BUY',
                            'shares': shares_to_buy,
                            'price': current_price,
                            'cost': fee
                        })
        
        # ★ ACTION 2: SELL ★
        elif action == 2:
            if self.positions[ticker] > 0 and len(self.entry_prices[ticker]) > 0:
                shares_to_sell = self.positions[ticker]
                proceeds = shares_to_sell * current_price
                fee = proceeds * self.commission
                
                # ✅ CALCULER PNL RÉALISÉ
                pnl_total = 0.0
                for _ in range(shares_to_sell):
                    if len(self.entry_prices[ticker]) > 0:
                        entry_price = self.entry_prices[ticker].popleft()
                        pnl = (current_price - entry_price) / entry_price
                        pnl_total += pnl
                
                avg_pnl = pnl_total / shares_to_sell if shares_to_sell > 0 else 0
                
                # ✅ REWARD = PNL MOYEN DU TRADE
                reward = avg_pnl
                
                # Exécuter vente
                self.balance += (proceeds - fee)
                self.positions[ticker] = 0
                self.entry_steps[ticker] = None
                
                self.total_pnl += avg_pnl
                
                self.trades_history.append({
                    'step': self.current_step,
                    'ticker': ticker,
                    'action': 'SELL',
                    'shares': shares_to_sell,
                    'price': current_price,
                    'cost': fee,
                    'pnl': avg_pnl
                })
        
        # ★ ACTION 0: HOLD ★
        else:
            # Reward sur PnL latent si on tient une position
            if self.positions[ticker] > 0 and len(self.entry_prices[ticker]) > 0:
                avg_entry = np.mean(list(self.entry_prices[ticker]))
                unrealized_pnl = (current_price - avg_entry) / avg_entry
                
                # ✅ Petit reward (0.5% du PnL latent)
                reward = unrealized_pnl * 0.005
        
        # Clip reward individuel
        reward = np.clip(reward, -0.3, 0.3)
        
        return reward
    
    def _get_observation(self):
        """Observation (inchangée)"""
        obs = []
        idx = self.current_step
        
        # Features par asset
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
        
        obs = np.array(obs, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs
    
    def render(self, mode='human'):
        """Afficher état"""
        if mode == 'human':
            print(f"\n{'='*60}")
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Peak Value: ${self.peak_portfolio_value:,.2f}")
            print(f"Cash: ${self.balance:,.2f}")
            print(f"Total PnL: {self.total_pnl:.4f}")
            print(f"Positions:")
            for ticker, shares in self.positions.items():
                if shares > 0:
                    current_price = self.precomputed[ticker]['close'][self.current_step]
                    value = shares * current_price
                    if len(self.entry_prices[ticker]) > 0:
                        avg_entry = np.mean(list(self.entry_prices[ticker]))
                        pnl = (current_price - avg_entry) / avg_entry * 100
                        print(f"  {ticker}: {shares} shares @ ${current_price:.2f} (PnL: {pnl:+.2f}%)")
            print(f"Total Trades: {len(self.trades_history)}")
            print(f"={'='*60}")
