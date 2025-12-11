#!/usr/bin/env python3
"""
Observation Builder V7 - Structure d'observation 3D pour Transformer
====================================================================

Problématé résolu:
- Véritable structure temporelle (lookback dimension)
- Compatible Transformer + Attention
- Features groupées logiquement

Structure:
  obs_temporal: (n_tickers, lookback, n_features)  - Features temporelles
  obs_portfolio: (n_tickers,)                      - Position de chaque asset
  obs_account: (3,)                               - Cash, return, drawdown

Total: (n_tickers * lookback * n_features + n_tickers + 3,)
"""

import numpy as np
from typing import Dict, List, Tuple
import pandas as pd


class ObservationBuilderV7:
    """Construction d'observation 3D optimisée pour Transformer."""
    
    def __init__(
        self,
        n_tickers: int,
        lookback: int,
        feature_columns: List[str],
        normalize: bool = True,
    ):
        """
        Initialize observation builder.
        
        Args:
            n_tickers: Nombre d'assets
            lookback: Nombre de steps historiques
            feature_columns: Liste des colonnes de features
            normalize: Si True, normaliser les features
        """
        self.n_tickers = n_tickers
        self.lookback = lookback
        self.feature_columns = feature_columns
        self.n_features = len(feature_columns)
        self.normalize = normalize
        
        # Cache pour std/mean (pour normalisation)
        self.feature_stats = {}
    
    def fit(
        self,
        processed_data: Dict[str, pd.DataFrame],
        tickers: List[str],
    ) -> None:
        """
        Fit normalizer statistics on historical data.
        
        Args:
            processed_data: Dict[ticker] = DataFrame avec features
            tickers: Liste des tickers
        """
        if not self.normalize:
            return
        
        # Collect all features
        all_features = []
        for ticker in tickers:
            df = processed_data[ticker]
            features = df[self.feature_columns].values
            all_features.append(features)
        
        all_features = np.vstack(all_features)
        
        # Compute stats per feature
        self.feature_stats['mean'] = np.nanmean(all_features, axis=0)
        self.feature_stats['std'] = np.nanstd(all_features, axis=0)
        
        # Avoid division by zero
        self.feature_stats['std'][self.feature_stats['std'] < 1e-8] = 1.0
    
    def build_observation(
        self,
        processed_data: Dict[str, pd.DataFrame],
        tickers: List[str],
        current_step: int,
        portfolio: Dict[str, float],
        balance: float,
        equity: float,
        initial_balance: float,
        peak_value: float,
    ) -> np.ndarray:
        """
        Construire observation 3D + portfolio + account state.
        
        Args:
            processed_data: Dict[ticker] = DataFrame
            tickers: Liste des tickers
            current_step: Step actuel
            portfolio: Dict[ticker] = quantity
            balance: Cash disponible
            equity: Valeur totale
            initial_balance: Capital initial
            peak_value: Peak equity atteint
        
        Returns:
            Observation 1D flatten
        """
        
        # ===== PART 1: Temporal Features (3D) =====
        obs_temporal = self._build_temporal_features(
            processed_data, tickers, current_step
        )
        
        # ===== PART 2: Portfolio State =====
        obs_portfolio = self._build_portfolio_state(
            processed_data, tickers, current_step, portfolio, equity
        )
        
        # ===== PART 3: Account State =====
        obs_account = self._build_account_state(
            equity, initial_balance, peak_value, balance
        )
        
        # ===== Combine all parts =====
        obs = np.concatenate([
            obs_temporal.flatten(),  # (n_tickers * lookback * n_features)
            obs_portfolio,           # (n_tickers)
            obs_account,             # (3)
        ])
        
        return obs.astype(np.float32)
    
    def _build_temporal_features(
        self,
        processed_data: Dict[str, pd.DataFrame],
        tickers: List[str],
        current_step: int,
    ) -> np.ndarray:
        """
        Build 3D temporal feature matrix.
        
        Shape: (n_tickers, lookback, n_features)
        """
        obs_temporal = np.zeros(
            (self.n_tickers, self.lookback, self.n_features),
            dtype=np.float32
        )
        
        for ticker_idx, ticker in enumerate(tickers):
            df = processed_data[ticker]
            
            # Historique: t-lookback jusqu'à t
            for t in range(self.lookback):
                idx = current_step - self.lookback + t
                
                if 0 <= idx < len(df):
                    row = df.iloc[idx]
                    features = row[self.feature_columns].values
                else:
                    # Out of range: zero padding
                    features = np.zeros(self.n_features)
                
                # Process features
                features = self._process_features(features)
                
                obs_temporal[ticker_idx, t, :] = features
        
        return obs_temporal
    
    def _build_portfolio_state(
        self,
        processed_data: Dict[str, pd.DataFrame],
        tickers: List[str],
        current_step: int,
        portfolio: Dict[str, float],
        equity: float,
    ) -> np.ndarray:
        """
        Build portfolio state vector.
        
        Shape: (n_tickers,)
        Value: Position as % of equity
        """
        obs_portfolio = np.zeros(self.n_tickers, dtype=np.float32)
        
        for ticker_idx, ticker in enumerate(tickers):
            price = self._get_current_price(
                processed_data[ticker], current_step
            )
            
            if price > 0:
                position_value = portfolio[ticker] * price
                position_pct = position_value / (equity + 1e-8)
            else:
                position_pct = 0.0
            
            # Clip to [0, 1]
            obs_portfolio[ticker_idx] = np.clip(position_pct, 0, 1)
        
        return obs_portfolio
    
    def _build_account_state(
        self,
        equity: float,
        initial_balance: float,
        peak_value: float,
        balance: float,
    ) -> np.ndarray:
        """
        Build account state vector.
        
        Shape: (3,)
        Values: [cash_pct, total_return, drawdown]
        """
        cash_pct = np.clip(balance / (equity + 1e-8), 0, 1)
        total_return = np.clip(
            (equity - initial_balance) / (initial_balance + 1e-8),
            -1, 5
        )
        drawdown = np.clip(
            (peak_value - equity) / (peak_value + 1e-8),
            0, 1
        )
        
        return np.array(
            [cash_pct, total_return, drawdown],
            dtype=np.float32
        )
    
    def _process_features(self, features: np.ndarray) -> np.ndarray:
        """
        Process and normalize features.
        
        Args:
            features: Raw features array
        
        Returns:
            Processed features clipped to [-10, 10]
        """
        # Handle NaN and Inf
        features = np.nan_to_num(
            features, nan=0.0, posinf=10.0, neginf=-10.0
        )
        
        # Normalize if fitted
        if self.normalize and self.feature_stats:
            mean = self.feature_stats.get('mean', 0)
            std = self.feature_stats.get('std', 1)
            features = (features - mean) / (std + 1e-8)
        
        # Clip to reasonable range
        features = np.clip(features, -10, 10)
        
        return features
    
    def _get_current_price(
        self,
        df: pd.DataFrame,
        current_step: int,
    ) -> float:
        """
        Get current price for a ticker.
        
        Args:
            df: DataFrame for ticker
            current_step: Current step index
        
        Returns:
            Current close price
        """
        if current_step >= len(df):
            return float(df.iloc[-1]['Close'])
        
        price = df.iloc[current_step]['Close']
        
        # Handle invalid prices
        if price <= 0 or np.isnan(price) or np.isinf(price):
            # Fallback to median
            return float(df['Close'].median())
        
        return float(price)
    
    def get_observation_space_size(self) -> int:
        """
        Get total observation space size.
        
        Returns:
            obs_size = n_tickers * lookback * n_features + n_tickers + 3
        """
        return (
            self.n_tickers * self.lookback * self.n_features +
            self.n_tickers +
            3
        )
