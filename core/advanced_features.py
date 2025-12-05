#!/usr/bin/env python3
"""
Advanced Features : Regime Detection + Reward Shaping
Module combiné pour Phase 2 best practices
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy import stats

# ============================================================================
# MARKET REGIME DETECTION
# ============================================================================

class MarketRegimeDetector:
    """
    Détecte le régime de marché actuel (Bull/Bear/Sideways)
    """
    
    def __init__(self, trend_threshold=0.015, vol_threshold=0.025, window=50):
        self.trend_threshold = trend_threshold
        self.vol_threshold = vol_threshold
        self.window = window
        self.regime_history = []
        
    def detect(self, prices: pd.Series) -> Dict:
        """Détecte régime actuel"""
        
        if len(prices) < self.window:
            return {'regime': 'UNKNOWN', 'confidence': 0.0}
        
        recent = prices.iloc[-self.window:].values
        
        # Tendance (régression linéaire)
        x = np.arange(len(recent))
        slope, _, r_value, _, _ = stats.linregress(x, recent)
        trend = slope / recent.mean()
        
        # Volatilité
        returns = np.diff(recent) / recent[:-1]
        volatility = returns.std()
        
        # Momentum
        momentum = (recent[-1] - recent[0]) / recent[0]
        
        # Classification
        regime, confidence = self._classify(trend, volatility, momentum, r_value**2)
        
        self.regime_history.append(regime)
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]
        
        return {'regime': regime, 'confidence': confidence}
    
    def _classify(self, trend, volatility, momentum, r_squared):
        """Classifie régime"""
        
        confidence = 0.5
        
        if trend > self.trend_threshold and momentum > 0.05:
            regime = 'BULL'
            if volatility < self.vol_threshold:
                confidence += 0.2
            if r_squared > 0.7:
                confidence += 0.2
                
        elif trend < -self.trend_threshold and momentum < -0.05:
            regime = 'BEAR'
            if volatility > self.vol_threshold * 1.5:
                confidence += 0.2
            if r_squared > 0.6:
                confidence += 0.2
                
        else:
            regime = 'SIDEWAYS'
            if abs(trend) < self.trend_threshold / 2:
                confidence += 0.2
            if volatility < self.vol_threshold * 0.8:
                confidence += 0.1
        
        return regime, float(np.clip(confidence, 0.0, 1.0))

# ============================================================================
# REWARD SHAPING
# ============================================================================

class AdvancedRewardCalculator:
    """
    Reward function sophistiquée multi-objectif
    
    Encourage :
    - Returns positifs
    - Sharpe élevé (stabilité)
    - Faible drawdown
    - Peu de trades (réduire coûts)
    """
    
    def __init__(self, 
                 return_weight=1.0,
                 volatility_weight=0.1,
                 drawdown_weight=0.5,
                 sharpe_weight=0.2,
                 trading_weight=0.01):
        """
        Args:
            *_weight: Pondération de chaque composante
        """
        self.return_weight = return_weight
        self.volatility_weight = volatility_weight
        self.drawdown_weight = drawdown_weight
        self.sharpe_weight = sharpe_weight
        self.trading_weight = trading_weight
        
    def calculate(self, 
                  prev_value: float,
                  new_value: float,
                  action: np.ndarray,
                  portfolio_history: list) -> float:
        """
        Calcule reward composé
        
        Args:
            prev_value: Valeur portfolio précédente
            new_value: Valeur portfolio actuelle
            action: Action prise (pour pénalité overtrading)
            portfolio_history: Historique valeurs
            
        Returns:
            reward: Reward total
        """
        
        # 1. Return de base
        return_reward = (new_value - prev_value) / prev_value
        return_reward *= self.return_weight
        
        # 2. Pénalité volatilité
        volatility_penalty = 0
        if len(portfolio_history) > 10:
            returns = pd.Series(portfolio_history).pct_change().dropna()
            vol = returns.std()
            volatility_penalty = -self.volatility_weight * vol
        
        # 3. Pénalité drawdown
        drawdown_penalty = 0
        if len(portfolio_history) > 0:
            peak = max(portfolio_history)
            drawdown = (new_value - peak) / peak
            if drawdown < -0.05:  # Seuil 5%
                drawdown_penalty = self.drawdown_weight * drawdown
        
        # 4. Bonus Sharpe
        sharpe_bonus = 0
        if len(portfolio_history) > 50:
            sharpe = self._rolling_sharpe(portfolio_history, window=50)
            if sharpe > 1.0:
                sharpe_bonus = self.sharpe_weight * (sharpe - 1.0)
        
        # 5. Pénalité overtrading
        trading_penalty = -self.trading_weight * abs(action).sum()
        
        # Total
        total_reward = (
            return_reward +
            volatility_penalty +
            drawdown_penalty +
            sharpe_bonus +
            trading_penalty
        )
        
        return float(total_reward)
    
    def _rolling_sharpe(self, values, window=50):
        """Calcule Sharpe glissant"""
        recent = pd.Series(values[-window:])
        returns = recent.pct_change().dropna()
        
        if returns.std() == 0:
            return 0
        
        return (returns.mean() / returns.std()) * np.sqrt(252)

# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 ADVANCED FEATURES - Phase 2")
    print("="*80 + "\n")
    
    print("📊 1. MARKET REGIME DETECTOR")
    print("-" * 60)
    
    # Test regime detector
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    bull_prices = pd.Series(100 + np.cumsum(np.random.randn(200) * 2 + 0.5), index=dates)
    
    detector = MarketRegimeDetector()
    result = detector.detect(bull_prices)
    
    print(f"  Régime détecté : {result['regime']}")
    print(f"  Confidence      : {result['confidence']:.2f}\n")
    
    print("🎯 2. REWARD SHAPING")
    print("-" * 60)
    
    # Test reward calculator
    calc = AdvancedRewardCalculator()
    
    portfolio_hist = [10000, 10050, 10100, 10080, 10120]
    action = np.array([0.5, -0.3, 0.2])
    
    reward = calc.calculate(
        prev_value=10100,
        new_value=10120,
        action=action,
        portfolio_history=portfolio_hist
    )
    
    print(f"  Reward calculé  : {reward:.6f}")
    print(f"  (positif = bon, négatif = mauvais)\n")
    
    print("✅ Intégration dans universal_environment.py :")
    print("""
    # Dans __init__()
    from core.advanced_features import MarketRegimeDetector, AdvancedRewardCalculator
    
    self.regime_detector = MarketRegimeDetector()
    self.reward_calculator = AdvancedRewardCalculator()
    
    # Dans step()
    # Détecter régime
    regime_info = self.regime_detector.detect(self.data[ticker]['Close'])
    
    # Calculer reward avancé
    reward = self.reward_calculator.calculate(
        prev_value=self.portfolio_value,
        new_value=new_portfolio_value,
        action=action,
        portfolio_history=self.portfolio_history
    )
    """)
    
    print("\n" + "="*80)
    print("✅ Module prêt")
    print("="*80 + "\n")
