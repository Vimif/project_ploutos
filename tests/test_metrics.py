"""Tests métriques"""

import numpy as np
from utils.metrics import (
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_win_rate,
    calculate_all_metrics
)

def test_sharpe_ratio():
    """Test calcul Sharpe"""
    returns = np.array([0.01, 0.02, -0.01, 0.03, 0.01])
    sharpe = calculate_sharpe_ratio(returns)
    
    assert isinstance(sharpe, float)
    assert sharpe > 0  # Returns positifs en moyenne

def test_max_drawdown():
    """Test calcul drawdown"""
    values = np.array([100, 110, 105, 95, 100, 105])
    dd = calculate_max_drawdown(values)
    
    assert isinstance(dd, float)
    assert dd < 0  # Drawdown toujours négatif

def test_profit_factor():
    """Test profit factor"""
    returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
    pf = calculate_profit_factor(returns)
    
    assert isinstance(pf, float)
    assert pf > 0

def test_win_rate():
    """Test win rate"""
    returns = np.array([0.01, -0.01, 0.02, -0.02, 0.03])
    wr = calculate_win_rate(returns)
    
    assert 0 <= wr <= 1
    assert wr == 0.6  # 3/5 = 60%

def test_all_metrics():
    """Test calcul groupé"""
    values = [100, 102, 101, 105, 104, 107]
    metrics = calculate_all_metrics(values)
    
    assert 'sharpe' in metrics
    assert 'max_drawdown' in metrics
    assert 'total_return' in metrics
    assert metrics['total_return'] > 0  # Gains nets
