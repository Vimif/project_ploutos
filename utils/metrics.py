"""Calculs de métriques financières"""

import numpy as np
import pandas as pd
from typing import List, Dict

def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252 * 6.5  # Heures de trading
) -> float:
    """
    Calcule le Sharpe ratio
    
    Args:
        returns: Array des returns
        risk_free_rate: Taux sans risque
        periods_per_year: Nombre de périodes par an
        
    Returns:
        Sharpe ratio annualisé
    """
    if len(returns) == 0:
        return 0.0
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0.0
    
    sharpe = (mean_return - risk_free_rate) / std_return
    return sharpe * np.sqrt(periods_per_year)

def calculate_max_drawdown(values: np.ndarray) -> float:
    """
    Calcule le drawdown maximum
    
    Args:
        values: Array des valeurs du portfolio
        
    Returns:
        Max drawdown (valeur négative)
    """
    if len(values) == 0:
        return 0.0
    
    cumulative = np.array(values)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    
    return float(np.min(drawdown))

def calculate_profit_factor(returns: np.ndarray) -> float:
    """
    Calcule le profit factor
    
    Args:
        returns: Array des returns
        
    Returns:
        Profit factor (gains / pertes)
    """
    if len(returns) == 0:
        return 0.0
    
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    
    if losses == 0:
        return float('inf') if gains > 0 else 0.0
    
    return gains / losses

def calculate_win_rate(returns: np.ndarray) -> float:
    """
    Calcule le taux de réussite
    
    Args:
        returns: Array des returns
        
    Returns:
        Win rate (0-1)
    """
    if len(returns) == 0:
        return 0.0
    
    return float((returns > 0).sum() / len(returns))

def calculate_all_metrics(values: List[float]) -> Dict[str, float]:
    """
    Calcule toutes les métriques d'un coup
    
    Args:
        values: Liste des valeurs du portfolio
        
    Returns:
        Dict avec toutes les métriques
    """
    if not values:
        return {
            'total_return': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0
        }
    
    # Convertir en numpy
    values_arr = np.array(values)
    returns = np.diff(values_arr) / values_arr[:-1]
    
    # Calculer métriques
    initial = values[0]
    final = values[-1]
    total_return = (final - initial) / initial * 100
    
    return {
        'total_return': float(total_return),
        'sharpe': calculate_sharpe_ratio(returns),
        'max_drawdown': calculate_max_drawdown(values_arr) * 100,
        'win_rate': calculate_win_rate(returns) * 100,
        'profit_factor': calculate_profit_factor(returns),
        'final_value': float(final)
    }
